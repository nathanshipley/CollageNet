"""
Core rendering machinery for the img2img + projection experiment.

Extracted from img2img_experiment.ipynb so it can be shared between the
notebook (sandbox / debugging) and app.py (Gradio web UI). Nothing here
is project-source — it lives on the experiment-img2img branch only.

Top-level surface:
    CollageEngine               - holds loaded SDXL pipelines + patch bank
    LatentRandomSquareRegionProjector - quadtree segmentation projector
    snap8, side_by_side, label_image, format_label_text,
        format_val_for_filename - pure helpers
    DEFAULT_RENDER_PARAMS       - sensible starting values
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
from diffusers import StableDiffusionXLImg2ImgPipeline
from PIL import Image, ImageDraw, ImageFont

# Ensure the project root is importable regardless of where this file is
# imported from (notebook in cwd vs. app.py launched from elsewhere).
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.append(str(_PROJECT_ROOT))

import patch_dictionary_core  # noqa: E402
from patch_dictionary_core import (  # noqa: E402
    LatentFelzenszwalbRegionProjector,
    latent_labels_to_debug_image,
)
from render_config import GenerationConfig, ModelConfig, ProjectionConfig  # noqa: E402
from render_runtime import (  # noqa: E402
    DEVICE,
    build_runtime_patch_bank,
    load_pipeline,
    make_generator,
    make_projector_for_cfg,
    set_seed,
)
from source_latents import prepare_latents_from_images  # noqa: E402


# --- Random-square projector -------------------------------------------------

class LatentRandomSquareRegionProjector(LatentFelzenszwalbRegionProjector):
    """Tile the latent canvas with a quadtree of mixed-size squares.

    Subclass of the felzenszwalb projector that overrides only the segmentation
    step. Sizes are powers of 2 (2, 4, 8, 16, ...); `min_size` is the smallest
    tile allowed and `split_prob` is the chance of subdividing at each
    recursion level.
    """

    def __init__(self, cfg, patch_bank, *, min_size: int = 2, split_prob: float = 0.65):
        super().__init__(cfg=cfg, patch_bank=patch_bank)
        self.random_square_min_size = max(1, int(min_size))
        self.random_square_split_prob = float(min(max(split_prob, 0.0), 1.0))

    def compute_felzenszwalb_labels(self, latents):
        _, _, h, w = latents.shape
        labels = np.full((h, w), -1, dtype=np.int32)
        next_label = [0]
        rng = self.rng
        min_size = self.random_square_min_size
        split_prob = self.random_square_split_prob

        def assign(y0, y1, x0, x1):
            ph = y1 - y0
            pw = x1 - x0
            if ph <= min_size or pw <= min_size or rng.random() > split_prob:
                labels[y0:y1, x0:x1] = next_label[0]
                next_label[0] += 1
                return
            ymid = y0 + ph // 2
            xmid = x0 + pw // 2
            assign(y0, ymid, x0, xmid)
            assign(y0, ymid, xmid, x1)
            assign(ymid, y1, x0, xmid)
            assign(ymid, y1, xmid, x1)

        assign(0, h, 0, w)
        return labels, {
            "random_square_min_size_used": float(min_size),
            "random_square_split_prob_used": float(split_prob),
            "num_random_squares": int(next_label[0]),
        }

    def maybe_save_debug_labels(self, labels, step_index, timestep):
        if self.debug_output_dir is None:
            return
        every_n = max(1, int(getattr(self.cfg, "debug_every_n_projections", 1)))
        if len(self.projection_events) % every_n != 0:
            return
        debug_image = latent_labels_to_debug_image(labels, seed=int(step_index))
        debug_path = self.debug_output_dir / f"random_square_step_{step_index:03d}_t{int(timestep):04d}.png"
        debug_image = debug_image.resize(
            (labels.shape[1] * 4, labels.shape[0] * 4),
            Image.Resampling.NEAREST,
        )
        debug_image.save(debug_path)


# --- Pure helpers ------------------------------------------------------------

def snap8(v: int) -> int:
    return max(8, v - (v % 8))


def side_by_side(left: Image.Image, right: Image.Image) -> Image.Image:
    """Compose two images side-by-side; left is rescaled to match right's size."""
    if left.size != right.size:
        left = left.resize(right.size, Image.Resampling.LANCZOS)
    w, h = right.size
    combined = Image.new('RGB', (w * 2, h), (255, 255, 255))
    combined.paste(left, (0, 0))
    combined.paste(right, (w, 0))
    return combined


def load_font(size: int):
    """Try a few common monospace font paths; fall back to PIL default."""
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/Library/Fonts/Menlo.ttc",
        "C:/Windows/Fonts/consola.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def label_image(img: Image.Image, text: str) -> Image.Image:
    """Draw multi-line text in a small dark box at top-left; returns a copy."""
    base = img.convert('RGBA')
    overlay = Image.new('RGBA', base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = load_font(20)
    pad, margin = 10, 12
    bbox = draw.multiline_textbbox((margin, margin), text, font=font, spacing=4)
    box = (bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad)
    draw.rectangle(box, fill=(0, 0, 0, 200))
    draw.multiline_text((margin, margin), text, fill=(255, 255, 255, 255),
                        font=font, spacing=4)
    return Image.alpha_composite(base, overlay).convert('RGB')


def format_val_for_filename(val) -> str:
    if isinstance(val, float):
        return f"{val:.2f}".replace('.', '_')
    return str(val).replace(' ', '_').replace('/', '_')


def format_label_text(p: dict, highlight: str | None = None) -> str:
    """Multi-line settings overlay text drawn on labeled images."""
    lines = []
    if highlight:
        lines.append(f">>> {highlight} <<<")
    lines.append(f"seed={p['seed']}")
    lines.append(
        f"skip={p['skip']}  free={p['free']}  proj={p['proj']}  "
        f"total={p['skip'] + p['free'] + p['proj']}"
    )
    lines.append(f"strength={p['strength']:.3f}  proj_start={p['proj_start_frac']:.3f}")
    lines.append(f"alpha_end={p['alpha_end']:.2f}")
    lines.append(f"region={p['region_method']}  patch={p['patch_size']}")
    if p['region_method'] == 'random_square':
        lines.append(
            f"rs: min_size={p.get('rs_min_size', '?')}  "
            f"split_prob={p.get('rs_split_prob', 0.0):.2f}"
        )
    elif p['region_method'] == 'felzenszwalb':
        lines.append(
            f"felz: scale={p['felz_scale']:.1f}  sigma={p['felz_sigma']:.2f}  "
            f"min={p['felz_min_size']}"
        )
    prompt_trunc = p['prompt'] if len(p['prompt']) <= 60 else p['prompt'][:57] + '...'
    lines.append(f"prompt: {prompt_trunc}")
    return '\n'.join(lines)


# --- Defaults ----------------------------------------------------------------

DEFAULT_GEN_CFG = GenerationConfig(
    prompt="a portrait on a white background.",
    negative_prompt="blurry, low quality, deformed, extra limbs, text, watermark",
    height=1024,
    width=1024,
    num_inference_steps=30,
    guidance_scale=6.5,
    seed=1,
)

_DEFAULT_PATCH_CFG_BASE = dict(
    region_method="felzenszwalb",
    patch_size=1,
    do_rotated=True,
    total_patches=50000,
    projection_start_frac=0.6,
    projection_end_frac=1.0,
    alpha_start=0.0,
    alpha_end=0.10,
    region_candidate_count=128,
    region_min_area=1,
    region_max_area=1200,
    felzenszwalb_scale=8.0,
    felzenszwalb_sigma=2.4,
    felzenszwalb_min_size=12,
)

DEFAULT_RENDER_PARAMS = dict(
    prompt="a portrait on a white background.",
    skip=4,
    free=12,
    proj=9,
    alpha_end=0.10,
    region_method='felzenszwalb',
    patch_size=1,
    felz_scale=8.0,
    felz_sigma=2.4,
    felz_min_size=12,
    rs_min_size=2,
    rs_split_prob=0.65,
    seed=1,
)


# --- Engine ------------------------------------------------------------------

class CollageEngine:
    """Loaded SDXL pipelines + a patch bank derived from user source images.

    Lifecycle:
        1. __init__ loads the SDXL text2img and img2img pipelines (~30s).
        2. load_source_images(folder) encodes images and builds the patch bank.
        3. render_one(...) renders one combo, returning (original, diffusion,
           sharp_or_None, params_dict).
    """

    def __init__(self, model_cfg: ModelConfig | None = None,
                 gen_cfg: GenerationConfig | None = None):
        self.model_cfg = model_cfg or ModelConfig()
        self.gen_cfg = gen_cfg or DEFAULT_GEN_CFG

        print(f"Loading SDXL pipeline (device={DEVICE})...")
        self.pipe = load_pipeline(self.model_cfg)
        self.img2img_pipe = StableDiffusionXLImg2ImgPipeline.from_pipe(self.pipe)
        print("Pipelines ready.")

        self.patch_cfg: ProjectionConfig | None = None
        self.patch_bank = None
        self._current_bank_params: dict | None = None
        self._source_dir: Path | None = None

    @property
    def is_ready(self) -> bool:
        return self.patch_bank is not None

    def load_source_images(
        self,
        source_dir: Path,
        *,
        max_dim: int = 1536,
        latents_dir: Path | None = None,
    ) -> str:
        """Encode (if needed) a folder of images and build the patch bank.

        Returns a short status string for UI display.
        """
        source_dir = Path(source_dir).expanduser()
        if not source_dir.is_dir():
            raise FileNotFoundError(f"Source images directory not found: {source_dir}")

        if latents_dir is None:
            latents_dir = _PROJECT_ROOT / "user-latents" / source_dir.name

        had_existing = latents_dir.exists() and any(latents_dir.glob("*.npz"))
        if not had_existing:
            latents_dir.mkdir(parents=True, exist_ok=True)
            print(f"Encoding {source_dir} -> {latents_dir} ...")
            prepare_latents_from_images(
                input_dir=source_dir,
                output_dir=latents_dir,
                vae=self.pipe.vae,
                device=DEVICE,
                max_width=max_dim,
                max_height=max_dim,
                skip_existing=True,
            )

        self.patch_cfg = ProjectionConfig(latent_dir=latents_dir, **_DEFAULT_PATCH_CFG_BASE)
        print("Building patch bank...")
        self.patch_bank = build_runtime_patch_bank(self.patch_cfg)
        self._current_bank_params = {
            'region_method': self.patch_cfg.region_method,
            'patch_size': self.patch_cfg.patch_size,
            'do_rotated': self.patch_cfg.do_rotated,
            'total_patches': self.patch_cfg.total_patches,
        }
        self._source_dir = source_dir

        n_patches = int(self.patch_bank.raw_patches.shape[0])
        n_files = len(self.patch_bank.source_counts)
        return (
            f"Patch bank: {n_patches} patches across {n_files} files "
            f"({'reused cached latents' if had_existing else 'encoded fresh'})"
        )

    def ensure_source_loaded(self, source_dir: Path, *, max_dim: int = 1536) -> str | None:
        """No-op if `source_dir` is already loaded; otherwise (re)build the bank.

        Returns a status string if a load happened, None if it was a no-op.
        """
        source_dir = Path(source_dir).expanduser()
        if self.is_ready and self._source_dir == source_dir:
            return None
        return self.load_source_images(source_dir, max_dim=max_dim)

    def ensure_bank(self, region_method: str, patch_size: int) -> None:
        """Rebuild the patch bank if region_method or patch_size changed."""
        if self.patch_cfg is None:
            raise RuntimeError("Call load_source_images() first.")
        # Region methods always use patch_size=1 in the bank — patches are
        # individual latent cells, regions are made by combining them.
        effective_patch_size = patch_size if region_method == 'square' else 1
        requested = {
            'region_method': region_method,
            'patch_size': effective_patch_size,
            'do_rotated': self.patch_cfg.do_rotated,
            'total_patches': self.patch_cfg.total_patches,
        }
        if requested == self._current_bank_params:
            return
        rebuild_cfg = replace(
            self.patch_cfg,
            region_method=region_method,
            patch_size=effective_patch_size,
        )
        print(f"Rebuilding patch bank: {requested}")
        self.patch_bank = build_runtime_patch_bank(rebuild_cfg)
        self._current_bank_params = requested

    def render_one(
        self,
        *,
        image_path,
        prompt,
        skip,
        free,
        proj,
        alpha_end,
        region_method,
        patch_size,
        felz_scale,
        felz_sigma,
        felz_min_size,
        rs_min_size,
        rs_split_prob,
        seed,
    ) -> tuple[Image.Image, Image.Image, Image.Image | None, dict]:
        """Render one combo. Returns (original, diffusion, sharp_or_None, params)."""
        if not self.is_ready:
            raise RuntimeError("Call load_source_images() first.")

        total = int(skip) + int(free) + int(proj)
        if total <= 0:
            raise ValueError("Need at least one step.")

        strength = min(max((free + proj) / total, 1e-3), 1.0)
        proj_start_frac = min(max((skip + free) / total, 0.0), 1.0)

        self.ensure_bank(region_method, int(patch_size))

        original_input = Image.open(image_path).convert("RGB")
        target_w = snap8(self.gen_cfg.width)
        target_h = snap8(self.gen_cfg.height)
        init_image = original_input.resize((target_w, target_h), Image.Resampling.LANCZOS)

        live_patch_cfg = replace(
            self.patch_cfg,
            # random_square is dispatched manually below; the cfg's
            # projector_mode mapping doesn't know about it, so pretend it's
            # felzenszwalb for any internal lookups.
            region_method=region_method if region_method != 'random_square' else 'felzenszwalb',
            patch_size=int(patch_size) if region_method == 'square' else 1,
            projection_start_frac=proj_start_frac,
            alpha_end=float(alpha_end),
            felzenszwalb_scale=float(felz_scale),
            felzenszwalb_sigma=float(felz_sigma),
            felzenszwalb_min_size=int(felz_min_size),
        )

        if region_method == 'random_square':
            projector = LatentRandomSquareRegionProjector(
                cfg=live_patch_cfg, patch_bank=self.patch_bank,
                min_size=int(rs_min_size), split_prob=float(rs_split_prob),
            )
        else:
            projector = make_projector_for_cfg(live_patch_cfg, self.patch_bank)

        set_seed(int(seed))
        result = self.img2img_pipe(
            prompt=prompt,
            negative_prompt=self.gen_cfg.negative_prompt,
            image=init_image,
            strength=strength,
            num_inference_steps=total,
            guidance_scale=self.gen_cfg.guidance_scale,
            generator=make_generator(int(seed)),
            callback_on_step_end=projector,
            callback_on_step_end_tensor_inputs=["latents"],
        )
        diffusion_image = result.images[0]

        sharp_image = None
        if (
            getattr(projector, 'final_region_assignments', None) is not None
            and getattr(projector, 'final_assignment_grid_shape', None) is not None
        ):
            sharp_image = patch_dictionary_core.render_pixel_collage_from_region_assignments(
                patch_bank=self.patch_bank,
                region_assignments=projector.final_region_assignments,
                latent_canvas_shape=projector.final_assignment_grid_shape,
                pixel_render_scale=1,
            )
        elif (
            getattr(projector, 'final_selected_patch_indices', None) is not None
            and getattr(projector, 'final_assignment_grid_shape', None) is not None
        ):
            sharp_image = patch_dictionary_core.render_pixel_collage_from_assignments(
                patch_bank=self.patch_bank,
                selected_patch_indices=projector.final_selected_patch_indices.numpy()[0],
                grid_shape=projector.final_assignment_grid_shape,
                patch_size=live_patch_cfg.patch_size,
                pixel_render_scale=1,
            )

        params = {
            'prompt': prompt, 'seed': int(seed),
            'skip': int(skip), 'free': int(free), 'proj': int(proj),
            'strength': strength, 'proj_start_frac': proj_start_frac,
            'alpha_end': float(alpha_end),
            'region_method': region_method, 'patch_size': int(patch_size),
            'felz_scale': float(felz_scale),
            'felz_sigma': float(felz_sigma),
            'felz_min_size': int(felz_min_size),
            'rs_min_size': int(rs_min_size),
            'rs_split_prob': float(rs_split_prob),
        }
        return original_input, diffusion_image, sharp_image, params
