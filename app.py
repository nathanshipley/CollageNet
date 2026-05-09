"""
Gradio web UI for the CollageNet img2img experiment.

Run with:
    pip install gradio
    python app.py

Opens at http://localhost:7860 by default. Pass `--share` to get a public URL.

Layout:
    - Top: source images folder + input image + prompt + seed
    - Shared knobs (step counts, region method, felz/rs sub-knobs)
    - Tabs: Render | Sweep | Party

Source images are loaded lazily — the first action (Render / Sweep / Party)
that uses them will trigger a build of the patch bank if it hasn't been
built yet (or if the folder path has changed). Cached encodings under
`user-latents/<folder-name>/` are reused on subsequent runs.

Outputs land in `outputs/img2img_experiment/...` next to this file.
"""

from __future__ import annotations

import argparse
import random
from datetime import datetime
from pathlib import Path

import gradio as gr
from PIL import Image

from experiment_core import (
    DEFAULT_RENDER_PARAMS,
    CollageEngine,
    format_label_text,
    format_val_for_filename,
    label_image,
    side_by_side,
)


_HERE = Path(__file__).resolve().parent
OUTPUT_DIR = _HERE / "outputs" / "img2img_experiment"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Long edge cap when encoding source images. Edit if your GPU is tight or
# you want richer per-image latents.
SOURCE_MAX_DIM = 1536


# --- Engine: loaded once at import time -------------------------------------

print("Initializing CollageEngine...")
engine = CollageEngine()


# --- Helpers shared by the three render callbacks ---------------------------

def _ensure_inputs_ok(image_path, source_dir):
    """Validate the always-required inputs. Returns an error string or None."""
    if not source_dir:
        return "Set the source images folder path at the top first."
    if not image_path:
        return "Upload an input image first."
    return None


def _load_if_needed(source_dir, progress=None):
    """Lazy-load source images. Returns a status prefix to prepend to info text."""
    src = Path(source_dir).expanduser()
    if engine.is_ready and engine._source_dir == src:
        return ""
    if progress is not None:
        progress(0, desc="Loading source images / building patch bank...")
    status = engine.ensure_source_loaded(src, max_dim=SOURCE_MAX_DIM)
    return f"{status}\n" if status else ""


# --- Render tab callback -----------------------------------------------------

def on_render(image_path, source_dir, prompt, skip, free, proj, alpha_end,
              region_method, patch_size,
              felz_scale, felz_sigma, felz_min_size,
              rs_min_size, rs_split_prob, seed,
              progress=gr.Progress()):
    err = _ensure_inputs_ok(image_path, source_dir)
    if err:
        return None, None, None, err

    try:
        load_status = _load_if_needed(source_dir, progress)
    except Exception as e:
        return None, None, None, f"Could not load source images: {e}"

    progress(0.5, desc="Rendering...")
    try:
        original, diffusion, sharp, params = engine.render_one(
            image_path=image_path, prompt=prompt,
            skip=skip, free=free, proj=proj, alpha_end=alpha_end,
            region_method=region_method, patch_size=patch_size,
            felz_scale=felz_scale, felz_sigma=felz_sigma, felz_min_size=felz_min_size,
            rs_min_size=rs_min_size, rs_split_prob=rs_split_prob,
            seed=seed,
        )
    except Exception as e:
        return None, None, None, f"{load_status}Error: {e}"

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    input_stem = Path(image_path).stem
    run_dir = OUTPUT_DIR / f"{timestamp}_seed{int(seed):04d}_{input_stem}"
    run_dir.mkdir(parents=True, exist_ok=True)

    diffusion.save(run_dir / 'diffusion.png')
    if sharp is not None:
        sbs = side_by_side(original, sharp)
        sbs.save(run_dir / 'sidebyside.png')
        sharp.save(run_dir / 'sharp.png')
        primary = sbs
        sharp_alone = sharp
    else:
        primary = side_by_side(original, diffusion)
        primary.save(run_dir / 'sidebyside.png')
        sharp_alone = None

    info = (
        f"{load_status}"
        f"Saved -> outputs/img2img_experiment/{run_dir.name}\n"
        f"steps: skip={params['skip']} free={params['free']} proj={params['proj']}  "
        f"strength={params['strength']:.3f}  proj_start_frac={params['proj_start_frac']:.3f}  "
        f"alpha_end={params['alpha_end']:.2f}  seed={params['seed']}"
    )
    return primary, sharp_alone, diffusion, info


# --- Sweep tab callback ------------------------------------------------------

# Sweep schedule: vary one knob low/mid/high. Edit in code if you want different values.
SWEEP_SCHEDULE = [
    ('alpha_end', [0.05, 0.20, 0.50]),
    ('skip', [0, 9, 20]),
    ('proj', [3, 12, 24]),
    ('region_method', ['felzenszwalb', 'threshold', 'square', 'random_square']),
    ('felz_min_size', [2, 12, 32]),
    ('rs_split_prob', [0.30, 0.60, 0.90]),
]


def on_sweep(image_path, source_dir, prompt, skip, free, proj, alpha_end,
             region_method, patch_size,
             felz_scale, felz_sigma, felz_min_size,
             rs_min_size, rs_split_prob, seed,
             progress=gr.Progress()):
    err = _ensure_inputs_ok(image_path, source_dir)
    if err:
        return [], err

    try:
        load_status = _load_if_needed(source_dir, progress)
    except Exception as e:
        return [], f"Could not load source images: {e}"

    baseline = dict(
        image_path=image_path, prompt=prompt,
        skip=skip, free=free, proj=proj, alpha_end=alpha_end,
        region_method=region_method, patch_size=patch_size,
        felz_scale=felz_scale, felz_sigma=felz_sigma, felz_min_size=felz_min_size,
        rs_min_size=rs_min_size, rs_split_prob=rs_split_prob,
        seed=seed,
    )

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    sweep_dir = OUTPUT_DIR / f"sweep_{timestamp}_{Path(image_path).stem}"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    total_renders = 1 + sum(len(values) for _, values in SWEEP_SCHEDULE)
    gallery: list[tuple[Image.Image, str]] = []

    progress(0, desc=f"baseline (1/{total_renders})")
    try:
        original, diffusion, sharp, params = engine.render_one(**baseline)
    except Exception as e:
        return [], f"{load_status}Baseline error: {e}"

    diffusion.save(sweep_dir / 'diffusion__00_baseline.png')
    if sharp is not None:
        sbs = side_by_side(original, sharp)
        sbs.save(sweep_dir / 'sidebyside__00_baseline.png')
        sharp.save(sweep_dir / 'sharp__00_baseline.png')
        labeled = label_image(sharp, format_label_text(params, 'baseline'))
        labeled.save(sweep_dir / 'labeled__00_baseline.png')
        gallery.append((labeled, "baseline"))

    idx = 1
    for sweep_idx, (var_name, values) in enumerate(SWEEP_SCHEDULE, start=1):
        for val in values:
            idx += 1
            variant = dict(baseline)
            variant[var_name] = val
            stem = f"{sweep_idx:02d}_{var_name}_{format_val_for_filename(val)}"
            progress(idx / total_renders, desc=f"{stem} ({idx}/{total_renders})")

            try:
                _, variant_diffusion, variant_sharp, variant_params = engine.render_one(**variant)
            except Exception as e:
                gallery.append((Image.new('RGB', (256, 256), 'red'), f"{stem}: {e}"))
                continue

            variant_diffusion.save(sweep_dir / f"diffusion__{stem}.png")
            if variant_sharp is None:
                continue
            variant_sharp.save(sweep_dir / f"sharp__{stem}.png")
            labeled = label_image(
                variant_sharp,
                format_label_text(variant_params, f"{var_name}={val}"),
            )
            labeled.save(sweep_dir / f"labeled__{stem}.png")
            gallery.append((labeled, f"{var_name}={val}"))

    (sweep_dir / 'sweep_info.txt').write_text(
        "baseline (knobs at sweep time):\n"
        + '\n'.join(f"  {k}: {v}" for k, v in baseline.items())
        + "\n\nsweeps:\n"
        + '\n'.join(f"  {name}: {values}" for name, values in SWEEP_SCHEDULE)
        + "\n"
    )

    info = (
        f"{load_status}"
        f"Saved {total_renders} renders -> outputs/img2img_experiment/{sweep_dir.name}\n"
        f"Files grouped by type prefix: diffusion__ / labeled__ / sharp__ / sidebyside__"
    )
    return gallery, info


# --- Party tab callback ------------------------------------------------------

PARTY_RANGES = {
    'skip':         (0, 25),
    'free':         (4, 25),
    'proj':         (0, 25),
    'alpha_end':    (0.05, 0.60),
    'patch_size':   [1, 2, 4, 8],          # legal grid factors only
    'felz_scale':   (4.0, 50.0),
    'felz_sigma':   (0.5, 3.5),
    'felz_min_size':(2, 32),
    'rs_min_size':  (1, 8),
    'rs_split_prob':(0.30, 0.90),
    'region_method':['felzenszwalb', 'threshold', 'square', 'random_square'],
}


def _roll_combo(rng, image_path, prompt, fixed_region_method):
    region = fixed_region_method or rng.choice(PARTY_RANGES['region_method'])
    return dict(
        image_path=image_path,
        prompt=prompt,
        skip=rng.randint(*PARTY_RANGES['skip']),
        free=rng.randint(*PARTY_RANGES['free']),
        proj=rng.randint(*PARTY_RANGES['proj']),
        alpha_end=round(rng.uniform(*PARTY_RANGES['alpha_end']), 2),
        region_method=region,
        patch_size=rng.choice(PARTY_RANGES['patch_size']),
        felz_scale=round(rng.uniform(*PARTY_RANGES['felz_scale']), 1),
        felz_sigma=round(rng.uniform(*PARTY_RANGES['felz_sigma']), 2),
        felz_min_size=rng.randint(*PARTY_RANGES['felz_min_size']),
        rs_min_size=rng.randint(*PARTY_RANGES['rs_min_size']),
        rs_split_prob=round(rng.uniform(*PARTY_RANGES['rs_split_prob']), 2),
        seed=rng.randint(0, 2**31 - 1),
    )


def on_party(image_path, source_dir, prompt, party_count, party_region_pin,
             progress=gr.Progress()):
    err = _ensure_inputs_ok(image_path, source_dir)
    if err:
        return [], err

    try:
        load_status = _load_if_needed(source_dir, progress)
    except Exception as e:
        return [], f"Could not load source images: {e}"

    fixed_region = None if party_region_pin == 'randomize' else party_region_pin
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    region_tag = fixed_region or 'mixed'
    party_dir = OUTPUT_DIR / f"party_{timestamp}_{region_tag}_{Path(image_path).stem}"
    party_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random()
    gallery: list[tuple[Image.Image, str]] = []
    log_lines = ["idx,seed,skip,free,proj,alpha_end,region_method,patch_size,felz_scale,felz_sigma,felz_min_size,rs_min_size,rs_split_prob"]

    N = int(party_count)
    for i in range(1, N + 1):
        combo = _roll_combo(rng, image_path, prompt, fixed_region)
        stem = f"{i:03d}"
        progress(i / N, desc=f"#{stem}  region={combo['region_method']}  seed={combo['seed']}")

        try:
            _, diffusion, sharp, params = engine.render_one(**combo)
        except Exception as e:
            gallery.append((Image.new('RGB', (256, 256), 'red'), f"#{stem}: {e}"))
            continue

        diffusion.save(party_dir / f"diffusion__{stem}.png")
        label_text = format_label_text(params, f"party #{stem}")
        if sharp is not None:
            sharp.save(party_dir / f"sharp__{stem}.png")
            labeled = label_image(sharp, label_text)
        else:
            labeled = label_image(diffusion, label_text)
        labeled.save(party_dir / f"labeled__{stem}.png")
        gallery.append((labeled, f"#{stem}  {combo['region_method']}"))

        log_lines.append(
            f"{stem},{combo['seed']},{combo['skip']},{combo['free']},{combo['proj']},"
            f"{combo['alpha_end']:.2f},{combo['region_method']},{combo['patch_size']},"
            f"{combo['felz_scale']:.1f},{combo['felz_sigma']:.2f},{combo['felz_min_size']},"
            f"{combo['rs_min_size']},{combo['rs_split_prob']:.2f}"
        )

    (party_dir / 'party_log.csv').write_text('\n'.join(log_lines) + '\n')
    info = (
        f"{load_status}"
        f"Saved {N} renders + party_log.csv -> outputs/img2img_experiment/{party_dir.name}\n"
        f"{'pinned to ' + fixed_region if fixed_region else 'region_method randomized per render'}"
    )
    return gallery, info


# --- Live derived-params readout --------------------------------------------

def _refresh_derived(skip, free, proj):
    total = int(skip) + int(free) + int(proj)
    if total <= 0:
        return "<i>need at least one step</i>"
    strength = (free + proj) / total
    proj_start = (skip + free) / total
    return (
        f"<code>num_inference_steps={total} &nbsp; "
        f"strength={strength:.3f} &nbsp; "
        f"projection_start_frac={proj_start:.3f}</code>"
    )


# --- UI layout ---------------------------------------------------------------

with gr.Blocks(title="CollageNet Experiment") as app:
    gr.Markdown("# CollageNet — img2img + projection experiment")

    source_dir_input = gr.Textbox(
        label="Source images folder",
        placeholder="path/to/your/source/images/folder",
        info="Loaded automatically the first time you render (and re-loaded if you change the path). Cached encodings live under user-latents/.",
    )

    with gr.Row():
        image_input = gr.Image(label="Input image", type="filepath", height=320)
        with gr.Column():
            prompt_box = gr.Textbox(
                label="Prompt",
                value=DEFAULT_RENDER_PARAMS['prompt'],
                lines=3,
            )
            seed_input = gr.Number(value=DEFAULT_RENDER_PARAMS['seed'], label="seed", precision=0)

    with gr.Accordion("Step counts", open=True):
        with gr.Row():
            skip_slider = gr.Slider(0, 29, value=DEFAULT_RENDER_PARAMS['skip'], step=1, label="skip_steps (img2img)")
            free_slider = gr.Slider(1, 40, value=DEFAULT_RENDER_PARAMS['free'], step=1, label="text2img_steps")
            proj_slider = gr.Slider(0, 40, value=DEFAULT_RENDER_PARAMS['proj'], step=1, label="projection_steps")
        alpha_end_slider = gr.Slider(0, 1, value=DEFAULT_RENDER_PARAMS['alpha_end'], step=0.05, label="projection alpha_end")
        derived_html = gr.HTML(value=_refresh_derived(
            DEFAULT_RENDER_PARAMS['skip'],
            DEFAULT_RENDER_PARAMS['free'],
            DEFAULT_RENDER_PARAMS['proj'],
        ))
        for w in (skip_slider, free_slider, proj_slider):
            w.change(_refresh_derived, [skip_slider, free_slider, proj_slider], derived_html)

    with gr.Accordion("Region method", open=True):
        region_dropdown = gr.Dropdown(
            ['felzenszwalb', 'threshold', 'square', 'random_square'],
            value=DEFAULT_RENDER_PARAMS['region_method'],
            label="region_method",
        )
        patch_size_dropdown = gr.Dropdown(
            [1, 2, 4, 8, 16],
            value=DEFAULT_RENDER_PARAMS['patch_size'],
            label="patch_size (square only)",
        )

    with gr.Accordion("Felzenszwalb knobs", open=False):
        felz_scale_slider = gr.Slider(1, 100, value=DEFAULT_RENDER_PARAMS['felz_scale'], step=1, label="felz_scale")
        felz_sigma_slider = gr.Slider(0.1, 4.0, value=DEFAULT_RENDER_PARAMS['felz_sigma'], step=0.1, label="felz_sigma")
        felz_min_size_slider = gr.Slider(1, 64, value=DEFAULT_RENDER_PARAMS['felz_min_size'], step=1, label="felz_min_size")

    with gr.Accordion("Random-square knobs", open=False):
        rs_min_size_slider = gr.Slider(1, 16, value=DEFAULT_RENDER_PARAMS['rs_min_size'], step=1, label="rs_min_size")
        rs_split_prob_slider = gr.Slider(0, 1, value=DEFAULT_RENDER_PARAMS['rs_split_prob'], step=0.05, label="rs_split_prob")

    # Group all the shared inputs once for reuse below.
    shared_inputs = [
        image_input, source_dir_input, prompt_box,
        skip_slider, free_slider, proj_slider, alpha_end_slider,
        region_dropdown, patch_size_dropdown,
        felz_scale_slider, felz_sigma_slider, felz_min_size_slider,
        rs_min_size_slider, rs_split_prob_slider,
        seed_input,
    ]

    with gr.Tabs():
        with gr.TabItem("Render"):
            render_btn = gr.Button("Render", variant="primary")
            primary_output = gr.Image(label="Input | Sharp collage", type="pil", height=512)
            with gr.Row():
                sharp_alone_output = gr.Image(label="Sharp collage", type="pil", height=512)
                diffusion_output = gr.Image(label="Diffusion (VAE pass)", type="pil", height=512)
            render_info = gr.Textbox(label="Info", lines=3, interactive=False)
            render_btn.click(
                on_render, inputs=shared_inputs,
                outputs=[primary_output, sharp_alone_output, diffusion_output, render_info],
            )

        with gr.TabItem("Sweep"):
            gr.Markdown(
                "Vary one knob at a time across low/mid/high. ~16 renders, "
                "saved into a single folder along with a settings overlay version."
            )
            sweep_btn = gr.Button("Generate sweep", variant="primary")
            sweep_gallery = gr.Gallery(label="Sweep results (labeled)", columns=4, height=720)
            sweep_info = gr.Textbox(label="Info", lines=3, interactive=False)
            sweep_btn.click(on_sweep, inputs=shared_inputs,
                            outputs=[sweep_gallery, sweep_info])

        with gr.TabItem("Party"):
            gr.Markdown(
                "Random combos. Pin region_method to one value to avoid "
                "patch-bank rebuilds and stay focused on one segmentation style, "
                "or 'randomize' to roll it fresh each render."
            )
            with gr.Row():
                party_count_slider = gr.Slider(1, 40, value=12, step=1, label="party count")
                party_region_pin = gr.Dropdown(
                    ['randomize', 'felzenszwalb', 'threshold', 'square', 'random_square'],
                    value='randomize',
                    label="region_method",
                )
            party_btn = gr.Button("Party mode 🎉", variant="primary")
            party_gallery = gr.Gallery(label="Party results (labeled)", columns=4, height=720)
            party_info = gr.Textbox(label="Info", lines=3, interactive=False)
            party_btn.click(
                on_party,
                inputs=[image_input, source_dir_input, prompt_box, party_count_slider, party_region_pin],
                outputs=[party_gallery, party_info],
            )


def parse_args():
    parser = argparse.ArgumentParser(description="CollageNet img2img Gradio app")
    parser.add_argument("--share", action="store_true", help="Generate a public share URL")
    parser.add_argument("--port", type=int, default=7860, help="Port to bind (default 7860)")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind (default 127.0.0.1)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    app.launch(server_name=args.host, server_port=args.port, share=args.share)
