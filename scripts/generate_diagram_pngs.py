#!/usr/bin/env python3
"""Render Mermaid diagrams to PNG via Playwright + Mermaid.js and push to HF model cards.

Uses the exact same Mermaid.js library (loaded from CDN) that HuggingFace Hub uses,
rendered headlessly by Playwright/Chromium — output is pixel-identical to what HF
would show if it actually rendered the diagrams.

Requirements
------------
    pip install playwright
    python -m playwright install chromium

Usage
-----
    python scripts/generate_diagram_pngs.py
"""

from __future__ import annotations

import json
import re
import shutil
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "results" / "diagrams"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(PROJECT_ROOT))

HF_SCRATCH_REPO  = "HlexNC/chexvision-scratch"
HF_DENSENET_REPO = "HlexNC/chexvision-densenet"

# Default v2 configs used to populate architecture-specific diagrams
SCRATCH_CONFIG = {
    "model": {
        "architecture": {
            "block_config": [3, 4, 6, 3],
            "use_se": True,
            "dropout": 0.5,
        }
    }
}
DENSENET_CONFIG = {
    "model": {
        "type": "densenet",
        "fine_tuning": {
            "freeze_epochs": 5,
            "freeze_lr": 1e-3,
            "unfreeze_lr": 1e-4,
        },
    },
    "training": {"epochs": 60},
}


# ── Extract Mermaid source from hub.py ────────────────────────────────────────

def _strip_fence(block: str) -> str:
    lines = block.strip().splitlines()
    if lines and lines[0].strip() == "```mermaid":
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines)


def get_diagrams() -> dict[str, str]:
    """Return {png_filename: mermaid_source} for all diagrams."""
    from src.utils.hub import (
        _render_densenet_architecture,
        _render_densenet_finetuning,
        _render_pipeline_diagram,
        _render_scratch_architecture,
    )
    return {
        "arch_scratch.png":        _strip_fence(_render_scratch_architecture(SCRATCH_CONFIG)),
        "arch_densenet.png":       _strip_fence(_render_densenet_architecture()),
        "finetuning_densenet.png": _strip_fence(_render_densenet_finetuning(DENSENET_CONFIG)),
        "pipeline_training.png":   _strip_fence(_render_pipeline_diagram()),
    }


# ── Playwright + Mermaid.js rendering ────────────────────────────────────────

# "default" is the same theme mermaid2img.com and HF Hub use — blue/purple nodes,
# correct edge-label positioning.  htmlLabels must be False; True causes edge labels
# to render as floating positioned <div>s instead of inline SVG text.
# rankSpacing / nodeSpacing map to dagre's ranksep / nodesep (default 50 each).
# Reducing them tightens the vertical pipeline without breaking horizontal diagrams.
MERMAID_INIT = {
    "startOnLoad": False,
    "theme": "default",
    "themeVariables": {
        "fontSize": "16px",
        "fontFamily": "ui-sans-serif, system-ui, -apple-system, sans-serif",
    },
    "flowchart": {
        "htmlLabels": False,
        "curve": "basis",
        "rankSpacing": 30,
        "nodeSpacing": 30,
        "diagramPadding": 12,
    },
}

# Target SVG widths (logical CSS pixels).  device_scale_factor=2 doubles these
# for the final PNG, giving crisp 2× resolution output.
TARGET_WIDTHS: dict[str, int] = {
    "arch_scratch.png":        2400,   # wide horizontal pipeline
    "arch_densenet.png":       2000,   # slightly shorter horizontal
    "finetuning_densenet.png": 1400,   # two-box only
    "pipeline_training.png":    900,   # vertical — tall, not wide
}

# HTML template: loads Mermaid from CDN, renders one diagram, signals completion.
_HTML_TMPL = """\
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ background: #ffffff; }}
  #outer {{ display: inline-block; padding: 40px; }}
</style>
</head>
<body>
<div id="outer">
  <pre class="mermaid">{source}</pre>
</div>
<script type="module">
  import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs';
  mermaid.initialize({init_cfg});
  await mermaid.run({{ nodes: document.querySelectorAll('.mermaid') }});
  document.title = 'READY';
</script>
</body>
</html>
"""

# JS snippet: resize the rendered SVG to exactly `targetW` CSS pixels wide.
_RESIZE_JS = """\
(targetW) => {
    const svg = document.querySelector('.mermaid svg');
    if (!svg) return [0, 0];
    const vb = svg.getAttribute('viewBox');
    let vw, vh;
    if (vb) {
        [, , vw, vh] = vb.split(/[\\s,]+/).map(Number);
    } else {
        const r = svg.getBoundingClientRect();
        vw = r.width; vh = r.height;
    }
    const ratio = targetW / vw;
    svg.setAttribute('width',  targetW);
    svg.setAttribute('height', Math.ceil(vh * ratio));
    return [targetW, Math.ceil(vh * ratio)];
}
"""


def render_diagram(name: str, mmd_source: str, page) -> Path:
    """Render one Mermaid diagram to a high-resolution PNG."""
    target_w = TARGET_WIDTHS.get(name, 1600)

    safe_source = (mmd_source
                   .replace("&", "&amp;")
                   .replace("<", "&lt;")
                   .replace(">", "&gt;"))

    html = _HTML_TMPL.format(source=safe_source, init_cfg=json.dumps(MERMAID_INIT))

    tmp_html = OUT_DIR / f"_tmp_{name}.html"
    tmp_html.write_text(html, encoding="utf-8")

    try:
        page.goto(f"file:///{tmp_html.as_posix()}", wait_until="domcontentloaded")
        page.wait_for_function("document.title === 'READY'", timeout=30_000)

        # Force SVG to the target CSS width so text / boxes are large enough
        dims = page.evaluate(_RESIZE_JS, target_w)
        svg_w, svg_h = int(dims[0]), int(dims[1])

        # Resize viewport so the full diagram + padding is captured
        padding = 80
        page.set_viewport_size({
            "width":  svg_w + padding,
            "height": svg_h + padding,
        })

        # Screenshot the padded outer container (includes the 40px padding we set)
        outer = page.query_selector("#outer")
        if outer is None:
            raise RuntimeError(f"Container not found for {name}")

        out_png = OUT_DIR / name
        outer.screenshot(path=str(out_png), scale="device")

        size_kb = out_png.stat().st_size // 1024
        print(f"  OK  {name}  ({svg_w}×{svg_h} css px  →  {size_kb} KB PNG)")
        return out_png
    finally:
        tmp_html.unlink(missing_ok=True)


def render_all(diagrams: dict[str, str]) -> dict[str, Path]:
    """Render all diagrams and return {png_name: local_path}."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("ERROR: Playwright not installed.")
        print("  pip install playwright && python -m playwright install chromium")
        sys.exit(1)

    png_paths: dict[str, Path] = {}

    with sync_playwright() as pw:
        browser = pw.chromium.launch(
            args=["--no-sandbox", "--disable-setuid-sandbox"],
        )
        # device_scale_factor=2 doubles every CSS pixel → crisp 2× PNG
        context = browser.new_context(device_scale_factor=2)
        page = context.new_page()
        page.set_viewport_size({"width": 3000, "height": 2000})

        for name, source in diagrams.items():
            png_paths[name] = render_diagram(name, source, page)

        browser.close()

    return png_paths


# ── Post-processing: crop whitespace + rounded corners ───────────────────────

def post_process_png(path: Path) -> None:
    """Trim white margins and apply soft rounded corners to a PNG in-place.

    Corner radius is set to 1.5 % of the image width so every diagram shows
    ~12-16 px of curvature at the widths HF Hub typically renders them.
    Auto-crop removes Mermaid's outer white padding while keeping a small
    uniform margin so the diagram never feels clipped.
    """
    import numpy as np
    from PIL import Image, ImageDraw

    img = Image.open(path).convert("RGBA")
    rgb = np.array(img)[:, :, :3]

    # Treat pixels ≥ 248 on all channels as "white background"
    is_white = np.all(rgb >= 248, axis=2)
    not_white = ~is_white

    rows_with_content = np.any(not_white, axis=1)
    cols_with_content = np.any(not_white, axis=0)

    if not rows_with_content.any():
        return  # blank image — skip

    rmin = int(np.argmax(rows_with_content))
    rmax = int(len(rows_with_content) - 1 - np.argmax(rows_with_content[::-1]))
    cmin = int(np.argmax(cols_with_content))
    cmax = int(len(cols_with_content) - 1 - np.argmax(cols_with_content[::-1]))

    # Keep a small uniform margin around the content
    margin = 28
    rmin = max(0, rmin - margin)
    rmax = min(img.height - 1, rmax + margin)
    cmin = max(0, cmin - margin)
    cmax = min(img.width - 1, cmax + margin)

    img = img.crop((cmin, rmin, cmax + 1, rmax + 1))

    # Rounded-corner mask: radius = 1.5 % of width, clamped to [20, 80] px
    radius = max(20, min(80, int(img.width * 0.015)))
    mask = Image.new("L", img.size, 0)
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle([0, 0, img.width - 1, img.height - 1],
                           radius=radius, fill=255)
    img.putalpha(mask)

    img.save(path, "PNG", optimize=True)
    size_kb = path.stat().st_size // 1024
    print(f"  post  {path.name}  →  {img.width}×{img.height} px  "
          f"(r={radius}px  {size_kb} KB)")


def post_process_all(png_paths: dict[str, Path]) -> None:
    for path in png_paths.values():
        post_process_png(path)


# ── HF Hub helpers ────────────────────────────────────────────────────────────

def _get_token() -> str:
    try:
        from dotenv import load_dotenv
        load_dotenv(PROJECT_ROOT / ".env")
    except ImportError:
        pass
    import os
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise RuntimeError("HF_TOKEN not found — add it to .env or export it.")
    return token


def _fetch_readme(repo_id: str, token: str) -> str:
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(
        repo_id=repo_id, filename="README.md",
        repo_type="model", token=token, force_download=True,
    )
    return Path(path).read_text(encoding="utf-8")


# Display widths per diagram.
# HF Hub content area is ~860 px wide.  These percentages keep each diagram
# at a comfortable reading size without flooding the full column:
#   - Architecture diagrams are wide → 88 % fills the column nicely
#   - Fine-tuning (two boxes) → 62 % is enough, leaves breathing room
#   - Pipeline is tall/narrow → 42 % keeps it from dominating the page
DISPLAY_WIDTHS: dict[str, str] = {
    "arch_scratch.png":        "88%",
    "arch_densenet.png":       "88%",
    "finetuning_densenet.png": "62%",
    "pipeline_training.png":   "42%",
}


def _img_html(filename: str, alt: str, repo_id: str) -> str:
    """Centred <img> block using an absolute resolve/main URL (relative paths
    are not resolved by the HF Hub markdown renderer inside HTML tags)."""
    w = DISPLAY_WIDTHS.get(filename, "80%")
    url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
    return (
        f'<p align="center">\n'
        f'  <img src="{url}" width="{w}" alt="{alt}"/>\n'
        f'</p>'
    )


def _replace_diagram(content: str, section_heading: str,
                     img_alt: str, img_filename: str, repo_id: str) -> str:
    """Replace a ```mermaid block OR any existing image reference with a styled HTML tag."""
    html_tag = _img_html(img_filename, img_alt, repo_id)

    # 1. Mermaid code fence under the heading
    mermaid_pat = (
        r"(" + re.escape(section_heading) + r"\s*\n\s*)"
        r"```mermaid\n.*?```"
    )
    new_content, n = re.subn(mermaid_pat, r"\g<1>" + html_tag,
                              content, flags=re.DOTALL)
    if n:
        return new_content

    # 2. Bare markdown image from a previous run
    md_img_pat = r"!\[[^\]]*\]\(" + re.escape(img_filename) + r"\)"
    new_content, n = re.subn(md_img_pat, html_tag, content)
    if n:
        return new_content

    # 3. Existing <p align="center"><img src="…filename…"…/></p> block
    html_img_pat = (
        r'<p align="center">\s*\n\s*<img\s[^>]*'
        + re.escape(img_filename)
        + r'[^>]*/>\s*\n\s*</p>'
    )
    new_content, n = re.subn(html_img_pat, html_tag, content, flags=re.DOTALL)
    if n:
        return new_content

    print(f"    WARN: could not find any image block for '{img_filename}'")
    return content


def push_to_hf(png_paths: dict[str, Path]) -> None:
    token = _get_token()
    from huggingface_hub import HfApi
    api = HfApi(token=token)

    # ── scratch ──────────────────────────────────────────────────────────────
    print("\n=== Pushing to HlexNC/chexvision-scratch ===")
    readme = _fetch_readme(HF_SCRATCH_REPO, token)
    readme = _replace_diagram(readme, "## Architecture",
                               "SE-ResNet Architecture", "arch_scratch.png",
                               HF_SCRATCH_REPO)
    readme = _replace_diagram(readme, "## Training Pipeline",
                               "Training Pipeline", "pipeline_training.png",
                               HF_SCRATCH_REPO)
    with tempfile.TemporaryDirectory() as tmp:
        tp = Path(tmp)
        shutil.copy2(png_paths["arch_scratch.png"],      tp / "arch_scratch.png")
        shutil.copy2(png_paths["pipeline_training.png"], tp / "pipeline_training.png")
        (tp / "README.md").write_text(readme, encoding="utf-8")
        api.upload_folder(folder_path=str(tp), repo_id=HF_SCRATCH_REPO,
                          repo_type="model",
                          commit_message="Style diagram images with centred HTML + width constraints")
    print(f"  OK  {HF_SCRATCH_REPO}")

    # ── densenet ─────────────────────────────────────────────────────────────
    print("\n=== Pushing to HlexNC/chexvision-densenet ===")
    readme = _fetch_readme(HF_DENSENET_REPO, token)
    readme = _replace_diagram(readme, "## Architecture",
                               "DenseNet Architecture", "arch_densenet.png",
                               HF_DENSENET_REPO)
    readme = _replace_diagram(readme, "## Fine-Tuning Strategy",
                               "Fine-Tuning Strategy", "finetuning_densenet.png",
                               HF_DENSENET_REPO)
    readme = _replace_diagram(readme, "## Training Pipeline",
                               "Training Pipeline", "pipeline_training.png",
                               HF_DENSENET_REPO)
    with tempfile.TemporaryDirectory() as tmp:
        tp = Path(tmp)
        shutil.copy2(png_paths["arch_densenet.png"],       tp / "arch_densenet.png")
        shutil.copy2(png_paths["finetuning_densenet.png"], tp / "finetuning_densenet.png")
        shutil.copy2(png_paths["pipeline_training.png"],   tp / "pipeline_training.png")
        (tp / "README.md").write_text(readme, encoding="utf-8")
        api.upload_folder(folder_path=str(tp), repo_id=HF_DENSENET_REPO,
                          repo_type="model",
                          commit_message="Style diagram images with centred HTML + width constraints")
    print(f"  OK  {HF_DENSENET_REPO}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    print("\n=== Extracting Mermaid sources from hub.py ===")
    diagrams = get_diagrams()
    for name in diagrams:
        print(f"  found  {name}")

    print(f"\n=== Rendering PNGs via Playwright + Mermaid.js -> {OUT_DIR} ===")
    png_paths = render_all(diagrams)

    print("\n=== Post-processing: crop whitespace + rounded corners ===")
    post_process_all(png_paths)

    print("\n=== Pushing to HuggingFace Hub ===")
    push_to_hf(png_paths)

    print("\nDone. Diagrams are live on HF Hub.\n")


if __name__ == "__main__":
    main()
