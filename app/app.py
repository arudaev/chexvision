"""CheXVision — Streamlit Demo Application.

Upload a chest X-ray and get pathology predictions from both models.
"""

from __future__ import annotations

import base64
import io
import logging
import sys
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sample_cases import SAMPLE_CASES  # noqa: E402

from src.data.dataset import PATHOLOGY_LABELS  # noqa: E402
from src.data.transforms import get_eval_transforms  # noqa: E402
from src.models.densenet_transfer import CheXVisionDenseNet  # noqa: E402
from src.models.scratch_cnn import CheXVisionScratch  # noqa: E402

logger = logging.getLogger(__name__)

# HF Hub model repos
HF_SCRATCH_REPO = "HlexNC/chexvision-scratch"
HF_DENSENET_REPO = "HlexNC/chexvision-densenet"


def _try_load_checkpoint(
    repo_id: str,
    filename: str,
    local_path: Path,
    device: torch.device,
) -> dict | None:
    """Try loading a checkpoint from HF Hub first, then local path."""
    # Try HF Hub
    try:
        path = hf_hub_download(repo_id=repo_id, filename=filename)
        return torch.load(path, map_location=device, weights_only=False)
    except Exception:
        logger.debug("Could not download %s from %s", filename, repo_id)

    # Fall back to local
    if local_path.exists():
        return torch.load(local_path, map_location=device, weights_only=False)

    return None


@st.cache_resource
def load_models() -> dict[str, torch.nn.Module]:
    """Load trained models from HF Hub or local checkpoints (cached)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = {}

    # Scratch CNN
    ckpt = _try_load_checkpoint(
        HF_SCRATCH_REPO,
        "CheXVision-ResNet_best.pth",
        Path("checkpoints/CheXVision-ResNet_best.pth"),
        device,
    )
    if ckpt:
        # Build with the exact architecture the checkpoint was trained with,
        # not hardcoded defaults — so v1 and v2 checkpoints both load correctly.
        arch = ckpt.get("config", {}).get("model", {}).get("architecture", {})
        model = CheXVisionScratch(
            block_config=tuple(arch.get("block_config", [2, 2, 2, 2])),
            filter_sizes=tuple(arch.get("filter_sizes", [64, 128, 256, 512])),
            dropout=arch.get("dropout", 0.5),
            use_se=arch.get("use_se", False),  # False: old checkpoints predate SE blocks
        )
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device).eval()
        models["Custom CNN (From Scratch)"] = model

    # DenseNet-121
    ckpt = _try_load_checkpoint(
        HF_DENSENET_REPO,
        "CheXVision-DenseNet_best.pth",
        Path("checkpoints/CheXVision-DenseNet_best.pth"),
        device,
    )
    if ckpt:
        arch = ckpt.get("config", {}).get("model", {}).get("architecture", {})
        model = CheXVisionDenseNet(
            pretrained=False,
            dropout=arch.get("dropout", 0.3),
        )
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device).eval()
        models["DenseNet-121 (Transfer Learning)"] = model

    return models


def predict(model: torch.nn.Module, image: Image.Image) -> dict[str, np.ndarray]:
    """Run inference on a single image."""
    device = next(model.parameters()).device
    transform = get_eval_transforms(320)
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)

    ml_probs = torch.sigmoid(outputs["multilabel_logits"]).cpu().numpy()[0]
    bin_prob = torch.sigmoid(outputs["binary_logits"]).cpu().numpy()[0, 0]

    return {"multilabel_probs": ml_probs, "binary_prob": float(bin_prob)}


@st.cache_data(show_spinner=False)
def decode_sample_image(preview_png_b64: str) -> bytes:
    """Decode an embedded sample preview into PNG bytes."""
    return base64.b64decode(preview_png_b64)


def load_sample_image(sample: dict[str, str]) -> Image.Image:
    """Load one embedded sample preview as a PIL image."""
    image_bytes = decode_sample_image(sample["preview_png_b64"])
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def get_sample_case(sample_key: str | None) -> dict[str, str] | None:
    """Resolve one embedded sample-image definition by key."""
    if not sample_key:
        return None

    for sample in SAMPLE_CASES:
        if sample["key"] == sample_key:
            return sample
    return None


def render_sidebar_samples() -> dict[str, str] | None:
    """Render preview thumbnails for embedded sample X-rays."""
    st.sidebar.subheader("Sample X-rays")
    st.sidebar.caption(
        "Preview a few NIH examples, copy the filenames, or load one directly into the demo."
    )

    selected_sample = None
    for sample in SAMPLE_CASES:
        preview_image = load_sample_image(sample)
        st.sidebar.image(
            preview_image,
            caption=f"{sample['title']} - {sample['filename']}",
            use_container_width=True,
        )
        st.sidebar.caption(
            f"Expected: {sample['expected_labels']}  |  {sample['note']}"
        )
        st.sidebar.code(sample["filename"], language=None)
        if st.sidebar.button(
            f"Use {sample['title']}",
            key=f"use_sample_{sample['key']}",
            use_container_width=True,
        ):
            st.session_state["selected_sample_key"] = sample["key"]
            selected_sample = sample
        st.sidebar.divider()

    return selected_sample


def main() -> None:
    st.set_page_config(page_title="CheXVision", page_icon="\U0001fac1", layout="wide")

    st.title("CheXVision")
    st.markdown("**Large-Scale Chest X-Ray Pathology Detection** — Upload a chest X-ray for AI-powered analysis.")

    st.sidebar.header("About")
    st.sidebar.markdown("""
    **Dataset**: NIH Chest X-ray14 (112,120 images)

    **Two Models**:
    - Custom CNN (from scratch)
    - DenseNet-121 (transfer learning)

    **Two Tasks**:
    - Multi-label: 14 pathologies
    - Binary: Normal vs Abnormal

    [GitHub](https://github.com/arudaev/chexvision) |
    [Dataset](https://huggingface.co/datasets/HlexNC/chest-xray-14)
    """)
    sidebar_selected_sample = render_sidebar_samples()

    models = load_models()

    if not models:
        st.warning(
            "No trained models available yet. Models will appear here once training "
            "is complete and checkpoints are uploaded to HuggingFace Hub."
        )
        st.info(
            "**Training status**: Check model repos at "
            "[chexvision-scratch](https://huggingface.co/HlexNC/chexvision-scratch) and "
            "[chexvision-densenet](https://huggingface.co/HlexNC/chexvision-densenet)"
        )
        return

    uploaded_file = st.file_uploader("Upload a chest X-ray image", type=["png", "jpg", "jpeg"])
    selected_sample = sidebar_selected_sample or get_sample_case(
        st.session_state.get("selected_sample_key")
    )

    if uploaded_file is not None or selected_sample is not None:
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            image_caption = "Uploaded X-ray"
            image_note = None
        else:
            image = load_sample_image(selected_sample)
            image_caption = (
                f"Sample X-ray - {selected_sample['title']} "
                f"({selected_sample['filename']})"
            )
            image_note = (
                f"Expected label(s): {selected_sample['expected_labels']}  |  "
                f"{selected_sample['note']}"
            )

        col1, col2 = st.columns([1, 2])

        with col1:
            st.image(image, caption=image_caption, use_container_width=True)
            if image_note:
                st.caption(image_note)

        with col2:
            for model_name, model in models.items():
                st.subheader(model_name)
                result = predict(model, image)

                # Binary classification
                binary_label = "Abnormal" if result["binary_prob"] > 0.5 else "Normal"
                confidence = result["binary_prob"] if binary_label == "Abnormal" else 1 - result["binary_prob"]
                st.metric("Screening", f"{binary_label} ({confidence:.1%})")

                # Multi-label classification
                st.markdown("**Detected Pathologies:**")
                probs = result["multilabel_probs"]
                detected = [(PATHOLOGY_LABELS[i], probs[i]) for i in range(len(PATHOLOGY_LABELS)) if probs[i] > 0.5]

                if detected:
                    for label, prob in sorted(detected, key=lambda x: -x[1]):
                        st.progress(float(prob), text=f"{label}: {prob:.1%}")
                else:
                    st.success("No pathologies detected above threshold.")

                st.divider()


if __name__ == "__main__":
    main()
