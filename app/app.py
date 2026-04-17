"""CheXVision — Streamlit Demo Application.

Upload a chest X-ray and get pathology predictions from both models.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

# Hugging Face Streamlit Spaces launch `app/app.py` directly, which puts the
# `app/` directory on sys.path instead of the repository root.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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
        model = CheXVisionScratch()
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
        model = CheXVisionDenseNet(pretrained=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device).eval()
        models["DenseNet-121 (Transfer Learning)"] = model

    return models


def predict(model: torch.nn.Module, image: Image.Image) -> dict[str, np.ndarray]:
    """Run inference on a single image."""
    device = next(model.parameters()).device
    transform = get_eval_transforms(224)
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)

    ml_probs = torch.sigmoid(outputs["multilabel_logits"]).cpu().numpy()[0]
    bin_prob = torch.sigmoid(outputs["binary_logits"]).cpu().numpy()[0, 0]

    return {"multilabel_probs": ml_probs, "binary_prob": float(bin_prob)}


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

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

        col1, col2 = st.columns([1, 2])

        with col1:
            st.image(image, caption="Uploaded X-ray", use_container_width=True)

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
                        st.progress(prob, text=f"{label}: {prob:.1%}")
                else:
                    st.success("No pathologies detected above threshold.")

                st.divider()


if __name__ == "__main__":
    main()
