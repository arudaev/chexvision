"""CheXVision — Streamlit Demo Application.

Upload a chest X-ray and get pathology predictions from both models.
"""

from __future__ import annotations

import streamlit as st
import torch
import numpy as np
from PIL import Image
from pathlib import Path

from src.data.transforms import get_eval_transforms
from src.data.dataset import PATHOLOGY_LABELS
from src.models.scratch_cnn import CheXVisionScratch
from src.models.densenet_transfer import CheXVisionDenseNet


@st.cache_resource
def load_models() -> dict[str, torch.nn.Module]:
    """Load trained models (cached across reruns)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = {}

    scratch_path = Path("checkpoints/CheXVision-ResNet_best.pth")
    densenet_path = Path("checkpoints/CheXVision-DenseNet_best.pth")

    if scratch_path.exists():
        ckpt = torch.load(scratch_path, map_location=device, weights_only=False)
        model = CheXVisionScratch()
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device).eval()
        models["Custom CNN (From Scratch)"] = model

    if densenet_path.exists():
        ckpt = torch.load(densenet_path, map_location=device, weights_only=False)
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
    st.set_page_config(page_title="CheXVision", page_icon="🫁", layout="wide")

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
        st.warning("No trained model checkpoints found in `checkpoints/`. Train models first.")
        st.code("python -m src.training.trainer --config configs/scratch.yaml", language="bash")
        return

    uploaded_file = st.file_uploader("Upload a chest X-ray image", type=["png", "jpg", "jpeg", "dcm"])

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
