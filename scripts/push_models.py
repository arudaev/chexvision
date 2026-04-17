#!/usr/bin/env python3
"""Upload a trained CheXVision checkpoint to HuggingFace Hub.

Usage:
    python scripts/push_models.py --checkpoint checkpoints/CheXVision-ResNet_best.pth
    python scripts/push_models.py --checkpoint checkpoints/CheXVision-DenseNet_best.pth --repo-id HlexNC/chexvision-densenet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from src.utils.hub import load_hf_token, upload_model_artifacts

# Default repo IDs keyed by the model type stored in the checkpoint config.
DEFAULT_REPOS = {
    "scratch": "HlexNC/chexvision-scratch",
    "densenet": "HlexNC/chexvision-densenet",
}


def _detect_repo(checkpoint: dict) -> str:
    """Infer the default HF repo from the checkpoint's config."""
    config = checkpoint.get("config", {})
    model_type = config.get("model", {}).get("type", "")
    repo = DEFAULT_REPOS.get(model_type)
    if repo:
        return repo
    model_name = config.get("model", {}).get("name", "")
    if "DenseNet" in model_name or "densenet" in model_name:
        return DEFAULT_REPOS["densenet"]
    return DEFAULT_REPOS["scratch"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Push a CheXVision checkpoint to HuggingFace Hub")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to the .pth checkpoint file.",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help="HuggingFace repo ID (e.g. HlexNC/chexvision-scratch). "
        "Auto-detected from the checkpoint config if omitted.",
    )
    args = parser.parse_args()

    ckpt_path = args.checkpoint
    if not ckpt_path.exists():
        print(f"ERROR: Checkpoint not found: {ckpt_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading checkpoint: {ckpt_path} ...")
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    required_keys = {"model_state_dict", "config"}
    if not required_keys.issubset(checkpoint.keys()):
        print(
            f"WARNING: Checkpoint is missing expected keys {required_keys - checkpoint.keys()}. "
            "Proceeding anyway.",
            file=sys.stderr,
        )

    epoch = checkpoint.get("epoch", "?")
    best_auc = checkpoint.get("best_auc", "?")
    print(f"  Epoch: {epoch}  |  Best AUC-ROC: {best_auc}")

    repo_id = args.repo_id or _detect_repo(checkpoint)
    print(f"  Target repo: {repo_id}")

    try:
        token = load_hf_token(required=True)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    history_path = ckpt_path.with_name(ckpt_path.name.replace("_best.pth", "_history.json"))

    print(f"Uploading {ckpt_path.name} to {repo_id} ...")
    upload_model_artifacts(
        checkpoint_path=ckpt_path,
        repo_id=repo_id,
        token=token,
        checkpoint=checkpoint,
        history_path=history_path if history_path.exists() else None,
    )

    print(f"Upload complete: https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
