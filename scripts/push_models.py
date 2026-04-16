#!/usr/bin/env python3
"""Upload a trained CheXVision checkpoint to HuggingFace Hub.

Usage:
    python scripts/push_models.py --checkpoint checkpoints/CheXVision-ResNet_best.pth
    python scripts/push_models.py --checkpoint checkpoints/CheXVision-DenseNet_best.pth --repo-id HlexNC/chexvision-densenet
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
from huggingface_hub import HfApi


# Default repo IDs keyed by the model type stored in the checkpoint config.
DEFAULT_REPOS = {
    "scratch": "HlexNC/chexvision-scratch",
    "densenet": "HlexNC/chexvision-densenet",
}


def _load_token() -> str:
    """Resolve HF_TOKEN from .env file or environment variable."""
    # Try python-dotenv first (reads project-root .env)
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass  # python-dotenv not installed — fall through to env var

    token = os.environ.get("HF_TOKEN")
    if not token:
        print(
            "ERROR: HF_TOKEN not found. Set it in .env or as an environment variable.",
            file=sys.stderr,
        )
        sys.exit(1)
    return token


def _detect_repo(checkpoint: dict) -> str:
    """Infer the default HF repo from the checkpoint's config."""
    config = checkpoint.get("config", {})
    model_type = config.get("model", {}).get("type", "")
    repo = DEFAULT_REPOS.get(model_type)
    if repo:
        return repo
    # Fallback: try the model name
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

    # Validate checkpoint
    ckpt_path: Path = args.checkpoint
    if not ckpt_path.exists():
        print(f"ERROR: Checkpoint not found: {ckpt_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading checkpoint: {ckpt_path} ...")
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # Verify it has expected keys
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

    # Determine repo
    repo_id = args.repo_id or _detect_repo(checkpoint)
    print(f"  Target repo: {repo_id}")

    # Upload
    token = _load_token()
    api = HfApi()

    print(f"Uploading {ckpt_path.name} to {repo_id} ...")
    api.upload_file(
        path_or_fileobj=str(ckpt_path),
        path_in_repo=ckpt_path.name,
        repo_id=repo_id,
        repo_type="model",
        token=token,
    )

    print(f"Upload complete: https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
