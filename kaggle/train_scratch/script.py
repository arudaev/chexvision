#!/usr/bin/env python3
"""CheXVision - Train Custom CNN From Scratch (Kaggle Kernel).

This is a self-contained script designed to run as a Kaggle GPU kernel.
It installs dependencies, downloads the dataset from Hugging Face, trains
the model, and uploads the best checkpoint plus metadata back to the Hub.
"""

# ---------------------------------------------------------------------------
# 1. Install dependencies
# ---------------------------------------------------------------------------
import base64
import io
import os
import pathlib
import shutil
import subprocess
import sys
import zipfile


def configure_kaggle_hf_env() -> None:
    """Set HF runtime env vars before importing any HF-backed libraries."""
    os.environ.setdefault("HF_HOME", "/kaggle/working/hf_home")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "30")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")
    os.environ.setdefault("HF_HUB_VERBOSITY", "info")
    pathlib.Path(os.environ["HF_HOME"]).mkdir(parents=True, exist_ok=True)


configure_kaggle_hf_env()

# PyTorch 2.3.1 is the last release whose cu118 wheels include sm_60 binaries,
# which is required for the P100 GPU (CUDA compute capability 6.0).
# Newer PyTorch versions (including the one in Kaggle's Latest Container Image)
# only ship sm_70+, so we pin explicitly rather than relying on the container.
subprocess.check_call(
    [
        sys.executable, "-m", "pip", "install", "-q",
        "torch==2.3.1", "torchvision==0.18.1",
        "--index-url", "https://download.pytorch.org/whl/cu118",
    ]
)
subprocess.check_call(
    [
        sys.executable, "-m", "pip", "install", "-q",
        "numpy",
        "pandas",
        "Pillow",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "tqdm",
        "pyyaml",
        "huggingface-hub",
        "datasets",
        "python-dotenv",
    ]
)

# ---------------------------------------------------------------------------
# 2. Bootstrap the bundled project source
# ---------------------------------------------------------------------------
PROJECT_BUNDLE_B64 = "__CHEXVISION_PROJECT_BUNDLE_B64__"
REPO_DIR = pathlib.Path("/kaggle/working/chexvision")

if PROJECT_BUNDLE_B64 == "__CHEXVISION_PROJECT_BUNDLE_B64__":
    raise RuntimeError(
        "This Kaggle script is a template. Push it via scripts/dispatch.py so "
        "the project bundle is embedded before upload."
    )

if REPO_DIR.exists():
    shutil.rmtree(REPO_DIR)

REPO_DIR.mkdir(parents=True, exist_ok=True)
archive_bytes = base64.b64decode(PROJECT_BUNDLE_B64.encode("ascii"))
with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
    archive.extractall(REPO_DIR)

sys.path.insert(0, str(REPO_DIR))
os.chdir(REPO_DIR)

from src.utils.hub import (  # noqa: E402
    HF_DATASET_REPO,
    HF_DATASET_REVISION,
    configure_hf_runtime,
    upload_model_artifacts,
)

hf_token = configure_hf_runtime(required_token=True, check_dns=True)
PYTHON_ENV = os.environ.copy()
PYTHON_ENV["PYTHONPATH"] = str(REPO_DIR)
PYTHON_ENV["HF_TOKEN"] = hf_token
PYTHON_ENV["HUGGING_FACE_HUB_TOKEN"] = hf_token
PYTHON_ENV["CHEXVISION_HF_SNAPSHOT_DIR"] = "/kaggle/working/hf_datasets/chest_xray_14"

# ---------------------------------------------------------------------------
# 3. Download dataset
# ---------------------------------------------------------------------------
DATA_DIR = pathlib.Path("/kaggle/working/data")

subprocess.check_call(
    [
        sys.executable,
        "-m",
        "src.data.download",
        "--output-dir",
        str(DATA_DIR),
        "--repo-id",
        HF_DATASET_REPO,
        "--revision",
        HF_DATASET_REVISION,
        "--snapshot-dir",
        PYTHON_ENV["CHEXVISION_HF_SNAPSHOT_DIR"],
    ],
    cwd=REPO_DIR,
    env=PYTHON_ENV,
)

# ---------------------------------------------------------------------------
# 4. Prepare config
# ---------------------------------------------------------------------------
import yaml  # noqa: E402

os.environ["CHEXVISION_ALLOW_LOCAL"] = "1"  # Kaggle env detection should work, but just in case

CONFIG_PATH = REPO_DIR / "configs" / "scratch.yaml"
DEFAULT_PATH = REPO_DIR / "configs" / "default.yaml"

with open(DEFAULT_PATH, encoding="utf-8") as file:
    config = yaml.safe_load(file)

with open(CONFIG_PATH, encoding="utf-8") as file:
    scratch_cfg = yaml.safe_load(file)


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base* (mutates base)."""
    for key, value in override.items():
        if key == "_defaults_":
            continue
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


config = _deep_merge(config, scratch_cfg)
config["data"]["data_dir"] = str(DATA_DIR)
config["data"]["hf_dataset_repo"] = HF_DATASET_REPO
config["data"]["hf_dataset_revision"] = HF_DATASET_REVISION
config["logging"] = config.get("logging", {})
config["logging"]["checkpoint_dir"] = "/kaggle/working/checkpoints"

print("=== Training Config ===")
print(yaml.dump(config, default_flow_style=False))

# ---------------------------------------------------------------------------
# 5. Train
# ---------------------------------------------------------------------------
from src.training.trainer import train  # noqa: E402

train(config)

# ---------------------------------------------------------------------------
# 6. Upload best checkpoint to HuggingFace Hub
# ---------------------------------------------------------------------------
import torch  # noqa: E402

ckpt_dir = pathlib.Path("/kaggle/working/checkpoints")
best_ckpt = ckpt_dir / "CheXVision-ResNet_best.pth"
history_file = ckpt_dir / "CheXVision-ResNet_history.json"

if best_ckpt.exists():
    checkpoint = torch.load(best_ckpt, map_location="cpu", weights_only=False)
    upload_model_artifacts(
        checkpoint_path=best_ckpt,
        repo_id="arudaev/chexvision-scratch",
        token=hf_token,
        checkpoint=checkpoint,
        history_path=history_file if history_file.exists() else None,
    )
    print(
        "Uploaded checkpoint, training history, config, and model card to "
        "arudaev/chexvision-scratch"
    )
else:
    print(f"WARNING: Best checkpoint not found at {best_ckpt}")

print("Done.")
