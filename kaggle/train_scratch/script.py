#!/usr/bin/env python3
"""CheXVision — Train Custom CNN From Scratch (Kaggle Kernel).

This is a self-contained script designed to run as a Kaggle GPU kernel.
It installs dependencies, downloads the dataset, trains the model, and
uploads the best checkpoint to HuggingFace Hub.
"""

# ---------------------------------------------------------------------------
# 1. Install dependencies
# ---------------------------------------------------------------------------
import subprocess, sys

subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "torch", "torchvision", "numpy", "pandas", "Pillow",
    "scikit-learn", "matplotlib", "seaborn", "tqdm",
    "pyyaml", "huggingface-hub", "datasets",
])

# ---------------------------------------------------------------------------
# 2. Clone repo & add to path
# ---------------------------------------------------------------------------
import os, pathlib

REPO_URL = "https://github.com/arudaev/chexvision.git"
REPO_DIR = pathlib.Path("/kaggle/working/chexvision")

if not REPO_DIR.exists():
    subprocess.check_call(["git", "clone", REPO_URL, str(REPO_DIR)])

sys.path.insert(0, str(REPO_DIR))
os.chdir(REPO_DIR)

# ---------------------------------------------------------------------------
# 3. Download dataset
# ---------------------------------------------------------------------------
DATA_DIR = pathlib.Path("/kaggle/working/data")

subprocess.check_call([
    sys.executable, "-m", "src.data.download",
    "--output-dir", str(DATA_DIR),
])

# ---------------------------------------------------------------------------
# 4. Prepare config
# ---------------------------------------------------------------------------
import yaml

os.environ["CHEXVISION_ALLOW_LOCAL"] = "1"  # Kaggle env detection should work, but just in case

CONFIG_PATH = REPO_DIR / "configs" / "scratch.yaml"
DEFAULT_PATH = REPO_DIR / "configs" / "default.yaml"

# Load default config first, then overlay model-specific config
with open(DEFAULT_PATH) as f:
    config = yaml.safe_load(f)

with open(CONFIG_PATH) as f:
    scratch_cfg = yaml.safe_load(f)

# Deep-merge scratch overrides into default
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

# Point paths to Kaggle working directory
config["data"]["data_dir"] = str(DATA_DIR)
config["logging"] = config.get("logging", {})
config["logging"]["checkpoint_dir"] = "/kaggle/working/checkpoints"

print("=== Training Config ===")
print(yaml.dump(config, default_flow_style=False))

# ---------------------------------------------------------------------------
# 5. Train
# ---------------------------------------------------------------------------
from src.training.trainer import train

train(config)

# ---------------------------------------------------------------------------
# 6. Upload best checkpoint to HuggingFace Hub
# ---------------------------------------------------------------------------
from huggingface_hub import HfApi

try:
    from kaggle_secrets import UserSecretsClient
    secrets = UserSecretsClient()
    hf_token = secrets.get_secret("HF_TOKEN")
except Exception:
    hf_token = os.environ.get("HF_TOKEN")

if hf_token:
    ckpt_dir = pathlib.Path("/kaggle/working/checkpoints")
    best_ckpt = ckpt_dir / "CheXVision-ResNet_best.pth"

    if best_ckpt.exists():
        api = HfApi()
        api.upload_file(
            path_or_fileobj=str(best_ckpt),
            path_in_repo=best_ckpt.name,
            repo_id="HlexNC/chexvision-scratch",
            repo_type="model",
            token=hf_token,
        )
        print(f"Uploaded {best_ckpt.name} to HlexNC/chexvision-scratch")

        # Also upload training history
        history_file = ckpt_dir / "CheXVision-ResNet_history.json"
        if history_file.exists():
            api.upload_file(
                path_or_fileobj=str(history_file),
                path_in_repo=history_file.name,
                repo_id="HlexNC/chexvision-scratch",
                repo_type="model",
                token=hf_token,
            )
            print(f"Uploaded {history_file.name} to HlexNC/chexvision-scratch")
    else:
        print(f"WARNING: Best checkpoint not found at {best_ckpt}")
else:
    print("WARNING: HF_TOKEN not available — skipping model upload.")

print("Done.")
