#!/usr/bin/env python3
"""CheXVision - Build the 320x320 HF dataset from the raw NIH source repo."""

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
    os.environ.setdefault("HF_DATASETS_CACHE", "/kaggle/working/hf_datasets_cache")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "30")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "600")
    os.environ.setdefault("HF_HUB_VERBOSITY", "info")
    pathlib.Path(os.environ["HF_HOME"]).mkdir(parents=True, exist_ok=True)
    pathlib.Path(os.environ["HF_DATASETS_CACHE"]).mkdir(parents=True, exist_ok=True)


configure_kaggle_hf_env()

subprocess.check_call(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "Pillow>=10.0.0",
        "datasets>=2.16.0",
        "huggingface-hub>=0.20.0",
        "pandas>=2.0.0",
        "pyarrow",
        "python-dotenv",
    ]
)

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

from src.data.resize_320_pipeline import run_resize_320_pipeline  # noqa: E402

run_resize_320_pipeline()
