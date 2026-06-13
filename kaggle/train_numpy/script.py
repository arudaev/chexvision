#!/usr/bin/env python3
"""CheXVision — train the from-scratch NumPy net (Kaggle CPU kernel).

The from-scratch model lives in the ``chexvision-mini`` submodule mounted at
``src/numpy_net``. This runner bootstraps the bundled project source (injected by
scripts/dispatch.py), puts the submodule's package on the path, and runs its
training pipeline in ``kaggle`` mode — streaming a subset of the dataset from HF
and uploading artifacts to ``arudaev/chexvision-mini``.

Pure NumPy → CPU only; no torch is installed here. Push via scripts/dispatch.py.
"""

import base64
import io
import os
import pathlib
import shutil
import subprocess
import sys
import zipfile

# ---------------------------------------------------------------------------
# 1. Lightweight dependencies (no torch — the net is pure NumPy).
# ---------------------------------------------------------------------------
subprocess.check_call(
    [
        sys.executable, "-m", "pip", "install", "-q",
        "numpy", "scikit-learn", "matplotlib", "pyyaml",
        "Pillow", "datasets", "huggingface-hub", "python-dotenv", "tqdm",
    ]
)

# ---------------------------------------------------------------------------
# 2. Bootstrap the bundled project source.
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
with zipfile.ZipFile(io.BytesIO(base64.b64decode(PROJECT_BUNDLE_B64.encode("ascii")))) as archive:
    archive.extractall(REPO_DIR)

# The submodule mounts at src/numpy_net/, so its package is src/numpy_net/chexvision_mini.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
sys.path.insert(0, str(REPO_DIR / "src" / "numpy_net"))

# ---------------------------------------------------------------------------
# 3. Train, then upload artifacts to HF.
#    The mini's train.py is upload-free by design (writes to --output-dir);
#    the caller (this kernel) performs the HF upload via the mini's hub helper.
# ---------------------------------------------------------------------------
from chexvision_mini.hub import upload_results  # noqa: E402
from chexvision_mini.train import main  # noqa: E402

ARTIFACTS = "/kaggle/working/artifacts"
main(["--mode", "kaggle", "--output-dir", ARTIFACTS])
upload_results(ARTIFACTS, "arudaev/chexvision-mini")
