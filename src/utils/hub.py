"""Helpers for Hugging Face Hub authentication, runtime config, and uploads."""

from __future__ import annotations

import json
import os
import shutil
import socket
import tempfile
from pathlib import Path
from typing import Any

HF_DATASET_REPO = "HlexNC/chest-xray-14"
HF_DATASET_REVISION = os.environ.get(
    "CHEXVISION_DATASET_REVISION",
    "c4e9a86b38de3b1604afa6e9f514d156eb9d20bf",
)


def _load_dotenv_if_available() -> None:
    """Load project-root environment variables when python-dotenv is installed."""
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        return


def _set_hf_token_env(token: str) -> str:
    """Persist the resolved token to the canonical HF environment variables."""
    os.environ["HF_TOKEN"] = token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = token
    return token


def load_hf_token(required: bool = False) -> str | None:
    """Resolve an HF token from env vars, project .env, Kaggle dataset, or Kaggle secrets.

    Resolution order (first non-empty value wins):
      1. Environment variables (HF_TOKEN / legacy aliases)
      2. ``/kaggle/input/chexvision-secrets/hf_token.txt``  — preferred automated path;
         attach the private dataset ``hlexnc/chexvision-secrets`` via dataset_sources in
         kernel-metadata.json so every API-pushed kernel gets it without manual UI steps.
      3. Kaggle UserSecretsClient — works only for interactive sessions, kept as fallback.
    """
    _load_dotenv_if_available()
    kaggle_secret_error: str | None = None

    # 1. Environment variables (highest priority; set by .env, GitHub Actions, etc.)
    token_names = ("HF_TOKEN", "HUGGINGFACEHUB_API_TOKEN", "HUGGING_FACE_HUB_TOKEN")
    for name in token_names:
        token = os.environ.get(name, "").strip()
        if token:
            return _set_hf_token_env(token)

    # 2. Token file from an attached Kaggle dataset source.
    #    Create the private dataset once at kaggle.com/datasets/new, upload a plain-text
    #    file named ``hf_token.txt`` that contains only the token, and add
    #    ``"hlexnc/chexvision-secrets"`` to ``dataset_sources`` in kernel-metadata.json.
    #    After that, every automated ``kaggle kernels push`` will have the token available
    #    here without any manual secrets UI steps.
    token_file = Path("/kaggle/input/chexvision-secrets/hf_token.txt")
    if token_file.exists():
        token = token_file.read_text(encoding="utf-8").strip()
        if token:
            print(f"[hub] Loaded HF_TOKEN from Kaggle dataset source: {token_file}")
            return _set_hf_token_env(token)

    # 3. Kaggle UserSecretsClient — interactive sessions only (fallback).
    try:
        from kaggle_secrets import UserSecretsClient

        token = UserSecretsClient().get_secret("HF_TOKEN").strip()
    except Exception as exc:
        token = ""
        kaggle_secret_error = f"{type(exc).__name__}: {exc}"

    if token:
        return _set_hf_token_env(token)

    if required:
        if os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
            detail = f" Kaggle reported: {kaggle_secret_error}" if kaggle_secret_error else ""
            raise RuntimeError(
                "HF_TOKEN not found. Preferred fix: create a private Kaggle dataset "
                "'hlexnc/chexvision-secrets' with a file 'hf_token.txt' containing your "
                "HF token, then add it to dataset_sources in kernel-metadata.json. "
                f"Alternatively enable HF_TOKEN in Kaggle Secrets (interactive only).{detail}"
            )
        raise RuntimeError(
            "HF_TOKEN not found. Set it in .env, export it in the environment, "
            "or add it to Kaggle Secrets."
        )
    return None


def configure_hf_runtime(
    token: str | None = None,
    *,
    required_token: bool = False,
    check_dns: bool = False,
) -> str | None:
    """Set the HF runtime environment before importing HF client libraries."""
    resolved_token = token or load_hf_token(required=required_token)

    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") and "HF_HOME" not in os.environ:
        os.environ["HF_HOME"] = "/kaggle/working/hf_home"

    hf_home = os.environ.get("HF_HOME", "").strip()
    if hf_home:
        Path(hf_home).mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "30")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")
    os.environ.setdefault("HF_HUB_VERBOSITY", "info")

    if resolved_token:
        _set_hf_token_env(resolved_token)

    if check_dns:
        try:
            socket.getaddrinfo("huggingface.co", 443)
        except OSError as exc:
            raise RuntimeError(
                "Failed to resolve huggingface.co from the current runtime. "
                "Check Kaggle internet access or a platform-side DNS issue."
            ) from exc

    return resolved_token


def _safe_metric(history: dict[str, Any] | None, key: str) -> float | None:
    """Return the best numeric value recorded for a metric history key."""
    if not history:
        return None
    values = history.get(key, [])
    if not isinstance(values, list) or not values:
        return None
    return float(max(values))


def _architecture_summary(config: dict[str, Any]) -> str:
    """Produce a short human-readable architecture summary."""
    model_cfg = config.get("model", {})
    model_type = model_cfg.get("type", "")
    if model_type == "densenet":
        return "DenseNet-121 transfer learning with a shared backbone and dual classification heads."
    return "Custom residual CNN trained from scratch with shared features and dual classification heads."


def render_model_card(
    repo_id: str,
    checkpoint: dict[str, Any],
    history: dict[str, Any] | None = None,
) -> str:
    """Render a lightweight Hugging Face model card for a trained checkpoint."""
    config = checkpoint.get("config", {})
    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    train_cfg = config.get("training", {})
    model_name = model_cfg.get("name", repo_id.split("/")[-1])
    dataset_repo = data_cfg.get("hf_dataset_repo", HF_DATASET_REPO)
    dataset_revision = data_cfg.get("hf_dataset_revision", HF_DATASET_REVISION)
    best_auc = checkpoint.get("best_auc")
    epoch = checkpoint.get("epoch")
    best_binary_auc = _safe_metric(history, "binary_auc_roc")
    best_binary_f1 = _safe_metric(history, "binary_f1")

    metrics_lines = []
    if isinstance(best_auc, (int, float)):
        metrics_lines.append(f"- Best validation macro AUC-ROC: `{best_auc:.4f}`")
    if isinstance(best_binary_auc, float):
        metrics_lines.append(f"- Best validation binary AUC-ROC: `{best_binary_auc:.4f}`")
    if isinstance(best_binary_f1, float):
        metrics_lines.append(f"- Best validation binary F1: `{best_binary_f1:.4f}`")
    if epoch is not None:
        metrics_lines.append(f"- Saved checkpoint epoch: `{epoch}`")
    metrics_block = "\n".join(metrics_lines) if metrics_lines else "- Metrics will appear after the first successful training run."

    return f"""---
license: mit
language:
- en
library_name: pytorch
pipeline_tag: image-classification
tags:
- chexvision
- medical-imaging
- chest-xray
- radiology
- pytorch
datasets:
- {dataset_repo}
---

# {model_name}

## Model Details

- Repository: `{repo_id}`
- Training platform: Kaggle GPU kernel
- Dataset: [{dataset_repo}](https://huggingface.co/datasets/{dataset_repo})
- Dataset revision: `{dataset_revision}`
- Architecture: {_architecture_summary(config)}
- Training epochs configured: `{train_cfg.get("epochs", "unknown")}`
- Batch size configured: `{train_cfg.get("batch_size", "unknown")}`

## Training Metrics

{metrics_block}

## Intended Use

This model is intended for research and educational work on automated chest X-ray pathology detection.
It predicts both 14 pathology labels and a binary normal-vs-abnormal signal for the CheXVision project.

## Limitations

- This repository does not provide clinical-grade validation.
- Predictions must not be used as a substitute for professional medical judgement.
- Performance can degrade on populations, devices, or preprocessing pipelines that differ from the training data.

## Training Procedure

Training is orchestrated from the CheXVision repository and runs on Kaggle GPU kernels.
The training kernels download a pinned snapshot of the public Hugging Face dataset, save the best checkpoint, and upload the checkpoint plus metadata back to this public model repository.
"""


def upload_model_artifacts(
    checkpoint_path: Path,
    repo_id: str,
    token: str,
    checkpoint: dict[str, Any] | None = None,
    history_path: Path | None = None,
) -> None:
    """Upload a checkpoint, metadata, and model card to the HF Hub."""
    checkpoint_path = Path(checkpoint_path)
    history_path = Path(history_path) if history_path else None
    history: dict[str, Any] | None = None
    if history_path and history_path.exists():
        history = json.loads(history_path.read_text(encoding="utf-8"))

    checkpoint = checkpoint or {}
    model_card = render_model_card(repo_id, checkpoint, history)
    training_config = json.dumps(checkpoint.get("config", {}), indent=2)

    configure_hf_runtime(token=token)

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        readme_path = tmp_path / "README.md"
        config_path = tmp_path / "training_config.json"
        staged_checkpoint = tmp_path / checkpoint_path.name
        readme_path.write_text(model_card, encoding="utf-8")
        config_path.write_text(training_config, encoding="utf-8")
        shutil.copy2(checkpoint_path, staged_checkpoint)

        if history_path and history_path.exists():
            shutil.copy2(history_path, tmp_path / history_path.name)

        api.upload_folder(
            folder_path=str(tmp_path),
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Upload trained artifacts for {checkpoint_path.stem}",
        )
