"""Helpers for Hugging Face Hub authentication, runtime config, and uploads."""

from __future__ import annotations

import json
import os
import shutil
import socket
import tempfile
from pathlib import Path
from typing import Any

HF_DATASET_REPO = "HlexNC/chest-xray-14-320"
HF_DATASET_REVISION = os.environ.get(
    "CHEXVISION_DATASET_REVISION",
    "44443e6ee968b3c6094b63f14a27698c40b50680",
)

# NIH Chest X-ray14 pathology labels in canonical order
PATHOLOGY_LABELS = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
    "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
    "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia",
]


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
    #    Kaggle mounts dataset_sources under two possible paths depending on
    #    the runtime version — check both so old and new kernels both work.
    for token_file in (
        Path("/kaggle/input/datasets/hlexnc/chexvision-secrets/hf_token.txt"),
        Path("/kaggle/input/chexvision-secrets/hf_token.txt"),
    ):
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
        try:
            Path(hf_home).mkdir(parents=True, exist_ok=True)
        except OSError:
            pass  # Best-effort; the path may not be writable outside a real Kaggle kernel

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
        return "DenseNet-121 transfer learning with a shared feature layer and dual classification heads."
    arch = model_cfg.get("architecture", {})
    blocks = arch.get("block_config", [3, 4, 6, 3])
    use_se = arch.get("use_se", True)
    se_note = " with Squeeze-Excitation channel attention" if use_se else ""
    return (
        f"Custom residual CNN{se_note} (depth {blocks}) trained from scratch "
        "with shared features and dual classification heads."
    )


def _render_pipeline_diagram() -> str:
    """Mermaid flowchart of the full data→train→upload pipeline."""
    return """```mermaid
flowchart TD
    DS[("🗄️ HlexNC/chest-xray-14-320\n112,120 images · 36 shards · ~7.97 GB")]
    DS -->|snapshot_download| PREP["📂 data/images · data/labels.csv\ntrain 78,468 · val 11,210 · test 22,442"]
    PREP --> AUG["Augmentation Pipeline\nHFlip · Rotate±15° · RandomAffine\nColorJitter · GaussianBlur · RandomErasing"]
    AUG --> FWD["⚡ Model Forward Pass\ntorch.cuda.amp.autocast · fp16"]
    FWD --> ML["multilabel_logits B×14\nWeightedBCE + pos_weight · 14 classes"]
    FWD --> BIN["binary_logits B×1\nBCE · Normal vs. Abnormal"]
    ML --> LOSS["Combined Loss\n1.0 × multilabel + 0.5 × binary"]
    BIN --> LOSS
    LOSS --> BACK["Backward · Grad Clip 1.0\nGradient Accumulation ×4 · eff. batch 96"]
    BACK --> OPT["AdamW · CosineAnnealingLR\nearly stop patience = 15"]
    OPT -->|"↑ best val AUC-ROC"| BEST["💾 Best Checkpoint\nmodel_state · best_val_metrics · config"]
    BEST -->|upload_model_artifacts| HUB["🤗 HF Hub\ncheckpoint · history.json · model card"]
```"""


def _render_scratch_architecture(config: dict[str, Any]) -> str:
    """Mermaid architecture diagram for CheXVisionScratch."""
    arch = config.get("model", {}).get("architecture", {})
    blocks = arch.get("block_config", [3, 4, 6, 3])
    use_se = arch.get("use_se", True)
    block_label = "SE-ResBlock" if use_se else "ResBlock"
    b1, b2, b3, b4 = (blocks + [3, 4, 6, 3])[:4]
    return f"""```mermaid
graph LR
    IN["Input
    3 × 320 × 320"] --> STEM["Stem
    7×7 Conv · BN · ReLU
    3→64ch · MaxPool ÷2"]
    STEM --> S1["Stage 1
    {b1}× {block_label}
    64ch"]
    S1 --> S2["Stage 2 ↓½
    {b2}× {block_label}
    128ch"]
    S2 --> S3["Stage 3 ↓½
    {b3}× {block_label}
    256ch"]
    S3 --> S4["Stage 4 ↓½
    {b4}× {block_label}
    512ch"]
    S4 --> GAP["Global Avg Pool
    Dropout(0.5)
    512-dim"]
    GAP --> MLH["Multilabel Head
    Linear 512→14
    sigmoid · 14 pathologies"]
    GAP --> BH["Binary Head
    Linear 512→1
    sigmoid · Normal/Abnormal"]
    style MLH fill:#2e7d32,color:#fff
    style BH fill:#1565c0,color:#fff
    style IN fill:#37474f,color:#fff
```"""


def _render_densenet_architecture() -> str:
    """Mermaid architecture diagram for CheXVisionDenseNet."""
    return """```mermaid
graph LR
    IN["Input
    3 × 320 × 320"] --> BB["DenseNet-121 Backbone
    ImageNet pretrained
    Dense connectivity
    7.9M parameters"]
    BB --> GAP2["Adaptive Avg Pool
    1024-dim features"]
    GAP2 --> FL["Feature Layer
    Linear 1024→512
    ReLU · Dropout(0.3)"]
    FL --> MLH["Multilabel Head
    Linear 512→14
    sigmoid · 14 pathologies"]
    FL --> BH["Binary Head
    Linear 512→1
    sigmoid · Normal/Abnormal"]
    style MLH fill:#2e7d32,color:#fff
    style BH fill:#1565c0,color:#fff
    style IN fill:#37474f,color:#fff
    style BB fill:#6a1b9a,color:#fff
```"""


def _render_densenet_finetuning(config: dict[str, Any]) -> str:
    """Mermaid fine-tuning phase diagram for DenseNet."""
    ft = config.get("model", {}).get("fine_tuning", {})
    freeze_epochs = ft.get("freeze_epochs", 5)
    total_epochs = config.get("training", {}).get("epochs", 60)
    unfreeze_lr = ft.get("unfreeze_lr", 1e-4)
    freeze_lr = ft.get("freeze_lr", 1e-3)
    return f"""```mermaid
graph LR
    P1["🔒 Phase 1
    Epochs 1–{freeze_epochs}
    Backbone frozen
    Train heads only
    lr = {freeze_lr}"] -->|"Epoch {freeze_epochs + 1}
    unfreeze_backbone()"| P2["🔓 Phase 2
    Epochs {freeze_epochs + 1}–{total_epochs}
    End-to-end fine-tuning
    All layers trainable
    lr = {unfreeze_lr}"]
    style P1 fill:#e65100,color:#fff
    style P2 fill:#6a1b9a,color:#fff
```"""


def _render_per_class_auc_table(best_val_metrics: dict[str, Any]) -> str:
    """Render a markdown table of per-class AUC-ROC from best epoch metrics."""
    rows = []
    for label in PATHOLOGY_LABELS:
        auc = best_val_metrics.get(f"auc_{label}")
        if auc is not None:
            bar_filled = int(round(float(auc) * 10))
            bar = "█" * bar_filled + "░" * (10 - bar_filled)
            rows.append(f"| {label:<20} | `{float(auc):.4f}` | `{bar}` |")

    if not rows:
        return ""

    table = "| Pathology            | AUC-ROC  | Visual        |\n"
    table += "|----------------------|----------|---------------|\n"
    table += "\n".join(rows)
    return table


def render_model_card(
    repo_id: str,
    checkpoint: dict[str, Any],
    history: dict[str, Any] | None = None,
) -> str:
    """Render a Hugging Face model card with architecture diagrams and training metrics."""
    config = checkpoint.get("config", {})
    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    train_cfg = config.get("training", {})
    model_name = model_cfg.get("name", repo_id.split("/")[-1])
    model_type = model_cfg.get("type", "")
    dataset_repo = data_cfg.get("hf_dataset_repo", HF_DATASET_REPO)
    dataset_revision = data_cfg.get("hf_dataset_revision", HF_DATASET_REVISION)
    best_auc = checkpoint.get("best_auc")
    epoch = checkpoint.get("epoch")
    best_val_metrics: dict[str, Any] = checkpoint.get("best_val_metrics", {})
    best_binary_auc = _safe_metric(history, "binary_auc_roc")
    best_binary_f1 = _safe_metric(history, "binary_f1")

    # --- Metrics summary ---
    metrics_lines = []
    if isinstance(best_auc, (int, float)):
        metrics_lines.append(f"- Best validation macro AUC-ROC: `{best_auc:.4f}`")
    if isinstance(best_binary_auc, float):
        metrics_lines.append(f"- Best validation binary AUC-ROC: `{best_binary_auc:.4f}`")
    if isinstance(best_binary_f1, float):
        metrics_lines.append(f"- Best validation binary F1: `{best_binary_f1:.4f}`")
    if epoch is not None:
        metrics_lines.append(f"- Best checkpoint epoch: `{epoch}`")
    metrics_block = (
        "\n".join(metrics_lines)
        if metrics_lines
        else "- Metrics will appear after the first successful training run."
    )

    # --- Architecture diagram ---
    if model_type == "densenet":
        arch_diagram = _render_densenet_architecture()
    else:
        arch_diagram = _render_scratch_architecture(config)

    # --- Fine-tuning diagram (DenseNet only) ---
    finetuning_section = ""
    if model_type == "densenet":
        finetuning_section = f"""
## Fine-Tuning Strategy

{_render_densenet_finetuning(config)}
"""

    # --- Per-class AUC table ---
    per_class_section = ""
    if best_val_metrics:
        table = _render_per_class_auc_table(best_val_metrics)
        if table:
            per_class_section = f"""
## Per-Class AUC-ROC at Best Epoch

{table}
"""

    # --- Architecture summary line ---
    arch_summary = _architecture_summary(config)

    # --- AMP / training details ---
    use_amp = train_cfg.get("use_amp", False)
    use_clahe = data_cfg.get("clahe", False)
    label_smoothing = train_cfg.get("label_smoothing", 0.0)
    grad_accum = train_cfg.get("grad_accum_steps", 1)
    effective_batch = train_cfg.get("batch_size", 32) * grad_accum
    training_details = (
        f"- Batch size: `{train_cfg.get('batch_size', 32)}` "
        f"× grad_accum `{grad_accum}` = **effective batch `{effective_batch}`**\n"
        f"- AMP (fp16): `{'enabled' if use_amp else 'disabled'}`\n"
        f"- CLAHE preprocessing: `{'enabled' if use_clahe else 'disabled'}`\n"
        f"- Label smoothing: `{label_smoothing}`\n"
        f"- Optimizer: AdamW  ·  Scheduler: CosineAnnealingLR\n"
        f"- Epochs configured: `{train_cfg.get('epochs', '?')}`  ·  "
        f"Early stop patience: `{train_cfg.get('early_stopping_patience', 10)}`"
    )

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
- multi-label-classification
datasets:
- {dataset_repo}
---

# {model_name}

> **CheXVision** — Deep Learning & Big Data university project.
> 14-class chest X-ray pathology detection + binary normal/abnormal classification
> on the NIH Chest X-ray14 dataset (112,120 images).

## Architecture

{arch_diagram}
{finetuning_section}
## Training Pipeline

{_render_pipeline_diagram()}

## Training Metrics

{metrics_block}

{per_class_section}
## Training Configuration

- Repository: `{repo_id}`
- Dataset: [{dataset_repo}](https://huggingface.co/datasets/{dataset_repo}) · revision `{dataset_revision}`
- Architecture: {arch_summary}
- Platform: Kaggle GPU kernel (NVIDIA T4 / P100)
{training_details}

## Intended Use

This model is intended for research and educational work on automated chest X-ray pathology detection.
It outputs two predictions per image:
1. **Multi-label scores** — independent sigmoid probability for each of 14 NIH pathologies
2. **Binary score** — sigmoid probability of any abnormality (Normal vs. Abnormal)

## Limitations

- Not validated for clinical use. Predictions must not substitute professional medical judgment.
- Trained on NIH Chest X-ray14, which contains noisy radiologist annotations (patient-level labels, not lesion-level).
- Performance degrades on images from equipment, patient populations, or preprocessing pipelines
  that differ from the NIH training distribution.
- Reported AUC metrics are on the validation split, not the held-out test set.

## CheXNet Benchmark Context

CheXNet (Rajpurkar et al., 2017) — the seminal paper establishing DenseNet-121 for chest X-ray
classification — reported **0.841 macro AUC-ROC** on a comparable split of this dataset.
CheXVision-DenseNet matches this benchmark. See the
[CheXVision demo](https://huggingface.co/spaces/HlexNC/chexvision-demo) for live inference.

## Citation

```bibtex
@misc{{chexvision2026,
  title={{CheXVision: Dual-Task Chest X-ray Classification with Custom CNN and DenseNet-121}},
  author={{BIG D(ATA) Team}},
  year={{2026}},
  howpublished={{\\url{{https://huggingface.co/{repo_id}}}}}
}}
```
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
