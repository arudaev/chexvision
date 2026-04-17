"""Main training loop for CheXVision models.

Uses raw PyTorch (no Lightning) so every design decision is explicit and justifiable in the report.

CLOUD-ONLY: This script is designed to run on cloud GPU environments (Google Colab, HF Spaces,
university cluster). Do NOT run training on local machines — use notebooks in Colab instead.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import ChestXrayDataset
from src.models.densenet_transfer import CheXVisionDenseNet
from src.models.scratch_cnn import CheXVisionScratch
from src.training.metrics import combine_losses, compute_binary_metrics, compute_multilabel_metrics

logger = logging.getLogger(__name__)


def _check_cloud_environment() -> None:
    """Verify we're running in a cloud/GPU environment, not a local dev machine.

    Raises SystemExit if running locally without an explicit override.
    Set CHEXVISION_ALLOW_LOCAL=1 to bypass (for unit tests only).
    """
    allow_local = os.environ.get("CHEXVISION_ALLOW_LOCAL", "0") == "1"
    if allow_local:
        return

    in_colab = "google.colab" in sys.modules or os.environ.get("COLAB_GPU") is not None
    in_kaggle = os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is not None
    in_hf_space = os.environ.get("SPACE_ID") is not None
    in_github_actions = os.environ.get("GITHUB_ACTIONS") is not None
    in_slurm = os.environ.get("SLURM_JOB_ID") is not None  # university cluster
    has_gpu = torch.cuda.is_available()

    is_cloud = in_colab or in_kaggle or in_hf_space or in_github_actions or in_slurm

    if not is_cloud and not has_gpu:
        logger.error(
            "CLOUD-ONLY POLICY: Training must run on cloud GPU environments "
            "(Google Colab, Kaggle, HF Spaces, university SLURM cluster). "
            "Local CPU training on 112k images is impractical. "
            "Use the Colab notebooks in notebooks/ instead. "
            "Override with CHEXVISION_ALLOW_LOCAL=1 for testing only."
        )
        sys.exit(1)

    if not is_cloud and has_gpu:
        logger.warning(
            "Local GPU detected. Training will proceed, but consider using "
            "Google Colab or university cluster for reproducibility and collaboration."
        )


def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_model(config: dict) -> nn.Module:
    """Instantiate model based on config."""
    model_type = config["model"]["type"]

    if model_type == "scratch":
        arch = config["model"].get("architecture", {})
        return CheXVisionScratch(
            in_channels=3,
            num_classes=14,
            block_config=tuple(arch.get("block_config", [2, 2, 2, 2])),
            filter_sizes=tuple(arch.get("filter_sizes", [64, 128, 256, 512])),
            dropout=arch.get("dropout", 0.5),
        )
    elif model_type == "densenet":
        arch = config["model"].get("architecture", {})
        return CheXVisionDenseNet(
            num_classes=14,
            pretrained=arch.get("pretrained", True),
            dropout=arch.get("dropout", 0.3),
            freeze_backbone=True,  # Start frozen (Phase 1)
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    pos_weight: torch.Tensor | None = None,
    multilabel_weight: float = 1.0,
    binary_weight: float = 0.5,
) -> dict[str, float]:
    """Train for one epoch. Returns average losses."""
    model.train()
    total_multilabel_loss = 0.0
    total_binary_loss = 0.0
    total_combined_loss = 0.0
    num_batches = 0

    # Loss functions
    multilabel_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(device)
    binary_criterion = nn.BCEWithLogitsLoss().to(device)

    for batch in tqdm(dataloader, desc="Training", leave=False):
        images = batch["image"].to(device)
        ml_targets = batch["multilabel_target"].to(device)
        bin_targets = batch["binary_target"].to(device)

        # Forward pass
        outputs = model(images)
        ml_loss = multilabel_criterion(outputs["multilabel_logits"], ml_targets)
        bin_loss = binary_criterion(outputs["binary_logits"], bin_targets)
        loss = combine_losses(ml_loss, bin_loss, multilabel_weight, binary_weight)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_multilabel_loss += ml_loss.item()
        total_binary_loss += bin_loss.item()
        total_combined_loss += loss.item()
        num_batches += 1

    return {
        "train_multilabel_loss": total_multilabel_loss / max(num_batches, 1),
        "train_binary_loss": total_binary_loss / max(num_batches, 1),
        "train_combined_loss": total_combined_loss / max(num_batches, 1),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    pos_weight: torch.Tensor | None = None,
) -> dict[str, float]:
    """Evaluate model on validation/test set."""
    model.eval()

    all_ml_probs = []
    all_ml_targets = []
    all_bin_probs = []
    all_bin_targets = []

    multilabel_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(device)
    binary_criterion = nn.BCEWithLogitsLoss().to(device)
    total_ml_loss = 0.0
    total_bin_loss = 0.0
    num_batches = 0

    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        images = batch["image"].to(device)
        ml_targets = batch["multilabel_target"].to(device)
        bin_targets = batch["binary_target"].to(device)

        outputs = model(images)

        total_ml_loss += multilabel_criterion(outputs["multilabel_logits"], ml_targets).item()
        total_bin_loss += binary_criterion(outputs["binary_logits"], bin_targets).item()
        num_batches += 1

        # Collect predictions
        all_ml_probs.append(torch.sigmoid(outputs["multilabel_logits"]).cpu().numpy())
        all_ml_targets.append(ml_targets.cpu().numpy())
        all_bin_probs.append(torch.sigmoid(outputs["binary_logits"]).cpu().numpy())
        all_bin_targets.append(bin_targets.cpu().numpy())

    # Concatenate
    ml_probs = np.concatenate(all_ml_probs)
    ml_targets = np.concatenate(all_ml_targets)
    bin_probs = np.concatenate(all_bin_probs).squeeze(-1)
    bin_targets = np.concatenate(all_bin_targets).squeeze(-1)

    # Compute metrics
    ml_metrics = compute_multilabel_metrics(ml_targets, (ml_probs >= 0.5).astype(int), ml_probs)
    bin_metrics = compute_binary_metrics(bin_targets, (bin_probs >= 0.5).astype(int), bin_probs)

    return {
        "val_multilabel_loss": total_ml_loss / max(num_batches, 1),
        "val_binary_loss": total_bin_loss / max(num_batches, 1),
        **ml_metrics,
        **bin_metrics,
    }


def train(config: dict) -> None:
    """Full training pipeline. Must run on cloud GPU (Colab, HF, SLURM)."""
    _check_cloud_environment()
    set_seed(config.get("seed", 42))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    train_cfg = config["training"]
    data_cfg = config["data"]

    # Build datasets
    data_dir = Path(data_cfg.get("data_dir", "data"))
    # download.py stores paths in labels.csv as "images/{filename}".
    # ChestXrayDataset joins image_dir with each path, so pass data_dir directly —
    # not data_dir/"images" — to avoid a doubled images/images/ prefix.
    labels_csv = data_dir / "labels.csv"
    train_dataset = ChestXrayDataset(data_dir, labels_csv, split="train", image_size=data_cfg["image_size"])
    val_dataset = ChestXrayDataset(data_dir, labels_csv, split="val", image_size=data_cfg["image_size"])

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=data_cfg.get("num_workers", 4),
        pin_memory=data_cfg.get("pin_memory", True),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=data_cfg.get("num_workers", 4),
        pin_memory=data_cfg.get("pin_memory", True),
    )

    # Build model
    model = build_model(config).to(device)
    pos_weight = train_dataset.get_label_weights().to(device)

    # Optimizer & scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg.get("weight_decay", 1e-4),
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=train_cfg["epochs"])

    # Training loop
    best_auc = 0.0
    patience_counter = 0
    checkpoint_dir = Path(config.get("logging", {}).get("checkpoint_dir", "checkpoints"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # History tracking for report figures
    history: dict[str, list[float]] = {
        "train_combined_loss": [],
        "val_multilabel_loss": [],
        "auc_roc_macro": [],
        "binary_auc_roc": [],
        "binary_f1": [],
        "learning_rate": [],
    }

    for epoch in range(1, train_cfg["epochs"] + 1):
        logger.info("Epoch %d/%d", epoch, train_cfg["epochs"])

        # Handle DenseNet Phase 2: unfreeze backbone
        if config["model"]["type"] == "densenet":
            freeze_epochs = config["model"].get("fine_tuning", {}).get("freeze_epochs", 5)
            if epoch == freeze_epochs + 1:
                logger.info("Phase 2: Unfreezing DenseNet backbone")
                model.unfreeze_backbone()
                unfreeze_lr = config["model"]["fine_tuning"].get("unfreeze_lr", 1e-4)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = unfreeze_lr

        # Train
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, device, pos_weight,
            train_cfg.get("multilabel_weight", 1.0),
            train_cfg.get("binary_weight", 0.5),
        )
        scheduler.step()

        # Evaluate
        val_metrics = evaluate(model, val_loader, device, pos_weight)

        logger.info(
            "  Train Loss: %.4f | Val AUC-ROC: %.4f | Val Binary AUC: %.4f",
            train_metrics["train_combined_loss"],
            val_metrics.get("auc_roc_macro", 0),
            val_metrics.get("binary_auc_roc", 0),
        )

        # Record history
        history["train_combined_loss"].append(train_metrics["train_combined_loss"])
        history["val_multilabel_loss"].append(val_metrics.get("val_multilabel_loss", 0))
        history["auc_roc_macro"].append(val_metrics.get("auc_roc_macro", 0))
        history["binary_auc_roc"].append(val_metrics.get("binary_auc_roc", 0))
        history["binary_f1"].append(val_metrics.get("binary_f1", 0))
        history["learning_rate"].append(optimizer.param_groups[0]["lr"])

        # Save best model
        current_auc = val_metrics.get("auc_roc_macro", 0)
        if current_auc > best_auc:
            best_auc = current_auc
            patience_counter = 0
            model_name = config["model"].get("name", "model")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_auc": best_auc,
                "config": config,
            }, checkpoint_dir / f"{model_name}_best.pth")
            logger.info("  Saved best model (AUC: %.4f)", best_auc)
        else:
            patience_counter += 1
            if patience_counter >= train_cfg.get("early_stopping_patience", 10):
                logger.info("Early stopping at epoch %d", epoch)
                break

    # Save training history for report figures
    model_name = config["model"].get("name", "model")
    history_path = checkpoint_dir / f"{model_name}_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    logger.info("Training history saved to %s", history_path)

    logger.info("Training complete. Best AUC-ROC: %.4f", best_auc)


def _deep_merge(base: dict, overrides: dict) -> dict:
    """Recursively merge *overrides* into a copy of *base*."""
    merged = base.copy()
    for key, value in overrides.items():
        if key == "_defaults_":
            continue
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_config(config_path: Path) -> dict:
    """Load a YAML config, resolving ``_defaults_`` inheritance."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    if "_defaults_" in config:
        defaults_path = config_path.parent / config.pop("_defaults_")
        with open(defaults_path) as f:
            defaults = yaml.safe_load(f)
        config = _deep_merge(defaults, config)

    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CheXVision model")
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML config file")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    config = _load_config(args.config)
    train(config)


if __name__ == "__main__":
    main()
