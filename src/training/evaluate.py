"""Model evaluation and comparison script."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset import PATHOLOGY_LABELS, ChestXrayDataset
from src.models.densenet_transfer import CheXVisionDenseNet
from src.models.scratch_cnn import CheXVisionScratch
from src.training.metrics import compute_binary_metrics, compute_multilabel_metrics
from src.training.trainer import set_seed

logger = logging.getLogger(__name__)


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
    """Load a trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]

    if config["model"]["type"] == "scratch":
        arch = config["model"].get("architecture", {})
        model = CheXVisionScratch(
            in_channels=3,
            num_classes=14,
            block_config=tuple(arch.get("block_config", [2, 2, 2, 2])),
            filter_sizes=tuple(arch.get("filter_sizes", [64, 128, 256, 512])),
            dropout=arch.get("dropout", 0.5),
        )
    else:
        arch = config["model"].get("architecture", {})
        model = CheXVisionDenseNet(
            num_classes=14,
            pretrained=False,
            dropout=arch.get("dropout", 0.3),
        )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, config


@torch.no_grad()
def predict(model: torch.nn.Module, dataloader: DataLoader, device: torch.device) -> dict[str, np.ndarray]:
    """Run inference and collect all predictions."""
    all_ml_probs, all_ml_targets = [], []
    all_bin_probs, all_bin_targets = [], []

    for batch in dataloader:
        images = batch["image"].to(device)
        outputs = model(images)

        all_ml_probs.append(torch.sigmoid(outputs["multilabel_logits"]).cpu().numpy())
        all_ml_targets.append(batch["multilabel_target"].numpy())
        all_bin_probs.append(torch.sigmoid(outputs["binary_logits"]).cpu().numpy())
        all_bin_targets.append(batch["binary_target"].numpy())

    return {
        "ml_probs": np.concatenate(all_ml_probs),
        "ml_targets": np.concatenate(all_ml_targets),
        "bin_probs": np.concatenate(all_bin_probs).squeeze(-1),
        "bin_targets": np.concatenate(all_bin_targets).squeeze(-1),
    }


def compare_models(results: dict[str, dict], output_dir: Path) -> None:
    """Generate comparison plots and summary."""
    output_dir.mkdir(parents=True, exist_ok=True)

    model_names = list(results.keys())

    # Per-class AUC comparison
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(PATHOLOGY_LABELS))
    width = 0.35

    for i, name in enumerate(model_names):
        aucs = [results[name]["ml_metrics"].get(f"auc_{label}", 0) for label in PATHOLOGY_LABELS]
        ax.bar(x + i * width, aucs, width, label=name)

    ax.set_xlabel("Pathology")
    ax.set_ylabel("AUC-ROC")
    ax.set_title("Per-Class AUC-ROC Comparison")
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(PATHOLOGY_LABELS, rotation=45, ha="right")
    ax.legend()
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(output_dir / "auc_comparison.png", dpi=150)
    plt.close()

    # Summary table
    summary = {}
    for name in model_names:
        summary[name] = {
            "macro_auc": results[name]["ml_metrics"]["auc_roc_macro"],
            "macro_f1": results[name]["ml_metrics"]["f1_macro"],
            "binary_auc": results[name]["bin_metrics"].get("binary_auc_roc", 0),
            "binary_f1": results[name]["bin_metrics"]["binary_f1"],
            "binary_accuracy": results[name]["bin_metrics"]["binary_accuracy"],
        }

    with open(output_dir / "comparison_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("Comparison results saved to %s", output_dir)
    for name, metrics in summary.items():
        logger.info("  %s: Macro AUC=%.4f, Binary AUC=%.4f", name, metrics["macro_auc"], metrics["binary_auc"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate and compare CheXVision models")
    parser.add_argument("--model-dir", type=Path, default=Path("checkpoints"), help="Directory with model checkpoints")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Data directory")
    parser.add_argument("--output-dir", type=Path, default=Path("results"), help="Output directory for plots")
    parser.add_argument("--compare", action="store_true", help="Compare all models in model-dir")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load test dataset
    test_dataset = ChestXrayDataset(args.data_dir / "images", args.data_dir / "labels.csv", split="test")
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Evaluate all checkpoints
    results = {}
    for ckpt_path in sorted(args.model_dir.glob("*_best.pth")):
        logger.info("Evaluating %s", ckpt_path.name)
        model, config = load_model(ckpt_path, device)
        preds = predict(model, test_loader, device)

        ml_metrics = compute_multilabel_metrics(preds["ml_targets"], (preds["ml_probs"] >= 0.5).astype(int), preds["ml_probs"])
        bin_metrics = compute_binary_metrics(preds["bin_targets"], (preds["bin_probs"] >= 0.5).astype(int), preds["bin_probs"])

        name = config["model"].get("name", ckpt_path.stem)
        results[name] = {"ml_metrics": ml_metrics, "bin_metrics": bin_metrics}

    if args.compare and len(results) >= 2:
        compare_models(results, args.output_dir)
    elif results:
        for name, r in results.items():
            logger.info("%s — Macro AUC: %.4f, Binary AUC: %.4f", name, r["ml_metrics"]["auc_roc_macro"], r["bin_metrics"].get("binary_auc_roc", 0))


if __name__ == "__main__":
    main()
