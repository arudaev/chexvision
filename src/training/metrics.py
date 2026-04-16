"""Evaluation metrics for chest X-ray classification."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.data.dataset import PATHOLOGY_LABELS


def compute_multilabel_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute metrics for multi-label classification.

    Args:
        y_true: Ground truth labels, shape (N, 14).
        y_pred: Predicted binary labels, shape (N, 14).
        y_prob: Predicted probabilities, shape (N, 14).
        threshold: Decision threshold for converting probabilities to binary predictions.

    Returns:
        Dictionary of metric names to values.
    """
    metrics: dict[str, float] = {}

    # Per-class AUC-ROC
    for i, label in enumerate(PATHOLOGY_LABELS):
        if y_true[:, i].sum() > 0:  # Need positive samples for AUC
            metrics[f"auc_{label}"] = float(roc_auc_score(y_true[:, i], y_prob[:, i]))

    # Macro-averaged AUC-ROC (primary metric)
    valid_aucs = [v for k, v in metrics.items() if k.startswith("auc_")]
    metrics["auc_roc_macro"] = float(np.mean(valid_aucs)) if valid_aucs else 0.0

    # F1, Precision, Recall (macro-averaged)
    y_pred_binary = (y_prob >= threshold).astype(int)
    metrics["f1_macro"] = float(f1_score(y_true, y_pred_binary, average="macro", zero_division=0))
    metrics["precision_macro"] = float(precision_score(y_true, y_pred_binary, average="macro", zero_division=0))
    metrics["recall_macro"] = float(recall_score(y_true, y_pred_binary, average="macro", zero_division=0))

    return metrics


def compute_binary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> dict[str, float]:
    """Compute metrics for binary classification (Normal vs Abnormal).

    Args:
        y_true: Ground truth labels, shape (N,).
        y_pred: Predicted binary labels, shape (N,).
        y_prob: Predicted probabilities, shape (N,).

    Returns:
        Dictionary of metric names to values.
    """
    metrics: dict[str, float] = {}

    metrics["binary_accuracy"] = float(accuracy_score(y_true, y_pred))
    metrics["binary_f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    metrics["binary_precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    metrics["binary_recall"] = float(recall_score(y_true, y_pred, zero_division=0))

    if y_true.sum() > 0:
        metrics["binary_auc_roc"] = float(roc_auc_score(y_true, y_prob))

    metrics["binary_confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()

    return metrics


def combine_losses(
    multilabel_loss: torch.Tensor,
    binary_loss: torch.Tensor,
    multilabel_weight: float = 1.0,
    binary_weight: float = 0.5,
) -> torch.Tensor:
    """Weighted combination of multi-label and binary task losses."""
    return multilabel_weight * multilabel_loss + binary_weight * binary_loss
