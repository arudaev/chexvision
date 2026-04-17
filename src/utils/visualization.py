"""Visualization utilities: Grad-CAM, ROC curves, training plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as functional
from sklearn.metrics import auc, roc_curve

from src.data.dataset import PATHOLOGY_LABELS


def plot_roc_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    labels: list[str] = PATHOLOGY_LABELS,
    save_path: Path | None = None,
) -> plt.Figure:
    """Plot ROC curves for each pathology class."""
    fig, ax = plt.subplots(figsize=(10, 8))

    for i, label in enumerate(labels):
        if y_true[:, i].sum() > 0:
            fpr, tpr, _ = roc_curve(y_true[:, i], y_prob[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f"{label} (AUC={roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Multi-Label Classification")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_confusion_matrix(
    cm: np.ndarray,
    labels: list[str] = ["Normal", "Abnormal"],
    save_path: Path | None = None,
) -> plt.Figure:
    """Plot confusion matrix for binary classification."""
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix — Binary Classification")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)

    return fig


def plot_training_history(
    history: dict[str, list[float]],
    save_path: Path | None = None,
) -> plt.Figure:
    """Plot training and validation loss/metric curves."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss curves
    if "train_combined_loss" in history:
        axes[0].plot(history["train_combined_loss"], label="Train Loss")
    if "val_multilabel_loss" in history:
        axes[0].plot(history["val_multilabel_loss"], label="Val ML Loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    # AUC-ROC
    if "auc_roc_macro" in history:
        axes[1].plot(history["auc_roc_macro"], label="Macro AUC-ROC", color="green")
    axes[1].set_title("Multi-Label AUC-ROC")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    # Binary metrics
    if "binary_auc_roc" in history:
        axes[2].plot(history["binary_auc_roc"], label="Binary AUC-ROC", color="orange")
    if "binary_f1" in history:
        axes[2].plot(history["binary_f1"], label="Binary F1", color="red")
    axes[2].set_title("Binary Classification")
    axes[2].set_xlabel("Epoch")
    axes[2].legend()

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)

    return fig


class GradCAM:
    """Grad-CAM implementation for model interpretability.

    Generates heatmaps showing which image regions the model focuses on for its predictions.
    Useful for verifying that the model attends to clinically relevant areas.
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.gradients: torch.Tensor | None = None
        self.activations: torch.Tensor | None = None

        # Register hooks
        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module: torch.nn.Module, input: tuple, output: torch.Tensor) -> None:
        self.activations = output.detach()

    def _backward_hook(self, module: torch.nn.Module, grad_input: tuple, grad_output: tuple) -> None:
        self.gradients = grad_output[0].detach()

    def generate(self, image: torch.Tensor, class_idx: int, task: str = "multilabel") -> np.ndarray:
        """Generate Grad-CAM heatmap for a given class.

        Args:
            image: Input image tensor (1, C, H, W).
            class_idx: Target class index.
            task: "multilabel" or "binary".

        Returns:
            Heatmap as numpy array (H, W), values in [0, 1].
        """
        self.model.eval()
        output = self.model(image)

        logits_key = f"{task}_logits"
        if task == "binary":
            target = output[logits_key].squeeze()
        else:
            target = output[logits_key][0, class_idx]

        self.model.zero_grad()
        target.backward()

        # Global average pooling of gradients
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = functional.relu(cam)

        # Resize to input image dimensions
        cam = functional.interpolate(cam, size=image.shape[2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        # Normalize to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 0:
            cam = (cam - cam_min) / (cam_max - cam_min)

        return cam
