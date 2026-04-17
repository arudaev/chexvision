"""Unit tests for evaluation metrics."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.training.metrics import combine_losses, compute_binary_metrics, compute_multilabel_metrics


class TestCombineLosses:
    def test_default_weights(self) -> None:
        ml_loss = torch.tensor(2.0)
        bin_loss = torch.tensor(1.0)
        combined = combine_losses(ml_loss, bin_loss)
        assert combined.item() == pytest.approx(2.5)  # 1.0*2.0 + 0.5*1.0

    def test_custom_weights(self) -> None:
        ml_loss = torch.tensor(1.0)
        bin_loss = torch.tensor(1.0)
        combined = combine_losses(ml_loss, bin_loss, multilabel_weight=0.7, binary_weight=0.3)
        assert combined.item() == pytest.approx(1.0)  # 0.7 + 0.3

    def test_zero_losses(self) -> None:
        combined = combine_losses(torch.tensor(0.0), torch.tensor(0.0))
        assert combined.item() == 0.0


class TestMultilabelMetrics:
    def test_perfect_predictions(self) -> None:
        # Must use 14 classes to match PATHOLOGY_LABELS
        n_classes = 14
        y_true = np.eye(n_classes, dtype=np.float64)  # one positive per class
        y_prob = np.eye(n_classes, dtype=np.float64) * 0.8 + 0.1  # high prob on diagonal
        y_pred = (y_prob >= 0.5).astype(int)
        metrics = compute_multilabel_metrics(y_true, y_pred, y_prob)
        assert metrics["f1_macro"] == pytest.approx(1.0)

    def test_all_zero_predictions(self) -> None:
        y_true = np.zeros((10, 14), dtype=np.float64)
        y_true[:5, 0] = 1.0  # first 5 samples have class 0
        y_pred = np.zeros_like(y_true)
        y_prob = np.full_like(y_true, 0.1)
        metrics = compute_multilabel_metrics(y_true, y_pred, y_prob)
        assert "f1_macro" in metrics
        assert metrics["f1_macro"] == 0.0

    def test_returns_per_class_auc(self) -> None:
        rng = np.random.RandomState(42)
        y_true = rng.randint(0, 2, size=(50, 14))
        y_prob = rng.random(size=(50, 14))
        y_pred = (y_prob >= 0.5).astype(int)
        metrics = compute_multilabel_metrics(y_true, y_pred, y_prob)
        # Should have per-class AUC for classes with positive samples
        auc_keys = [k for k in metrics if k.startswith("auc_")]
        assert len(auc_keys) > 0
        assert "auc_roc_macro" in metrics


class TestBinaryMetrics:
    def test_perfect_predictions(self) -> None:
        y_true = np.array([0, 1, 1, 0])
        y_pred = np.array([0, 1, 1, 0])
        y_prob = np.array([0.1, 0.9, 0.8, 0.2])
        metrics = compute_binary_metrics(y_true, y_pred, y_prob)
        assert metrics["binary_accuracy"] == 1.0
        assert metrics["binary_f1"] == 1.0
        assert metrics["binary_auc_roc"] == pytest.approx(1.0)

    def test_confusion_matrix_shape(self) -> None:
        y_true = np.array([0, 1, 1, 0])
        y_pred = np.array([0, 1, 0, 1])
        y_prob = np.array([0.3, 0.7, 0.4, 0.6])
        metrics = compute_binary_metrics(y_true, y_pred, y_prob)
        cm = metrics["binary_confusion_matrix"]
        assert len(cm) == 2
        assert len(cm[0]) == 2

    def test_all_same_predictions(self) -> None:
        y_true = np.array([0, 1, 1, 0])
        y_pred = np.zeros(4, dtype=int)
        y_prob = np.full(4, 0.3)
        metrics = compute_binary_metrics(y_true, y_pred, y_prob)
        assert "binary_accuracy" in metrics
        assert metrics["binary_accuracy"] == 0.5
