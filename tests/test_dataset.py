"""Unit tests for ChestXrayDataset label encoding and utilities."""

from __future__ import annotations

import numpy as np

from src.data.dataset import NUM_CLASSES, PATHOLOGY_LABELS


class TestPathologyLabels:
    def test_count(self) -> None:
        assert len(PATHOLOGY_LABELS) == 14

    def test_num_classes_matches(self) -> None:
        assert NUM_CLASSES == 14

    def test_no_duplicates(self) -> None:
        assert len(PATHOLOGY_LABELS) == len(set(PATHOLOGY_LABELS))


class TestMultilabelEncoding:
    """Test the _encode_multilabel logic without loading images from disk."""

    @staticmethod
    def _encode_multilabel(labels_list: list[str]) -> np.ndarray:
        """Standalone copy of the encoding logic for testing."""
        import pandas as pd

        targets = np.zeros((len(labels_list), NUM_CLASSES), dtype=np.float32)
        for i, labels_str in enumerate(labels_list):
            if labels_str == "No Finding" or pd.isna(labels_str):
                continue
            for label in labels_str.split("|"):
                label = label.strip()
                if label in PATHOLOGY_LABELS:
                    targets[i, PATHOLOGY_LABELS.index(label)] = 1.0
        return targets

    def test_no_finding(self) -> None:
        targets = self._encode_multilabel(["No Finding"])
        assert targets.shape == (1, 14)
        assert targets.sum() == 0.0

    def test_single_pathology(self) -> None:
        targets = self._encode_multilabel(["Effusion"])
        assert targets[0, PATHOLOGY_LABELS.index("Effusion")] == 1.0
        assert targets.sum() == 1.0

    def test_multiple_pathologies(self) -> None:
        targets = self._encode_multilabel(["Effusion|Fibrosis"])
        assert targets[0, PATHOLOGY_LABELS.index("Effusion")] == 1.0
        assert targets[0, PATHOLOGY_LABELS.index("Fibrosis")] == 1.0
        assert targets.sum() == 2.0

    def test_all_pathologies(self) -> None:
        labels_str = "|".join(PATHOLOGY_LABELS)
        targets = self._encode_multilabel([labels_str])
        assert targets.sum() == 14.0
        np.testing.assert_array_equal(targets[0], np.ones(14, dtype=np.float32))

    def test_batch_encoding(self) -> None:
        targets = self._encode_multilabel(["Effusion", "No Finding", "Mass|Nodule"])
        assert targets.shape == (3, 14)
        assert targets[0].sum() == 1.0  # Effusion
        assert targets[1].sum() == 0.0  # No Finding
        assert targets[2].sum() == 2.0  # Mass + Nodule


class TestBinaryEncoding:
    @staticmethod
    def _encode_binary(labels_list: list[str]) -> np.ndarray:
        import pandas as pd

        targets = np.zeros((len(labels_list), 1), dtype=np.float32)
        for i, labels_str in enumerate(labels_list):
            if labels_str != "No Finding" and not pd.isna(labels_str):
                targets[i, 0] = 1.0
        return targets

    def test_normal(self) -> None:
        targets = self._encode_binary(["No Finding"])
        assert targets[0, 0] == 0.0

    def test_abnormal(self) -> None:
        targets = self._encode_binary(["Effusion"])
        assert targets[0, 0] == 1.0

    def test_multi_pathology_is_abnormal(self) -> None:
        targets = self._encode_binary(["Effusion|Fibrosis"])
        assert targets[0, 0] == 1.0

    def test_batch_binary(self) -> None:
        targets = self._encode_binary(["No Finding", "Effusion", "No Finding"])
        assert targets.sum() == 1.0
        assert targets[1, 0] == 1.0
