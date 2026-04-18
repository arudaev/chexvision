"""PyTorch Dataset for NIH Chest X-ray14."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.data.transforms import get_eval_transforms, get_train_transforms

# All 14 pathology labels in the NIH Chest X-ray14 dataset
PATHOLOGY_LABELS = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
]

NUM_CLASSES = len(PATHOLOGY_LABELS)


class ChestXrayDataset(Dataset):
    """NIH Chest X-ray14 dataset for multi-label and binary classification.

    Each sample returns:
        image: Tensor of shape (C, H, W)
        multilabel_target: Tensor of shape (14,) — one-hot encoded pathology labels
        binary_target: Tensor of shape (1,) — 0 for Normal, 1 for Abnormal
    """

    def __init__(
        self,
        image_dir: Path | str,
        labels_csv: Path | str,
        split: str = "train",
        image_size: int = 224,
        transform: Any | None = None,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.split = split
        self.image_size = image_size

        # Load labels CSV
        df = pd.read_csv(labels_csv)
        if "split" in df.columns:
            df = df[df["split"] == split].reset_index(drop=True)

        self.image_paths = df["image_path"].tolist()
        self.labels_raw = df["labels"].tolist()

        # Set transforms
        if transform is not None:
            self.transform = transform
        elif split == "train":
            self.transform = get_train_transforms(image_size)
        else:
            self.transform = get_eval_transforms(image_size)

        # Precompute label vectors
        self._multilabel_targets = self._encode_multilabel(self.labels_raw)
        self._binary_targets = self._encode_binary(self.labels_raw)

    def _encode_multilabel(self, labels_list: list[str]) -> np.ndarray:
        """Convert string labels to multi-hot vectors."""
        targets = np.zeros((len(labels_list), NUM_CLASSES), dtype=np.float32)
        for i, labels_str in enumerate(labels_list):
            if labels_str == "No Finding" or pd.isna(labels_str):
                continue
            for label in labels_str.split("|"):
                label = label.strip()
                if label in PATHOLOGY_LABELS:
                    targets[i, PATHOLOGY_LABELS.index(label)] = 1.0
        return targets

    def _encode_binary(self, labels_list: list[str]) -> np.ndarray:
        """Convert labels to binary: 0=Normal, 1=Abnormal."""
        targets = np.zeros((len(labels_list), 1), dtype=np.float32)
        for i, labels_str in enumerate(labels_list):
            if labels_str != "No Finding" and not pd.isna(labels_str):
                targets[i, 0] = 1.0
        return targets

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        # Load image
        img_path = self.image_dir / self.image_paths[idx]
        pil_image = Image.open(img_path).convert("RGB")

        # Apply transforms — always required; get_train/eval_transforms() return a Tensor
        if self.transform is None:
            raise ValueError(
                "ChestXrayDataset requires a transform. "
                "Use get_train_transforms() or get_eval_transforms()."
            )
        image: torch.Tensor = self.transform(pil_image)

        return {
            "image": image,
            "multilabel_target": torch.from_numpy(self._multilabel_targets[idx]),
            "binary_target": torch.from_numpy(self._binary_targets[idx]),
        }

    def get_label_weights(self) -> torch.Tensor:
        """Compute positive class weights for handling class imbalance (pos_weight for BCEWithLogitsLoss)."""
        pos_counts = self._multilabel_targets.sum(axis=0)
        neg_counts = len(self) - pos_counts
        # Avoid division by zero
        pos_weights = neg_counts / np.maximum(pos_counts, 1.0)
        return torch.from_numpy(pos_weights.astype(np.float32))
