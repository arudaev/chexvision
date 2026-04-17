"""Tests for the dataset download helper."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image

from src.data import download as download_module


def test_download_dataset_snapshot_writes_images_and_labels(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """The download helper should materialize images and normalize split names."""
    snapshot_dir = tmp_path / "snapshot"
    output_dir = tmp_path / "data"

    def fake_snapshot_dataset_repo(repo_id: str, revision: str, target_dir: Path, token: str | None) -> Path:
        assert repo_id == download_module.HF_DATASET_REPO
        assert revision == "rev-test"
        assert target_dir == snapshot_dir
        assert token == "hf_test_token"
        return target_dir

    def fake_load_streaming_dataset(local_snapshot_dir: Path):
        assert local_snapshot_dir == snapshot_dir
        image = Image.new("L", (16, 16), 128)
        return {
            "train": [
                {"image": image.copy(), "labels": "No Finding", "filename": "train_a.png"},
            ],
            "validation": [
                {"image": image.copy(), "labels": "Effusion", "filename": "val_a.png"},
            ],
            "test": [
                {"image": image.copy(), "labels": "Mass|Nodule", "filename": "test_a.png"},
            ],
        }

    monkeypatch.setattr(download_module, "configure_hf_runtime", lambda: "hf_test_token")
    monkeypatch.setattr(download_module, "_snapshot_dataset_repo", fake_snapshot_dataset_repo)
    monkeypatch.setattr(download_module, "_load_streaming_dataset", fake_load_streaming_dataset)

    download_module.download_dataset(
        output_dir,
        revision="rev-test",
        snapshot_dir=snapshot_dir,
    )

    labels = pd.read_csv(output_dir / "labels.csv")
    assert labels["split"].tolist() == ["train", "val", "test"]
    assert labels["labels"].tolist() == ["No Finding", "Effusion", "Mass|Nodule"]

    for image_path in labels["image_path"]:
        assert (output_dir / image_path).exists()


def test_build_local_parquet_files_uses_snapshot_layout(tmp_path: Path) -> None:
    """The parquet loader should point at the pinned local snapshot layout."""
    data_files = download_module._build_local_parquet_files(tmp_path / "snapshot")
    assert data_files["train"].endswith("data\\train-*.parquet") or data_files["train"].endswith("data/train-*.parquet")
    assert data_files["validation"].endswith("validation-*.parquet")
    assert data_files["test"].endswith("test-*.parquet")


def test_default_snapshot_dir_prefers_env(monkeypatch, tmp_path: Path) -> None:
    """An explicit snapshot dir env var should override the default path."""
    desired_dir = tmp_path / "custom_snapshot"
    monkeypatch.setenv(download_module.SNAPSHOT_DIR_ENV, str(desired_dir))

    snapshot_dir = download_module._default_snapshot_dir(tmp_path / "data")

    assert snapshot_dir == desired_dir
