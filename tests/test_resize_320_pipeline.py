"""Tests for the resize_320 Kaggle dataset pipeline."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from src.data import resize_320_pipeline as pipeline


def test_download_source_file_uses_local_dir_instead_of_global_cache(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Raw source downloads should land in a pipeline-owned local folder."""
    calls: list[dict[str, object]] = []

    def fake_hf_hub_download(**kwargs):
        calls.append(kwargs)
        if kwargs.get("dry_run"):
            return SimpleNamespace(file_size=1024)
        return str(Path(kwargs["local_dir"]) / kwargs["filename"])

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_hf_hub_download)
    monkeypatch.setattr(
        pipeline.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(
            total=10 * 1024 * 1024 * 1024,
            used=1024,
            free=2 * 1024 * 1024 * 1024,
        ),
    )

    config = pipeline.PipelineConfig(work_dir=tmp_path, is_kaggle=True)
    zip_path = pipeline.download_source_file(
        config,
        token="hf_test_token",
        filename="data/images/images_001.zip",
        download_name="source_zip",
    )

    assert zip_path == tmp_path / "source_zip" / "data" / "images" / "images_001.zip"
    assert calls[0]["dry_run"] is True
    assert "local_dir" not in calls[0]
    assert calls[1]["local_dir"] == str(tmp_path / "source_zip")
    assert calls[1]["repo_id"] == pipeline.SOURCE_REPO
    assert calls[1]["repo_type"] == "dataset"


def test_ensure_disk_headroom_raises_before_download(monkeypatch, tmp_path: Path) -> None:
    """Large raw ZIP downloads should fail early with a clear disk-space error."""
    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download",
        lambda **_kwargs: SimpleNamespace(file_size=5 * 1024 * 1024 * 1024),
    )
    monkeypatch.setattr(
        pipeline.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(total=10_000, used=9_500, free=500),
    )

    with pytest.raises(RuntimeError, match="Not enough free disk space"):
        pipeline.ensure_disk_headroom(
            tmp_path / "source_zip",
            repo_id=pipeline.SOURCE_REPO,
            filename="data/images/images_006.zip",
            repo_type="dataset",
            token="hf_test_token",
        )


def test_cleanup_local_download_dir_removes_hf_local_dir(tmp_path: Path) -> None:
    """Throwaway local download folders should be fully removable after each ZIP."""
    config = pipeline.PipelineConfig(work_dir=tmp_path)
    download_dir = pipeline.local_download_dir(config, "source_zip")
    cached_file = download_dir / ".cache" / "huggingface" / "metadata.json"
    cached_file.parent.mkdir(parents=True, exist_ok=True)
    cached_file.write_text("{}", encoding="utf-8")

    pipeline.cleanup_local_download_dir(config, "source_zip")

    assert not download_dir.exists()
