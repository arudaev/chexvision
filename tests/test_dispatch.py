"""Tests for the Kaggle dispatch helper."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from scripts import dispatch


def test_build_kaggle_bundle_contains_project_sources() -> None:
    """The Kaggle bundle should embed the files the remote run needs."""
    bundle_dir = dispatch._build_kaggle_bundle("scratch")
    try:
        assert (bundle_dir / "kernel-metadata.json").exists()
        assert (bundle_dir / "script.py").exists()
        script = (bundle_dir / "script.py").read_text(encoding="utf-8")
        assert "git clone" not in script
        assert script.count(dispatch.BUNDLE_SENTINEL) == 1
        assert "PROJECT_BUNDLE_B64 =" in script
        assert "src.data.download" in script
        assert "HF_HUB_DISABLE_XET" in script
        assert "CHEXVISION_PARQUET_URLS_FILE" not in script

        metadata = json.loads((bundle_dir / "kernel-metadata.json").read_text(encoding="utf-8"))
        assert metadata["id"] == dispatch.KERNEL_SLUGS["scratch"]
        assert metadata["enable_gpu"] is True
        assert metadata["enable_internet"] is True
        assert metadata["code_file"] == "script.py"
    finally:
        shutil.rmtree(bundle_dir, ignore_errors=True)


def test_render_kernel_metadata_forces_training_runtime_flags(monkeypatch, tmp_path: Path) -> None:
    """Dispatch should force internet and GPU on for pushed training kernels."""
    project_root = tmp_path / "repo"
    kernel_dir = project_root / "kaggle" / "train_scratch"
    kernel_dir.mkdir(parents=True)
    metadata_path = kernel_dir / "kernel-metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "id": "someone/old-slug",
                "title": "Temp Kernel",
                "code_file": "old.py",
                "language": "python",
                "kernel_type": "script",
                "enable_gpu": False,
                "enable_internet": False,
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(dispatch, "PROJECT_ROOT", project_root)
    monkeypatch.setitem(dispatch.KERNEL_DIRS, "scratch", Path("kaggle/train_scratch"))

    metadata = dispatch._render_kernel_metadata("scratch")

    assert metadata["id"] == dispatch.KERNEL_SLUGS["scratch"]
    assert metadata["enable_gpu"] is True
    assert metadata["enable_internet"] is True
    assert metadata["code_file"] == "script.py"


def test_render_kernel_metadata_requires_source_file(monkeypatch, tmp_path: Path) -> None:
    """A missing source metadata file should surface as a normal file error."""
    project_root = tmp_path / "repo"
    (project_root / "kaggle" / "train_scratch").mkdir(parents=True)

    monkeypatch.setattr(dispatch, "PROJECT_ROOT", project_root)
    monkeypatch.setitem(dispatch.KERNEL_DIRS, "scratch", Path("kaggle/train_scratch"))

    with pytest.raises(FileNotFoundError):
        dispatch._render_kernel_metadata("scratch")


def test_should_bundle_path_skips_local_cache_artifacts() -> None:
    """The Kaggle bundle should not include compiled or build artefacts."""
    assert dispatch._should_bundle_path(Path("src/data/download.py")) is True
    assert dispatch._should_bundle_path(Path("src/__pycache__/download.cpython-313.pyc")) is False
    assert dispatch._should_bundle_path(Path("src/chexvision.egg-info/PKG-INFO")) is False
