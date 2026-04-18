"""Regression tests for the Streamlit app entrypoint."""

from __future__ import annotations

import runpy
from pathlib import Path


def test_app_entrypoint_loads_when_run_by_path() -> None:
    """Loading app/app.py by path should succeed outside the repo root."""
    globals_dict = runpy.run_path(str(Path("app") / "app.py"), run_name="chexvision_app_test")
    assert "main" in globals_dict


def test_space_app_tree_stays_text_only() -> None:
    """The Space deploy currently rejects tracked binary image assets."""
    forbidden_suffixes = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
    app_root = Path("app")

    offending = sorted(
        path.relative_to(app_root).as_posix()
        for path in app_root.rglob("*")
        if path.is_file() and path.suffix.lower() in forbidden_suffixes
    )

    assert offending == []
