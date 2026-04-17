"""Regression tests for the Streamlit app entrypoint."""

from __future__ import annotations

import runpy
from pathlib import Path


def test_app_entrypoint_loads_when_run_by_path() -> None:
    """Loading app/app.py by path should succeed outside the repo root."""
    globals_dict = runpy.run_path(str(Path("app") / "app.py"), run_name="chexvision_app_test")
    assert "main" in globals_dict
