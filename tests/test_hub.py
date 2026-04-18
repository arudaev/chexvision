"""Tests for Hugging Face Hub helpers."""

from __future__ import annotations

import os

from src.utils.hub import (
    HF_DATASET_REPO,
    HF_DATASET_REVISION,
    configure_hf_runtime,
    load_hf_token,
    render_model_card,
)


def test_load_hf_token_prefers_environment(monkeypatch) -> None:
    """The shared token loader should resolve the standard HF env var."""
    monkeypatch.setenv("HF_TOKEN", "hf_test_token")
    assert load_hf_token() == "hf_test_token"


def test_configure_hf_runtime_sets_expected_env(monkeypatch) -> None:
    """Configuring the HF runtime should disable Xet and normalize the token env."""
    monkeypatch.setenv("KAGGLE_KERNEL_RUN_TYPE", "Interactive")
    monkeypatch.setenv("HF_TOKEN", "hf_test_token")
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("HF_HUB_DISABLE_XET", raising=False)
    monkeypatch.delenv("HF_HUB_ETAG_TIMEOUT", raising=False)
    monkeypatch.delenv("HF_HUB_DOWNLOAD_TIMEOUT", raising=False)
    monkeypatch.delenv("HF_HUB_VERBOSITY", raising=False)

    token = configure_hf_runtime()

    assert token == "hf_test_token"
    assert "kaggle/working/hf_home" in os.environ["HF_HOME"].replace("\\", "/")
    assert os.environ["HF_HUB_DISABLE_XET"] == "1"
    assert os.environ["HF_HUB_ETAG_TIMEOUT"] == "30"
    assert os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] == "300"
    assert os.environ["HUGGING_FACE_HUB_TOKEN"] == "hf_test_token"


def test_load_hf_token_required_mentions_kaggle_secrets(monkeypatch) -> None:
    """Kaggle runs should fail with a Kaggle-specific missing-token message."""
    monkeypatch.setenv("KAGGLE_KERNEL_RUN_TYPE", "Interactive")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACEHUB_API_TOKEN", raising=False)
    monkeypatch.setattr("src.utils.hub._load_dotenv_if_available", lambda: None)

    try:
        load_hf_token(required=True)
    except RuntimeError as exc:
        assert "Kaggle Secrets" in str(exc)
    else:
        raise AssertionError("Expected load_hf_token(required=True) to raise on Kaggle.")


def test_render_model_card_includes_dataset_and_metrics() -> None:
    """Rendered model cards should include the project dataset and metrics."""
    checkpoint = {
        "epoch": 12,
        "best_auc": 0.9132,
        "config": {
            "data": {
                "hf_dataset_repo": HF_DATASET_REPO,
                "hf_dataset_revision": HF_DATASET_REVISION,
            },
            "model": {"type": "densenet", "name": "CheXVision-DenseNet"},
            "training": {"epochs": 30, "batch_size": 32},
        },
    }
    history = {
        "auc_roc_macro": [0.8, 0.9132],
        "binary_auc_roc": [0.75, 0.881],
        "binary_f1": [0.4, 0.61],
    }

    card = render_model_card("arudaev/chexvision-densenet", checkpoint, history)

    assert HF_DATASET_REPO in card
    assert HF_DATASET_REVISION in card
    assert "CheXVision-DenseNet" in card
    assert "Best validation macro AUC-ROC: `0.9132`" in card
    assert "Best validation binary AUC-ROC: `0.8810`" in card
    assert "Best validation binary F1: `0.6100`" in card
