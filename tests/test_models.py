"""Unit tests for CheXVision model architectures."""

import pytest
import torch

from src.models.densenet_transfer import CheXVisionDenseNet
from src.models.scratch_cnn import CheXVisionScratch


@pytest.fixture
def dummy_input() -> torch.Tensor:
    """Batch of 2 fake 224x224 RGB images."""
    return torch.randn(2, 3, 224, 224)


class TestCheXVisionScratch:
    def test_output_shapes(self, dummy_input: torch.Tensor) -> None:
        model = CheXVisionScratch(in_channels=3, num_classes=14)
        outputs = model(dummy_input)

        assert "multilabel_logits" in outputs
        assert "binary_logits" in outputs
        assert outputs["multilabel_logits"].shape == (2, 14)
        assert outputs["binary_logits"].shape == (2, 1)

    def test_custom_block_config(self, dummy_input: torch.Tensor) -> None:
        model = CheXVisionScratch(
            in_channels=3,
            num_classes=14,
            block_config=(1, 1, 1, 1),
            filter_sizes=(32, 64, 128, 256),
        )
        outputs = model(dummy_input)
        assert outputs["multilabel_logits"].shape == (2, 14)

    def test_gradient_flow(self, dummy_input: torch.Tensor) -> None:
        model = CheXVisionScratch()
        outputs = model(dummy_input)
        loss = outputs["multilabel_logits"].sum() + outputs["binary_logits"].sum()
        loss.backward()

        # Verify gradients exist for all parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"


class TestCheXVisionDenseNet:
    def test_output_shapes(self, dummy_input: torch.Tensor) -> None:
        model = CheXVisionDenseNet(num_classes=14, pretrained=False)
        outputs = model(dummy_input)

        assert outputs["multilabel_logits"].shape == (2, 14)
        assert outputs["binary_logits"].shape == (2, 1)

    def test_freeze_unfreeze(self) -> None:
        model = CheXVisionDenseNet(pretrained=False, freeze_backbone=True)

        # Check backbone is frozen
        for param in model.backbone.parameters():
            assert not param.requires_grad

        # Check heads are trainable
        for param in model.multilabel_head.parameters():
            assert param.requires_grad

        # Unfreeze
        model.unfreeze_backbone()
        for param in model.backbone.parameters():
            assert param.requires_grad

    def test_gradient_flow(self, dummy_input: torch.Tensor) -> None:
        model = CheXVisionDenseNet(pretrained=False)
        outputs = model(dummy_input)
        loss = outputs["multilabel_logits"].sum() + outputs["binary_logits"].sum()
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
