"""Unit tests for data augmentation pipelines."""

from __future__ import annotations

import torch
from PIL import Image

from src.data.transforms import get_eval_transforms, get_train_transforms


class TestTrainTransforms:
    def test_output_shape_default(self) -> None:
        transform = get_train_transforms(224)
        img = Image.new("RGB", (512, 512), color=(128, 128, 128))
        tensor = transform(img)
        assert tensor.shape == (3, 224, 224)
        assert isinstance(tensor, torch.Tensor)

    def test_output_shape_custom_size(self) -> None:
        transform = get_train_transforms(128)
        img = Image.new("RGB", (300, 400))
        tensor = transform(img)
        assert tensor.shape == (3, 128, 128)

    def test_normalized_range(self) -> None:
        transform = get_train_transforms(224)
        img = Image.new("RGB", (224, 224), color=(255, 255, 255))
        tensor = transform(img)
        # After ImageNet normalization, white pixels should be > 1.0
        assert tensor.max().item() > 1.0

    def test_deterministic_with_seed(self) -> None:
        transform = get_train_transforms(224)
        img = Image.new("RGB", (224, 224), color=(100, 100, 100))
        torch.manual_seed(42)
        t1 = transform(img)
        torch.manual_seed(42)
        t2 = transform(img)
        torch.testing.assert_close(t1, t2)


class TestEvalTransforms:
    def test_output_shape(self) -> None:
        transform = get_eval_transforms(224)
        img = Image.new("RGB", (512, 512), color=(128, 128, 128))
        tensor = transform(img)
        assert tensor.shape == (3, 224, 224)

    def test_deterministic(self) -> None:
        """Eval transforms have no randomness — same input, same output."""
        transform = get_eval_transforms(224)
        img = Image.new("RGB", (300, 300), color=(50, 100, 150))
        t1 = transform(img)
        t2 = transform(img)
        torch.testing.assert_close(t1, t2)

    def test_grayscale_input_converted(self) -> None:
        """Even if input is grayscale, .convert('RGB') happens before transform in the dataset."""
        transform = get_eval_transforms(224)
        img = Image.new("L", (224, 224), color=128).convert("RGB")
        tensor = transform(img)
        assert tensor.shape == (3, 224, 224)
