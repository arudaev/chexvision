"""Data augmentation and preprocessing pipelines."""

from __future__ import annotations

from torchvision import transforms


def get_train_transforms(image_size: int = 224) -> transforms.Compose:
    """Training transforms with data augmentation."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet statistics
            std=[0.229, 0.224, 0.225],
        ),
    ])


def get_eval_transforms(image_size: int = 224) -> transforms.Compose:
    """Evaluation/test transforms (no augmentation)."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
