"""Data augmentation and preprocessing pipelines.

Design rationale for each augmentation:
- RandomHorizontalFlip: chest X-rays have bilateral symmetry; flipping is anatomically valid.
- RandomRotation: slight patient positioning variation in real radiographs.
- RandomAffine (translate + shear): simulates positioning shifts and beam angle variation.
- ColorJitter (brightness/contrast): compensates for varying X-ray exposure settings.
- GaussianBlur: simulates varying sharpness due to patient motion or detector resolution.
- RandomErasing (CutOut): forces model to rely on distributed features, not single bright regions;
  also simulates radio-opaque artifacts (leads, clips, implants).
- ImageNet normalization: even for grayscale medical images, ImageNet stats are standard when
  using ImageNet-pretrained backbones (DenseNet-121). Both models use 3-channel RGB.
"""

from __future__ import annotations

from torchvision import transforms


def get_train_transforms(image_size: int = 224) -> transforms.Compose:
    """Training transforms with medically-motivated data augmentation."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        # Slight translation (5%) and shear (5°) — patient positioning variation
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), shear=5),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.05),
        # Simulate varying focus/motion blur in radiography equipment
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet statistics
            std=[0.229, 0.224, 0.225],
        ),
        # CutOut: simulate radio-opaque objects; forces distributed feature learning
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.08), ratio=(0.5, 2.0)),
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
