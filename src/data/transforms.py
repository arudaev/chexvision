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
- CLAHE (optional): Contrast Limited Adaptive Histogram Equalisation enhances local contrast,
  making low-contrast findings (nodules, infiltrations) more visible before the network sees them.
  Applied in LAB colour space so brightness is enhanced without shifting colour balance.
"""

from __future__ import annotations

import numpy as np
from PIL import Image
from torchvision import transforms


class CLAHETransform:
    """Apply CLAHE to a PIL image to enhance local contrast.

    Standard preprocessing in radiology AI — boosts visibility of small, low-contrast
    findings (Nodule, Infiltration, Pneumonia) that are otherwise hard to learn from.
    Applied in LAB colour space on the L (lightness) channel only.
    """

    def __init__(self, clip_limit: float = 2.0, tile_grid_size: tuple[int, int] = (8, 8)) -> None:
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, img: Image.Image) -> Image.Image:
        import cv2  # lazy import — only required when CLAHE is enabled
        img_np = np.array(img.convert("RGB"))
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return Image.fromarray(result)


def get_train_transforms(image_size: int = 320, use_clahe: bool = True) -> transforms.Compose:
    """Training transforms with medically-motivated data augmentation.

    Args:
        image_size: Target spatial resolution (both sides).
        use_clahe: Prepend CLAHE contrast enhancement. Recommended for chest X-rays.
    """
    steps: list = []
    if use_clahe:
        steps.append(CLAHETransform())
    steps += [
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
    ]
    return transforms.Compose(steps)


def get_eval_transforms(image_size: int = 320, use_clahe: bool = True) -> transforms.Compose:
    """Evaluation/test transforms (no augmentation, optional CLAHE).

    Args:
        image_size: Target spatial resolution (both sides).
        use_clahe: Prepend CLAHE contrast enhancement. Should match training setting.
    """
    steps: list = []
    if use_clahe:
        steps.append(CLAHETransform())
    steps += [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
    return transforms.Compose(steps)
