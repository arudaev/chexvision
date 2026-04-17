"""Model architectures for CheXVision."""

from src.models.densenet_transfer import CheXVisionDenseNet
from src.models.scratch_cnn import CheXVisionScratch

__all__ = ["CheXVisionScratch", "CheXVisionDenseNet"]
