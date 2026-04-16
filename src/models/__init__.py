"""Model architectures for CheXVision."""

from src.models.scratch_cnn import CheXVisionScratch
from src.models.densenet_transfer import CheXVisionDenseNet

__all__ = ["CheXVisionScratch", "CheXVisionDenseNet"]
