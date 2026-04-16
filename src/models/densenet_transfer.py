"""Model 2: DenseNet-121 with transfer learning (CheXNet approach).

Design decisions:
- DenseNet-121: Dense connections promote feature reuse and reduce parameters vs. ResNet.
  Chosen because CheXNet (Rajpurkar et al., 2017) demonstrated radiologist-level performance with this backbone.
- ImageNet pretraining: Provides strong low-level feature initialization even for medical images.
  Textures, edges, and structural patterns transfer well across domains.
- Two-phase fine-tuning:
  Phase 1 — Freeze backbone, train only classifier head (prevents catastrophic forgetting).
  Phase 2 — Unfreeze all layers with reduced LR for end-to-end refinement.
- Dual heads: Same rationale as scratch model — shared features, specialized predictions.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


class CheXVisionDenseNet(nn.Module):
    """DenseNet-121 fine-tuned for chest X-ray pathology detection.

    Architecture:
        - DenseNet-121 backbone (pretrained on ImageNet)
        - Replace final classifier with dual heads:
            - Multi-label: 14 pathology classes (sigmoid)
            - Binary: Normal vs Abnormal (sigmoid)
    """

    def __init__(
        self,
        num_classes: int = 14,
        pretrained: bool = True,
        dropout: float = 0.3,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()

        # Load pretrained DenseNet-121
        weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.densenet121(weights=weights)

        # Get the number of features from the classifier
        num_features = self.backbone.classifier.in_features

        # Remove the original classifier
        self.backbone.classifier = nn.Identity()

        # Shared feature layer
        self.feature_layer = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )

        # Multi-label classification head (14 pathologies)
        self.multilabel_head = nn.Linear(512, num_classes)

        # Binary classification head (Normal vs Abnormal)
        self.binary_head = nn.Linear(512, 1)

        # Optionally freeze backbone (Phase 1 of fine-tuning)
        if freeze_backbone:
            self.freeze_backbone()

        # Initialize new layers
        self._init_new_layers()

    def _init_new_layers(self) -> None:
        for module in [self.feature_layer, self.multilabel_head, self.binary_head]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0)

    def freeze_backbone(self) -> None:
        """Freeze all backbone parameters (Phase 1: train only classifier heads)."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreeze backbone parameters (Phase 2: end-to-end fine-tuning)."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass returning both task predictions.

        Args:
            x: Input tensor of shape (B, 3, 224, 224).

        Returns:
            Dict with 'multilabel_logits' (B, 14) and 'binary_logits' (B, 1).
        """
        features = self.backbone(x)
        features = self.feature_layer(features)

        return {
            "multilabel_logits": self.multilabel_head(features),
            "binary_logits": self.binary_head(features),
        }
