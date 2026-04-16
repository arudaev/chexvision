"""Model 1: Custom ResNet-style CNN built entirely from scratch.

Design decisions:
- Residual connections: Enable training of deeper networks by mitigating vanishing gradients.
- Batch normalization: Stabilizes training and allows higher learning rates.
- Global average pooling: Reduces parameters vs. fully-connected layers, less prone to overfitting.
- Dual heads: Shared backbone extracts features once; separate heads specialize per task.
- Sigmoid activation: Multi-label (not mutually exclusive) requires independent probabilities per class.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """Basic residual block with two 3x3 convolutions and a skip connection."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Skip connection with 1x1 conv if dimensions change
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out


class CheXVisionScratch(nn.Module):
    """Custom ResNet-style CNN for chest X-ray classification.

    Architecture:
        - Initial 7x7 conv + max pool
        - 4 stages of residual blocks: [64, 128, 256, 512] filters
        - Global average pooling
        - Two classification heads:
            - Multi-label: 14 pathology classes (sigmoid)
            - Binary: Normal vs Abnormal (sigmoid)
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 14,
        block_config: tuple[int, ...] = (2, 2, 2, 2),
        filter_sizes: tuple[int, ...] = (64, 128, 256, 512),
        dropout: float = 0.5,
    ) -> None:
        super().__init__()

        # Initial convolution: 7x7 conv captures low-level features (edges, textures)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, filter_sizes[0], kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(filter_sizes[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # Build residual stages
        self.stages = nn.ModuleList()
        current_channels = filter_sizes[0]
        for i, (num_blocks, out_channels) in enumerate(zip(block_config, filter_sizes)):
            stride = 1 if i == 0 else 2  # Downsample at each stage except the first
            blocks = [ResidualBlock(current_channels, out_channels, stride=stride)]
            for _ in range(1, num_blocks):
                blocks.append(ResidualBlock(out_channels, out_channels, stride=1))
            self.stages.append(nn.Sequential(*blocks))
            current_channels = out_channels

        # Global average pooling: reduces (B, C, H, W) → (B, C)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout)

        # Multi-label classification head (14 pathologies)
        self.multilabel_head = nn.Linear(filter_sizes[-1], num_classes)

        # Binary classification head (Normal vs Abnormal)
        self.binary_head = nn.Linear(filter_sizes[-1], 1)

        # Initialize weights (Kaiming for ReLU networks)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass returning both task predictions.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Dict with 'multilabel_logits' (B, 14) and 'binary_logits' (B, 1).
        """
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)

        return {
            "multilabel_logits": self.multilabel_head(x),
            "binary_logits": self.binary_head(x),
        }
