"""Model 1: Custom ResNet-style CNN built entirely from scratch.

Design decisions:
- Residual connections: Enable training of deeper networks by mitigating vanishing gradients.
- Batch normalization: Stabilizes training and allows higher learning rates.
- Global average pooling: Reduces parameters vs. fully-connected layers, less prone to overfitting.
- SE (Squeeze-and-Excitation) attention: Channel-wise recalibration — the model learns WHICH
  feature maps matter most for each pathology. Critical for multi-label medical imaging where
  different disease channels compete. Based on Hu et al. 2018 (CVPR best paper).
- Dual heads: Shared backbone extracts features once; separate heads specialize per task.
- Sigmoid activation: Multi-label (not mutually exclusive) requires independent probabilities per class.
- Depth [3,4,6,3]: ResNet-50 equivalent depth — 4x more capacity than the original ResNet-18
  style [2,2,2,2] baseline, justified by the complexity of 14 simultaneous pathology signals.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block — channel-wise attention.

    Mechanism:
      1. Squeeze: Global average pool → (B, C) descriptor of each channel's global response.
      2. Excitation: Two FC layers learn per-channel importance weights (gating with sigmoid).
      3. Scale: Multiply original feature map channels by learned weights.

    Why it helps for chest X-ray classification:
      Different pathologies activate different feature channels. SE teaches the network to
      amplify disease-relevant channels and suppress background texture channels — effectively
      a form of disease-specific feature selection.

    Reference: Hu et al., "Squeeze-and-Excitation Networks", CVPR 2018.
    """

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        bottleneck = max(channels // reduction, 4)
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, bottleneck, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        # Squeeze: global spatial information into channel descriptor
        scale = self.squeeze(x).view(b, c)
        # Excitation: learn channel importance weights
        scale = self.excitation(scale).view(b, c, 1, 1)
        # Scale: re-calibrate channel-wise feature responses
        return x * scale


class ResidualBlock(nn.Module):
    """Basic residual block with two 3×3 convolutions, a skip connection, and optional SE attention."""

    def __init__(
        self, in_channels: int, out_channels: int, stride: int = 1, use_se: bool = False
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # SE attention applied after second conv, before residual addition
        self.se = SEBlock(out_channels) if use_se else nn.Identity()

        # Skip connection with 1×1 conv if dimensions change
        self.shortcut: nn.Module = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)  # Channel attention before residual merge
        out += residual
        out = self.relu(out)
        return out


class CheXVisionScratch(nn.Module):
    """Custom ResNet-style CNN for chest X-ray classification.

    Architecture (default, ResNet-50 equivalent depth):
        - Stem: 7×7 conv (stride 2) + max pool → 64 channels, ¼ spatial resolution
        - Stage 1: 3× ResidualBlock [64 → 64, stride 1]
        - Stage 2: 4× ResidualBlock [64 → 128, stride 2]
        - Stage 3: 6× ResidualBlock [128 → 256, stride 2]
        - Stage 4: 3× ResidualBlock [256 → 512, stride 2]
        - Global average pooling → Dropout
        - Multilabel head: Linear(512 → 14) — 14 pathology classes (sigmoid)
        - Binary head:     Linear(512 → 1)  — Normal vs Abnormal (sigmoid)

    Parameter count (default config): ~23M parameters
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 14,
        block_config: tuple[int, ...] = (3, 4, 6, 3),
        filter_sizes: tuple[int, ...] = (64, 128, 256, 512),
        dropout: float = 0.5,
        use_se: bool = True,
    ) -> None:
        super().__init__()

        # Initial convolution: 7×7 conv captures low-level features (edges, textures)
        # stride=2 immediately reduces spatial resolution for efficiency
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, filter_sizes[0], kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(filter_sizes[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # Build residual stages — progressive downsampling at each stage
        self.stages = nn.ModuleList()
        current_channels = filter_sizes[0]
        for i, (num_blocks, out_channels) in enumerate(zip(block_config, filter_sizes)):
            stride = 1 if i == 0 else 2  # Downsample at each stage except the first
            blocks = [ResidualBlock(current_channels, out_channels, stride=stride, use_se=use_se)]
            for _ in range(1, num_blocks):
                blocks.append(ResidualBlock(out_channels, out_channels, stride=1, use_se=use_se))
            self.stages.append(nn.Sequential(*blocks))
            current_channels = out_channels

        # Global average pooling: reduces (B, C, H, W) → (B, C) — no spatial information lost
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout)

        # Multi-label classification head (14 pathologies — independent sigmoid per class)
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
                if m.bias is not None:
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
