"""
MiniFASNetV2 model architecture for face anti-spoofing.

This implements the MiniFASNet architecture with Fourier Transform
auxiliary loss for texture-based liveness detection.
"""

from typing import Tuple
import torch
from torch import nn
import torch.nn.functional as F


class MultiFTNet(nn.Module):
    """
    Main model with Fourier Transform auxiliary branch.
    
    Combines MiniFASNetV2SE with FT-based texture analysis.
    """

    def __init__(
        self,
        num_channels: int = 3,
        num_classes: int = 2,
        embedding_size: int = 128,
        conv6_kernel: Tuple[int, int] = (5, 5),
    ):
        """
        Initialize MultiFTNet model.

        Args:
            num_channels: Number of input channels (3 for RGB)
            num_classes: Number of output classes (2: real/spoof)
            embedding_size: Embedding dimension size
            conv6_kernel: Kernel size for conv6 layer
        """
        super(MultiFTNet, self).__init__()
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.model = MiniFASNetV2SE(
            embedding_size=embedding_size,
            conv6_kernel=conv6_kernel,
            num_classes=num_classes,
            num_channels=num_channels,
        )
        self.FTGenerator = FTGenerator(in_channels=128)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize model weights with Kaiming/Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            Training: (classifier_output, fourier_transform)
            Inference: classifier_output only
        """
        x = self.model.conv1(x)
        x = self.model.conv2_dw(x)
        x = self.model.conv_23(x)
        x = self.model.conv_3(x)
        x = self.model.conv_34(x)
        x = self.model.conv_4(x)
        x1 = self.model.conv_45(x)
        x1 = self.model.conv_5(x1)
        x1 = self.model.conv_6_sep(x1)
        x1 = self.model.conv_6_dw(x1)
        x1 = self.model.conv_6_flatten(x1)
        x1 = self.model.linear(x1)
        x1 = self.model.bn(x1)
        x1 = self.model.dropout(x1)
        classifier_output = self.model.logits(x1)

        if self.training:
            fourier_transform = self.FTGenerator(x)
            return classifier_output, fourier_transform
        else:
            return classifier_output


class FTGenerator(nn.Module):
    """Fourier Transform generator for texture analysis."""

    def __init__(self, in_channels: int = 128, out_channels: int = 1):
        """
        Initialize FT generator.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
        """
        super(FTGenerator, self).__init__()

        self.fourier_transform = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through FT generator."""
        return self.fourier_transform(x)


class Flatten(nn.Module):
    """Flatten tensor for fully connected layers."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Flatten all dimensions except batch."""
        return x.view(x.size(0), -1)


class Conv_block(nn.Module):
    """Convolution block with BN and PReLU activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel: Tuple[int, int] = (1, 1),
        stride: Tuple[int, int] = (1, 1),
        padding: Tuple[int, int] = (0, 0),
        groups: int = 1,
    ):
        super(Conv_block, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel,
            groups=groups,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.prelu = nn.PReLU(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.prelu(x)
        return x


class Linear_block(nn.Module):
    """Linear convolution block (no activation)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel: Tuple[int, int] = (1, 1),
        stride: Tuple[int, int] = (1, 1),
        padding: Tuple[int, int] = (0, 0),
        groups: int = 1,
    ):
        super(Linear_block, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels=out_channels,
            kernel_size=kernel,
            groups=groups,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        return x


class SEModule(nn.Module):
    """Squeeze-and-Excitation module for channel attention."""

    def __init__(self, channels: int, reduction: int):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(
            channels, channels // reduction, kernel_size=1, padding=0, bias=False
        )
        self.bn1 = nn.BatchNorm2d(channels // reduction)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(
            channels // reduction, channels, kernel_size=1, padding=0, bias=False
        )
        self.bn2 = nn.BatchNorm2d(channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.sigmoid(x)
        return module_input * x


class MiniFASNetV2SE(nn.Module):
    """MiniFASNet V2 with Squeeze-and-Excitation modules."""

    def __init__(
        self,
        embedding_size: int = 128,
        conv6_kernel: Tuple[int, int] = (5, 5),
        num_classes: int = 2,
        num_channels: int = 3,
        dropout_prob: float = 0.4,
    ):
        super(MiniFASNetV2SE, self).__init__()
        
        self.conv1 = Conv_block(num_channels, 32, kernel=(3, 3), stride=(2, 2), padding=(1, 1))
        self.conv2_dw = Conv_block(32, 32, kernel=(3, 3), stride=(1, 1), padding=(1, 1), groups=32)
        
        # Depth-wise separable convolutions
        self.conv_23 = Conv_block(32, 64, kernel=(3, 3), stride=(2, 2), padding=(1, 1))
        self.conv_3 = Conv_block(64, 64, kernel=(3, 3), stride=(1, 1), padding=(1, 1), groups=64)
        self.conv_34 = Conv_block(64, 128, kernel=(3, 3), stride=(2, 2), padding=(1, 1))
        self.conv_4 = Conv_block(128, 128, kernel=(3, 3), stride=(1, 1), padding=(1, 1), groups=128)
        self.conv_45 = Conv_block(128, 128, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_5 = Conv_block(128, 128, kernel=(3, 3), stride=(1, 1), padding=(1, 1), groups=128)
        
        # SE module
        self.se = SEModule(128, reduction=8)
        
        # Final layers
        self.conv_6_sep = Conv_block(128, embedding_size, kernel=(1, 1), stride=(1, 1), padding=(0, 0))
        self.conv_6_dw = Linear_block(
            embedding_size, embedding_size, 
            kernel=conv6_kernel, stride=(1, 1), padding=(0, 0), groups=embedding_size
        )
        self.conv_6_flatten = Flatten()
        self.linear = nn.Linear(embedding_size, embedding_size)
        self.bn = nn.BatchNorm1d(embedding_size)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.logits = nn.Linear(embedding_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Use MultiFTNet for forward pass")


def get_kernel(height: int, width: int) -> Tuple[int, int]:
    """
    Calculate kernel size based on input dimensions.

    Args:
        height: Input height
        width: Input width

    Returns:
        Kernel size tuple
    """
    kernel_size = ((height + 15) // 16, (width + 15) // 16)
    return kernel_size


def create_model(
    input_size: int = 128,
    num_classes: int = 2,
    num_channels: int = 3,
) -> MultiFTNet:
    """
    Create MiniFAS model with appropriate kernel size.

    Args:
        input_size: Input image size (square)
        num_classes: Number of output classes
        num_channels: Number of input channels

    Returns:
        Initialized MultiFTNet model
    """
    kernel_size = get_kernel(input_size, input_size)
    model = MultiFTNet(
        num_channels=num_channels,
        num_classes=num_classes,
        embedding_size=128,
        conv6_kernel=kernel_size,
    )
    return model
