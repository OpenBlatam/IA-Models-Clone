"""
MobileNet Building Blocks
Modular, reusable components for MobileNet architectures
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable


class ConvBNReLU(nn.Sequential):
    """
    Convolution-BatchNorm-ReLU block
    Reusable building block for MobileNet architectures
    """
    
    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        activation: Optional[Callable] = None,
    ):
        """
        Initialize ConvBNReLU block
        
        Args:
            in_planes: Input channels
            out_planes: Output channels
            kernel_size: Convolution kernel size
            stride: Convolution stride
            groups: Number of groups for depthwise convolution
            norm_layer: Normalization layer factory
            activation: Activation function
        """
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        
        if activation is None:
            activation = nn.ReLU6(inplace=True)
        
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size,
                stride,
                padding,
                groups=groups,
                bias=False
            ),
            norm_layer(out_planes),
            activation
        )


class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise Separable Convolution
    Efficient convolution block used in MobileNet
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ):
        """
        Initialize depthwise separable convolution
        
        Args:
            in_channels: Input channels
            out_channels: Output channels
            kernel_size: Kernel size
            stride: Stride
            norm_layer: Normalization layer factory
        """
        super(DepthwiseSeparableConv, self).__init__()
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        
        # Depthwise convolution
        padding = (kernel_size - 1) // 2
        self.depthwise = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size,
                stride,
                padding,
                groups=in_channels,
                bias=False
            ),
            norm_layer(in_channels),
            nn.ReLU6(inplace=True)
        )
        
        # Pointwise convolution
        self.pointwise = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
            norm_layer(out_channels),
            nn.ReLU6(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class InvertedResidual(nn.Module):
    """
    Inverted Residual Block
    Core building block of MobileNetV2/V3
    """
    
    def __init__(
        self,
        inp: int,
        oup: int,
        stride: int,
        expand_ratio: int,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        activation: Optional[Callable] = None,
    ):
        """
        Initialize inverted residual block
        
        Args:
            inp: Input channels
            oup: Output channels
            stride: Stride
            expand_ratio: Expansion ratio
            norm_layer: Normalization layer factory
            activation: Activation function
        """
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        
        if activation is None:
            activation = nn.ReLU6(inplace=True)
        
        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup
        
        layers = []
        if expand_ratio != 1:
            # Pointwise expansion
            layers.append(ConvBNReLU(
                inp, hidden_dim, kernel_size=1, norm_layer=norm_layer, activation=activation
            ))
        
        # Depthwise convolution
        layers.append(ConvBNReLU(
            hidden_dim,
            hidden_dim,
            stride=stride,
            groups=hidden_dim,
            norm_layer=norm_layer,
            activation=activation
        ))
        
        # Pointwise linear (no activation)
        layers.append(
            nn.Sequential(
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                norm_layer(oup),
            )
        )
        
        self.conv = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block
    Used in MobileNetV3 for channel attention
    """
    
    def __init__(
        self,
        channels: int,
        reduction: int = 4,
    ):
        """
        Initialize SE block
        
        Args:
            channels: Number of input channels
            reduction: Reduction ratio
        """
        super(SEBlock, self).__init__()
        reduced_channels = max(1, channels // reduction)
        
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, reduced_channels, 1),
            nn.ReLU6(inplace=True),
            nn.Conv2d(reduced_channels, channels, 1),
            nn.Hardsigmoid(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.se(x)


class HardSwish(nn.Module):
    """
    Hard Swish Activation
    Used in MobileNetV3
    """
    
    def __init__(self, inplace: bool = True):
        super(HardSwish, self).__init__()
        self.inplace = inplace
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * F.relu6(x + 3, inplace=self.inplace) / 6


class HardSigmoid(nn.Module):
    """
    Hard Sigmoid Activation
    Used in MobileNetV3 SE blocks
    """
    
    def __init__(self, inplace: bool = True):
        super(HardSigmoid, self).__init__()
        self.inplace = inplace
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu6(x + 3, inplace=self.inplace) / 6



