"""
Convolutional Layers

Implements various convolutional layer blocks.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class Conv1dBlock(nn.Module):
    """1D Convolutional block with normalization and activation."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: Optional[int] = None,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        norm: Optional[str] = None,
        activation: Optional[str] = None,
        dropout: float = 0.0
    ):
        super().__init__()
        
        if padding is None:
            padding = kernel_size // 2
        
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias
        )
        
        # Normalization
        if norm == "batch":
            self.norm = nn.BatchNorm1d(out_channels)
        elif norm == "layer":
            self.norm = nn.LayerNorm(out_channels)
        else:
            self.norm = None
        
        # Activation
        if activation:
            from .activation import create_activation
            self.activation = create_activation(activation)
        else:
            self.activation = None
        
        # Dropout
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        
        if self.norm:
            if isinstance(self.norm, nn.LayerNorm):
                # LayerNorm expects (batch, channels, seq_len)
                x = x.transpose(1, 2)
                x = self.norm(x)
                x = x.transpose(1, 2)
            else:
                x = self.norm(x)
        
        if self.activation:
            x = self.activation(x)
        
        if self.dropout:
            x = self.dropout(x)
        
        return x


class Conv2dBlock(nn.Module):
    """2D Convolutional block with normalization and activation."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: Optional[int] = None,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        norm: Optional[str] = None,
        activation: Optional[str] = None,
        dropout: float = 0.0
    ):
        super().__init__()
        
        if padding is None:
            padding = kernel_size // 2
        
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias
        )
        
        # Normalization
        if norm == "batch":
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm == "layer":
            self.norm = nn.LayerNorm(out_channels)
        else:
            self.norm = None
        
        # Activation
        if activation:
            from .activation import create_activation
            self.activation = create_activation(activation)
        else:
            self.activation = None
        
        # Dropout
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        
        if self.norm:
            if isinstance(self.norm, nn.LayerNorm):
                # LayerNorm expects (batch, channels, H, W)
                B, C, H, W = x.shape
                x = x.permute(0, 2, 3, 1)
                x = self.norm(x)
                x = x.permute(0, 3, 1, 2)
            else:
                x = self.norm(x)
        
        if self.activation:
            x = self.activation(x)
        
        if self.dropout:
            x = self.dropout(x)
        
        return x


class DepthwiseConv1d(nn.Module):
    """Depthwise separable 1D convolution."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: Optional[int] = None,
        bias: bool = False
    ):
        super().__init__()
        
        if padding is None:
            padding = kernel_size // 2
        
        self.depthwise = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=bias
        )
        
        self.pointwise = nn.Conv1d(
            in_channels,
            out_channels,
            1,
            bias=bias
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class SeparableConv1d(nn.Module):
    """Separable 1D convolution (depthwise + pointwise)."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: Optional[int] = None,
        bias: bool = False
    ):
        super().__init__()
        self.separable = DepthwiseConv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            bias
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.separable(x)



