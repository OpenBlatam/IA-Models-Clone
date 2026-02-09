"""
Normalization Layers

Implements various normalization techniques for deep learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class LayerNorm(nn.Module):
    """Layer Normalization."""
    
    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-5,
        elementwise_affine: bool = True
    ):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
            self.bias = nn.Parameter(torch.zeros(normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x,
            self.normalized_shape,
            self.weight,
            self.bias,
            self.eps
        )


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    
    def __init__(
        self,
        dim: int,
        eps: float = 1e-8
    ):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.norm(dim=-1, keepdim=True) * (x.shape[-1] ** -0.5)
        return self.weight * (x / (norm + self.eps))


class GroupNorm(nn.Module):
    """Group Normalization."""
    
    def __init__(
        self,
        num_groups: int,
        num_channels: int,
        eps: float = 1e-5,
        affine: bool = True
    ):
        super().__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        
        if affine:
            self.weight = nn.Parameter(torch.ones(num_channels))
            self.bias = nn.Parameter(torch.zeros(num_channels))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.group_norm(
            x,
            self.num_groups,
            self.weight,
            self.bias,
            self.eps
        )


class InstanceNorm(nn.Module):
    """Instance Normalization."""
    
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        affine: bool = True
    ):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        
        if affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.instance_norm(
            x,
            self.weight,
            self.bias,
            running_mean=None,
            running_var=None,
            use_input_stats=True,
            momentum=0.0,
            eps=self.eps
        )



