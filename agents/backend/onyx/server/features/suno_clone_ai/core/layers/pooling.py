"""
Pooling Layers

Implements various pooling operations.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class AdaptivePooling(nn.Module):
    """Adaptive pooling (1D or 2D)."""
    
    def __init__(
        self,
        output_size: int = 1,
        pool_type: str = "avg",
        dim: int = 1
    ):
        super().__init__()
        self.output_size = output_size
        self.pool_type = pool_type
        self.dim = dim
        
        if dim == 1:
            if pool_type == "avg":
                self.pool = nn.AdaptiveAvgPool1d(output_size)
            elif pool_type == "max":
                self.pool = nn.AdaptiveMaxPool1d(output_size)
            else:
                raise ValueError(f"Unknown pool type: {pool_type}")
        elif dim == 2:
            if pool_type == "avg":
                self.pool = nn.AdaptiveAvgPool2d(output_size)
            elif pool_type == "max":
                self.pool = nn.AdaptiveMaxPool2d(output_size)
            else:
                raise ValueError(f"Unknown pool type: {pool_type}")
        else:
            raise ValueError(f"Unsupported dim: {dim}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(x)


class GlobalPooling(nn.Module):
    """Global pooling (average or max)."""
    
    def __init__(
        self,
        pool_type: str = "avg",
        dim: int = -1,
        keepdim: bool = False
    ):
        super().__init__()
        self.pool_type = pool_type
        self.dim = dim
        self.keepdim = keepdim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pool_type == "avg":
            return x.mean(dim=self.dim, keepdim=self.keepdim)
        elif self.pool_type == "max":
            return x.max(dim=self.dim, keepdim=self.keepdim)[0]
        else:
            raise ValueError(f"Unknown pool type: {self.pool_type}")


def create_pooling(
    pool_type: str = "avg",
    output_size: Optional[int] = None,
    dim: int = 1
) -> nn.Module:
    """
    Create pooling layer.
    
    Args:
        pool_type: Type of pooling ('avg', 'max')
        output_size: Output size for adaptive pooling
        dim: Dimension (1 or 2)
        
    Returns:
        Pooling module
    """
    if output_size is not None:
        return AdaptivePooling(output_size, pool_type, dim)
    else:
        return GlobalPooling(pool_type, dim=-1)



