"""
Layer Normalization Module

Implements layer normalization with optional element-wise affine transformation.
"""

import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class LayerNorm(nn.Module):
    """
    Layer Normalization with optional element-wise affine transformation.
    
    Args:
        normalized_shape: Shape of the input to normalize (int or tuple).
        eps: Small value to prevent division by zero.
        elementwise_affine: If True, apply learnable affine transformation.
    """
    
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
        
        logger.debug(f"Initialized LayerNorm with normalized_shape={normalized_shape}, eps={eps}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply layer normalization.
        
        Args:
            x: Input tensor of shape [..., normalized_shape]
        
        Returns:
            Normalized tensor of same shape
        """
        return nn.functional.layer_norm(
            x, self.normalized_shape, self.weight, self.bias, self.eps
        )



