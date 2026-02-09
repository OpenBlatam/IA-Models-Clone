"""
Adaptive Normalization Module

Implements adaptive normalization that switches between different normalization types.
"""

import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class AdaptiveNormalization(nn.Module):
    """
    Adaptive normalization that switches between batch and layer norm.
    
    Args:
        normalized_shape: Shape for layer normalization (int or tuple).
        norm_type: Type of normalization ("layer", "batch", "instance").
        eps: Small value to prevent division by zero.
    """
    
    def __init__(
        self,
        normalized_shape: int,
        norm_type: str = "layer",  # "layer", "batch", "instance"
        eps: float = 1e-5
    ):
        super().__init__()
        self.norm_type = norm_type
        
        if norm_type == "layer":
            self.norm = nn.LayerNorm(normalized_shape, eps=eps)
        elif norm_type == "batch":
            self.norm = nn.BatchNorm1d(normalized_shape, eps=eps)
        elif norm_type == "instance":
            self.norm = nn.InstanceNorm1d(normalized_shape, eps=eps)
        else:
            raise ValueError(f"Unknown norm_type: {norm_type}")
        
        logger.debug(f"Initialized AdaptiveNormalization with norm_type='{norm_type}'")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply adaptive normalization.
        
        Args:
            x: Input tensor
        
        Returns:
            Normalized tensor
        """
        # Handle different input shapes
        if self.norm_type == "layer":
            return self.norm(x)
        elif self.norm_type in ["batch", "instance"]:
            # Ensure 3D input for batch/instance norm
            if x.dim() == 2:
                x = x.unsqueeze(-1)
            x = self.norm(x)
            if x.dim() == 3 and x.size(-1) == 1:
                x = x.squeeze(-1)
            return x
        return x



