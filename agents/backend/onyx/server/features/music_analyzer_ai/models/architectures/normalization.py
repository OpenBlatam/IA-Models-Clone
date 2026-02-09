"""
Modular Normalization Layers
Various normalization techniques for deep learning models
"""

import logging

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available")


class LayerNorm(nn.Module):
    """
    Layer Normalization with optional element-wise affine transformation
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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply layer normalization"""
        return nn.functional.layer_norm(
            x, self.normalized_shape, self.weight, self.bias, self.eps
        )


class BatchNorm1d(nn.Module):
    """
    Batch Normalization for 1D inputs with proper initialization
    """
    
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True
    ):
        super().__init__()
        self.bn = nn.BatchNorm1d(
            num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats
        )
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Initialize batch norm parameters"""
        if self.bn.affine:
            nn.init.ones_(self.bn.weight)
            nn.init.zeros_(self.bn.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply batch normalization"""
        return self.bn(x)


class AdaptiveNormalization(nn.Module):
    """
    Adaptive normalization that switches between batch and layer norm
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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply adaptive normalization"""
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



