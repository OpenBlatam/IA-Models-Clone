"""
Modular Residual Connections
Various residual connection strategies
"""

from typing import Optional, Callable
import logging

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available")


class ResidualConnection(nn.Module):
    """Standard residual connection"""
    
    def __init__(self, dropout: Optional[nn.Module] = None):
        super().__init__()
        self.dropout = dropout
    
    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        """Add residual connection"""
        if self.dropout:
            x = self.dropout(x)
        return x + residual


class PreNormResidual(nn.Module):
    """Pre-norm residual connection"""
    
    def __init__(
        self,
        norm: nn.Module,
        fn: nn.Module,
        dropout: Optional[nn.Module] = None
    ):
        super().__init__()
        self.norm = norm
        self.fn = fn
        self.dropout = dropout
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pre-norm: normalize before function"""
        residual = x
        x = self.norm(x)
        x = self.fn(x)
        if self.dropout:
            x = self.dropout(x)
        return x + residual


class PostNormResidual(nn.Module):
    """Post-norm residual connection"""
    
    def __init__(
        self,
        norm: nn.Module,
        fn: nn.Module,
        dropout: Optional[nn.Module] = None
    ):
        super().__init__()
        self.norm = norm
        self.fn = fn
        self.dropout = dropout
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Post-norm: normalize after function"""
        residual = x
        x = self.fn(x)
        if self.dropout:
            x = self.dropout(x)
        x = x + residual
        x = self.norm(x)
        return x


class GatedResidual(nn.Module):
    """Gated residual connection"""
    
    def __init__(self, gate_fn: Optional[Callable] = None):
        super().__init__()
        self.gate_fn = gate_fn or torch.sigmoid
    
    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        """Gated residual connection"""
        gate = self.gate_fn(x)
        return gate * x + (1 - gate) * residual



