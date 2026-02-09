"""
Normalization Layers
====================

Capas de normalización avanzadas.
"""

import logging

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None

logger = logging.getLogger(__name__)


class LayerNorm(nn.Module):
    """
    Layer Normalization.
    
    Normalización por capa con parámetros aprendibles.
    """
    
    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        """
        Inicializar LayerNorm.
        
        Args:
            normalized_shape: Forma a normalizar
            eps: Épsilon para estabilidad numérica
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required")
        
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor
            
        Returns:
            Tensor normalizado
        """
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        
        return self.weight * x_norm + self.bias


class BatchNorm1d(nn.Module):
    """
    Batch Normalization 1D.
    
    Wrapper para nn.BatchNorm1d con inicialización mejorada.
    """
    
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        """Inicializar BatchNorm1d."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required")
        
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features, eps=eps, momentum=momentum)
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Inicializar pesos."""
        nn.init.ones_(self.bn.weight)
        nn.init.zeros_(self.bn.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.bn(x)


class GroupNorm(nn.Module):
    """
    Group Normalization.
    
    Normalización por grupos.
    """
    
    def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-5):
        """Inicializar GroupNorm."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required")
        
        super().__init__()
        self.gn = nn.GroupNorm(num_groups, num_channels, eps=eps)
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Inicializar pesos."""
        nn.init.ones_(self.gn.weight)
        nn.init.zeros_(self.gn.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.gn(x)

