"""
Normalization Utils - Utilidades de Normalización Avanzada
===========================================================

Utilidades para técnicas de normalización avanzadas.
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class LayerNorm(nn.Module):
    """
    Layer Normalization.
    """
    
    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        """
        Inicializar LayerNorm.
        
        Args:
            normalized_shape: Forma normalizada
            eps: Epsilon
        """
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


class GroupNorm(nn.Module):
    """
    Group Normalization.
    
    Paper: https://arxiv.org/abs/1803.08494
    """
    
    def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-5):
        """
        Inicializar GroupNorm.
        
        Args:
            num_groups: Número de grupos
            num_channels: Número de canales
            eps: Epsilon
        """
        super().__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [N, C, H, W]
            
        Returns:
            Tensor normalizado
        """
        N, C, H, W = x.size()
        
        # Reshape para grupos
        x = x.view(N, self.num_groups, C // self.num_groups, H, W)
        
        # Calcular estadísticas por grupo
        mean = x.mean(dim=[2, 3, 4], keepdim=True)
        var = x.var(dim=[2, 3, 4], keepdim=True, unbiased=False)
        
        # Normalizar
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        
        # Reshape de vuelta
        x_norm = x_norm.view(N, C, H, W)
        
        return self.weight.view(1, C, 1, 1) * x_norm + self.bias.view(1, C, 1, 1)


class InstanceNorm(nn.Module):
    """
    Instance Normalization.
    """
    
    def __init__(self, num_features: int, eps: float = 1e-5):
        """
        Inicializar InstanceNorm.
        
        Args:
            num_features: Número de features
            eps: Epsilon
        """
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [N, C, H, W]
            
        Returns:
            Tensor normalizado
        """
        N, C, H, W = x.size()
        
        # Calcular estadísticas por instancia y canal
        mean = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True, unbiased=False)
        
        # Normalizar
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        
        return self.weight.view(1, C, 1, 1) * x_norm + self.bias.view(1, C, 1, 1)


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    
    Paper: https://arxiv.org/abs/1910.07467
    """
    
    def __init__(self, dim: int, eps: float = 1e-8):
        """
        Inicializar RMSNorm.
        
        Args:
            dim: Dimensión
            eps: Epsilon
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor
            
        Returns:
            Tensor normalizado
        """
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class AdaptiveNorm(nn.Module):
    """
    Adaptive Normalization.
    """
    
    def __init__(
        self,
        num_features: int,
        norm_type: str = "batch",
        eps: float = 1e-5
    ):
        """
        Inicializar AdaptiveNorm.
        
        Args:
            num_features: Número de features
            norm_type: Tipo de normalización
            eps: Epsilon
        """
        super().__init__()
        self.num_features = num_features
        self.norm_type = norm_type
        self.eps = eps
        
        if norm_type == "batch":
            self.norm = nn.BatchNorm2d(num_features, eps=eps)
        elif norm_type == "instance":
            self.norm = InstanceNorm(num_features, eps)
        elif norm_type == "layer":
            self.norm = nn.LayerNorm(num_features, eps=eps)
        else:
            raise ValueError(f"Unknown norm type: {norm_type}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor
            
        Returns:
            Tensor normalizado
        """
        return self.norm(x)




