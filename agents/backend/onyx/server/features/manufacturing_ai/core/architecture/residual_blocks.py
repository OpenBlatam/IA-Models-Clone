"""
Residual Blocks
===============

Bloques residuales y conexiones residuales.
"""

import logging
from typing import Optional, Callable

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None

logger = logging.getLogger(__name__)


class ResidualBlock(nn.Module):
    """
    Bloque residual básico.
    
    Implementa conexión residual con normalización y activación.
    """
    
    def __init__(
        self,
        dim: int,
        activation: str = "relu",
        dropout: float = 0.1,
        use_norm: bool = True
    ):
        """
        Inicializar bloque residual.
        
        Args:
            dim: Dimensión
            activation: Función de activación
            dropout: Tasa de dropout
            use_norm: Usar normalización
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required")
        
        super().__init__()
        
        # Capas principales
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        
        # Normalización
        if use_norm:
            self.norm1 = nn.LayerNorm(dim)
            self.norm2 = nn.LayerNorm(dim)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()
        
        # Activación
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "swish":
            self.activation = nn.SiLU()
        else:
            self.activation = nn.ReLU()
        
        self.dropout = nn.Dropout(dropout)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Inicializar pesos."""
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch, ..., dim]
            
        Returns:
            Output tensor
        """
        # Primer bloque
        residual = x
        x = self.norm1(x)
        x = self.activation(x)
        x = self.fc1(x)
        x = self.dropout(x)
        
        # Segundo bloque
        x = self.norm2(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.dropout(x)
        
        # Conexión residual
        x = x + residual
        
        return x


class ResidualConnection(nn.Module):
    """
    Wrapper para conexión residual.
    
    Agrega conexión residual a cualquier módulo.
    """
    
    def __init__(
        self,
        module: nn.Module,
        dropout: float = 0.1,
        use_norm: bool = True
    ):
        """
        Inicializar conexión residual.
        
        Args:
            module: Módulo a envolver
            dropout: Tasa de dropout
            use_norm: Usar normalización
        """
        super().__init__()
        self.module = module
        self.dropout = nn.Dropout(dropout)
        self.use_norm = use_norm
        
        # Determinar dimensión para normalización
        if hasattr(module, 'embed_dim'):
            self.norm = nn.LayerNorm(module.embed_dim) if use_norm else nn.Identity()
        else:
            self.norm = nn.Identity()
    
    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor
            *args: Argumentos adicionales
            **kwargs: Keyword arguments adicionales
            
        Returns:
            Output tensor
        """
        residual = x
        
        # Aplicar módulo
        if args or kwargs:
            output = self.module(x, *args, **kwargs)
        else:
            output = self.module(x)
        
        # Normalizar
        if self.use_norm:
            output = self.norm(output)
        
        # Dropout y conexión residual
        output = self.dropout(output)
        output = output + residual
        
        return output

