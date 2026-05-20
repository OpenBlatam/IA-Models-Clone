"""
Component Factory
=================

Factory para crear componentes arquitectónicos.
"""

import logging
from typing import Dict, Any, Optional

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None

from .attention_layers import MultiHeadAttention, SelfAttention, CrossAttention
from .residual_blocks import ResidualBlock, ResidualConnection
from .normalization import LayerNorm, BatchNorm1d, GroupNorm
from .activations import get_activation

logger = logging.getLogger(__name__)


class ComponentFactory:
    """
    Factory para componentes arquitectónicos.
    
    Crea componentes reutilizables de manera consistente.
    """
    
    @staticmethod
    def create_attention(
        embed_dim: int,
        num_heads: int = 8,
        attention_type: str = "self",
        dropout: float = 0.1
    ) -> nn.Module:
        """
        Crear capa de atención.
        
        Args:
            embed_dim: Dimensión de embedding
            num_heads: Número de heads
            attention_type: Tipo (self, cross, multi)
            dropout: Tasa de dropout
            
        Returns:
            Módulo de atención
        """
        if attention_type == "self":
            return SelfAttention(embed_dim, num_heads, dropout)
        elif attention_type == "cross":
            return CrossAttention(embed_dim, num_heads, dropout)
        elif attention_type == "multi":
            return MultiHeadAttention(embed_dim, num_heads, dropout)
        else:
            raise ValueError(f"Unknown attention type: {attention_type}")
    
    @staticmethod
    def create_residual_block(
        dim: int,
        activation: str = "gelu",
        dropout: float = 0.1,
        use_norm: bool = True
    ) -> ResidualBlock:
        """
        Crear bloque residual.
        
        Args:
            dim: Dimensión
            activation: Activación
            dropout: Dropout
            use_norm: Usar normalización
            
        Returns:
            Bloque residual
        """
        return ResidualBlock(dim, activation, dropout, use_norm)
    
    @staticmethod
    def create_normalization(
        norm_type: str,
        num_features: int,
        num_groups: Optional[int] = None
    ) -> nn.Module:
        """
        Crear normalización.
        
        Args:
            norm_type: Tipo (layer, batch, group)
            num_features: Número de features
            num_groups: Número de grupos (para GroupNorm)
            
        Returns:
            Módulo de normalización
        """
        if norm_type == "layer":
            return LayerNorm(num_features)
        elif norm_type == "batch":
            return BatchNorm1d(num_features)
        elif norm_type == "group":
            if num_groups is None:
                num_groups = max(1, num_features // 32)
            return GroupNorm(num_groups, num_features)
        else:
            return nn.Identity()
    
    @staticmethod
    def create_feed_forward(
        dim: int,
        hidden_dim: Optional[int] = None,
        activation: str = "gelu",
        dropout: float = 0.1
    ) -> nn.Module:
        """
        Crear feed-forward network.
        
        Args:
            dim: Dimensión de entrada/salida
            hidden_dim: Dimensión oculta (4*dim por defecto)
            activation: Activación
            dropout: Dropout
            
        Returns:
            Feed-forward network
        """
        if hidden_dim is None:
            hidden_dim = dim * 4
        
        return nn.Sequential(
            nn.Linear(dim, hidden_dim),
            get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    
    @staticmethod
    def create_transformer_block(
        embed_dim: int,
        num_heads: int = 8,
        ff_hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
        activation: str = "gelu"
    ) -> nn.Module:
        """
        Crear bloque transformer completo.
        
        Args:
            embed_dim: Dimensión de embedding
            num_heads: Número de heads
            ff_hidden_dim: Dimensión oculta FFN
            dropout: Dropout
            activation: Activación
            
        Returns:
            Bloque transformer
        """
        attention = SelfAttention(embed_dim, num_heads, dropout)
        ff = ComponentFactory.create_feed_forward(embed_dim, ff_hidden_dim, activation, dropout)
        
        class TransformerBlock(nn.Module):
            def __init__(self, attn, ff, embed_dim, dropout):
                super().__init__()
                self.attn = ResidualConnection(attn, dropout, use_norm=True)
                self.ff = ResidualConnection(ff, dropout, use_norm=True)
                self.norm1 = LayerNorm(embed_dim)
                self.norm2 = LayerNorm(embed_dim)
            
            def forward(self, x):
                x = self.attn(x)
                x = self.ff(x)
                return x
        
        return TransformerBlock(attention, ff, embed_dim, dropout)

