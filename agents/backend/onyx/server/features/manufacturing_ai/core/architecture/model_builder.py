"""
Model Builder
=============

Builder pattern para construir modelos complejos.
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None

from .attention_layers import MultiHeadAttention, SelfAttention
from .residual_blocks import ResidualBlock, ResidualConnection
from .normalization import LayerNorm
from .activations import get_activation

logger = logging.getLogger(__name__)


@dataclass
class ArchitectureConfig:
    """Configuración de arquitectura."""
    input_size: int
    output_size: int
    hidden_sizes: List[int] = field(default_factory=lambda: [128, 64, 32])
    activation: str = "gelu"
    dropout: float = 0.1
    use_residual: bool = True
    use_attention: bool = False
    num_attention_heads: int = 8
    use_norm: bool = True
    norm_type: str = "layer"  # layer, batch, group
    metadata: Dict[str, Any] = field(default_factory=dict)


class ModelBuilder:
    """
    Builder para modelos.
    
    Construye modelos complejos con diferentes componentes.
    """
    
    def __init__(self, config: ArchitectureConfig):
        """
        Inicializar builder.
        
        Args:
            config: Configuración de arquitectura
        """
        self.config = config
        self.layers = []
    
    def add_linear_layer(self, in_features: int, out_features: int, bias: bool = True):
        """Agregar capa lineal."""
        layer = nn.Linear(in_features, out_features, bias=bias)
        nn.init.xavier_uniform_(layer.weight)
        if bias:
            nn.init.zeros_(layer.bias)
        self.layers.append(layer)
        return self
    
    def add_activation(self, activation: Optional[str] = None):
        """Agregar activación."""
        act_name = activation or self.config.activation
        self.layers.append(get_activation(act_name))
        return self
    
    def add_normalization(self, dim: int, norm_type: Optional[str] = None):
        """Agregar normalización."""
        norm_type = norm_type or self.config.norm_type
        
        if norm_type == "layer":
            self.layers.append(LayerNorm(dim))
        elif norm_type == "batch":
            self.layers.append(nn.BatchNorm1d(dim))
        else:
            self.layers.append(nn.Identity())
        
        return self
    
    def add_dropout(self, p: Optional[float] = None):
        """Agregar dropout."""
        p = p if p is not None else self.config.dropout
        if p > 0:
            self.layers.append(nn.Dropout(p))
        return self
    
    def add_attention(self, embed_dim: int, num_heads: Optional[int] = None):
        """Agregar capa de atención."""
        if not self.config.use_attention:
            return self
        
        num_heads = num_heads or self.config.num_attention_heads
        attention = SelfAttention(embed_dim, num_heads, self.config.dropout)
        
        if self.config.use_residual:
            attention = ResidualConnection(attention, self.config.dropout, self.config.use_norm)
        
        self.layers.append(attention)
        return self
    
    def add_residual_block(self, dim: int):
        """Agregar bloque residual."""
        if not self.config.use_residual:
            return self
        
        block = ResidualBlock(
            dim,
            activation=self.config.activation,
            dropout=self.config.dropout,
            use_norm=self.config.use_norm
        )
        self.layers.append(block)
        return self
    
    def build_mlp(self) -> nn.Module:
        """
        Construir MLP.
        
        Returns:
            Modelo MLP
        """
        layers = []
        prev_size = self.config.input_size
        
        for hidden_size in self.config.hidden_sizes:
            # Linear
            layers.append(nn.Linear(prev_size, hidden_size))
            nn.init.xavier_uniform_(layers[-1].weight)
            nn.init.zeros_(layers[-1].bias)
            
            # Normalization
            if self.config.use_norm:
                layers.append(LayerNorm(hidden_size))
            
            # Activation
            layers.append(get_activation(self.config.activation))
            
            # Dropout
            if self.config.dropout > 0:
                layers.append(nn.Dropout(self.config.dropout))
            
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, self.config.output_size))
        nn.init.xavier_uniform_(layers[-1].weight)
        nn.init.zeros_(layers[-1].bias)
        
        return nn.Sequential(*layers)
    
    def build_transformer(self) -> nn.Module:
        """
        Construir Transformer.
        
        Returns:
            Modelo Transformer
        """
        # Implementación simplificada
        # En producción usaría arquitectura completa de Transformer
        embed_dim = self.config.hidden_sizes[0] if self.config.hidden_sizes else 128
        
        class TransformerModel(nn.Module):
            def __init__(self, config):
                super().__init__()
                self.input_proj = nn.Linear(config.input_size, embed_dim)
                self.attention = SelfAttention(embed_dim, config.num_attention_heads, config.dropout)
                self.output_proj = nn.Linear(embed_dim, config.output_size)
            
            def forward(self, x):
                x = self.input_proj(x)
                x = self.attention(x)
                x = self.output_proj(x)
                return x
        
        return TransformerModel(self.config)
    
    def build(self) -> nn.Module:
        """
        Construir modelo usando layers agregadas.
        
        Returns:
            Modelo construido
        """
        if not self.layers:
            # Construir MLP por defecto
            return self.build_mlp()
        
        return nn.Sequential(*self.layers)

