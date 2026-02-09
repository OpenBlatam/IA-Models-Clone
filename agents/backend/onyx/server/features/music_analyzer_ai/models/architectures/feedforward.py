"""
Modular Feedforward Networks
Various feedforward architectures for transformers and deep networks
"""

import logging
import math

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available")


class FeedForward(nn.Module):
    """
    Standard feedforward network with activation and dropout
    """
    
    def __init__(
        self,
        embed_dim: int,
        ff_dim: int,
        dropout: float = 0.1,
        activation: str = "gelu"
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim
        
        self.fc1 = nn.Linear(embed_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Activation function
        if activation == "gelu":
            self.activation = F.gelu
        elif activation == "relu":
            self.activation = F.relu
        elif activation == "swish":
            self.activation = lambda x: x * torch.sigmoid(x)
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Initialize feedforward parameters"""
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight, gain=0.1)
        if self.fc1.bias is not None:
            nn.init.zeros_(self.fc1.bias)
        if self.fc2.bias is not None:
            nn.init.zeros_(self.fc2.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class GatedFeedForward(nn.Module):
    """
    Gated feedforward network (GLU variant)
    """
    
    def __init__(
        self,
        embed_dim: int,
        ff_dim: int,
        dropout: float = 0.1,
        activation: str = "gelu"
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim
        
        self.fc1 = nn.Linear(embed_dim, ff_dim * 2)  # Double for gating
        self.fc2 = nn.Linear(ff_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Activation function
        if activation == "gelu":
            self.activation = F.gelu
        elif activation == "relu":
            self.activation = F.relu
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Initialize gated feedforward parameters"""
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight, gain=0.1)
        if self.fc1.bias is not None:
            nn.init.zeros_(self.fc1.bias)
        if self.fc2.bias is not None:
            nn.init.zeros_(self.fc2.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with gating"""
        x = self.fc1(x)
        # Split into gate and value
        gate, value = x.chunk(2, dim=-1)
        x = self.activation(gate) * value
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class ResidualFeedForward(nn.Module):
    """
    Feedforward network with residual connection and normalization
    """
    
    def __init__(
        self,
        embed_dim: int,
        ff_dim: int,
        dropout: float = 0.1,
        activation: str = "gelu",
        pre_norm: bool = True
    ):
        super().__init__()
        self.pre_norm = pre_norm
        
        self.ff = FeedForward(
            embed_dim=embed_dim,
            ff_dim=ff_dim,
            dropout=dropout,
            activation=activation
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection"""
        residual = x
        
        if self.pre_norm:
            # Pre-norm: normalize before feedforward
            x = self.norm(x)
            x = self.ff(x)
            x = self.dropout(x)
            x = x + residual
        else:
            # Post-norm: normalize after feedforward
            x = self.ff(x)
            x = self.dropout(x)
            x = x + residual
            x = self.norm(x)
        
        return x



