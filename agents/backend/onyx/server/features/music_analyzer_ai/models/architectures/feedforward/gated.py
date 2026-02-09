"""
Gated Feedforward Network Module

Implements gated feedforward network (GLU variant).
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class GatedFeedForward(nn.Module):
    """
    Gated feedforward network (GLU variant).
    
    Args:
        embed_dim: Embedding dimension.
        ff_dim: Feedforward hidden dimension.
        dropout: Dropout probability.
        activation: Activation function name ("gelu", "relu").
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
        logger.debug(f"Initialized GatedFeedForward with embed_dim={embed_dim}, ff_dim={ff_dim}, activation='{activation}'")
    
    def _reset_parameters(self):
        """Initialize gated feedforward parameters."""
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight, gain=0.1)
        if self.fc1.bias is not None:
            nn.init.zeros_(self.fc1.bias)
        if self.fc2.bias is not None:
            nn.init.zeros_(self.fc2.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with gating mechanism.
        
        Args:
            x: Input tensor [batch, seq_len, embed_dim]
        
        Returns:
            Output tensor [batch, seq_len, embed_dim]
        """
        x = self.fc1(x)
        # Split into gate and value
        gate, value = x.chunk(2, dim=-1)
        x = self.activation(gate) * value
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x



