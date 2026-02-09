"""
Base Embedding Module

Base class for feature embeddings.
"""

import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class FeatureEmbedding(nn.Module):
    """
    Base feature embedding layer.
    
    Args:
        input_dim: Input feature dimension.
        embed_dim: Embedding dimension.
        dropout: Dropout probability.
    """
    
    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        dropout: float = 0.1
    ):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self._reset_parameters()
        logger.debug(f"Initialized FeatureEmbedding with input_dim={input_dim}, embed_dim={embed_dim}")
    
    def _reset_parameters(self):
        """Initialize embedding parameters."""
        nn.init.xavier_uniform_(self.embedding.weight)
        if self.embedding.bias is not None:
            nn.init.zeros_(self.embedding.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [..., input_dim]
        
        Returns:
            Embedded tensor [..., embed_dim]
        """
        x = self.embedding(x)
        x = self.dropout(x)
        return x



