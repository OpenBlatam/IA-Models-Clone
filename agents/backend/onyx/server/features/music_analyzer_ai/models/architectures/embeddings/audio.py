"""
Audio Feature Embedding Module

Specialized embedding for audio features (MFCC, chroma, etc.).
"""

import logging
import torch
import torch.nn as nn

from .base import FeatureEmbedding

logger = logging.getLogger(__name__)


class AudioFeatureEmbedding(FeatureEmbedding):
    """
    Specialized embedding for audio features (MFCC, chroma, etc.).
    
    Args:
        input_dim: Input feature dimension.
        embed_dim: Embedding dimension.
        dropout: Dropout probability.
        use_layer_norm: If True, apply layer normalization.
    """
    
    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        dropout: float = 0.1,
        use_layer_norm: bool = True
    ):
        super().__init__(input_dim, embed_dim, dropout)
        self.use_layer_norm = use_layer_norm
        
        if use_layer_norm:
            self.norm = nn.LayerNorm(embed_dim)
        
        logger.debug(f"Initialized AudioFeatureEmbedding with use_layer_norm={use_layer_norm}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with optional normalization.
        
        Args:
            x: Input tensor [..., input_dim]
        
        Returns:
            Embedded tensor [..., embed_dim]
        """
        x = self.embedding(x)
        if self.use_layer_norm:
            x = self.norm(x)
        x = self.dropout(x)
        return x



