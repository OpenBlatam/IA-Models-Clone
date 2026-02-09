"""
Clothing Encoder
================

Encodes text descriptions into clothing embeddings.
"""

import torch
import torch.nn as nn

from ..constants import (
    CLOTHING_EMBEDDING_DIM,
    DROPOUT_RATE,
)


class ClothingEncoder(nn.Module):
    """Encodes text descriptions into clothing embeddings."""
    
    def __init__(self, text_hidden_size: int = 768, embedding_dim: int = CLOTHING_EMBEDDING_DIM):
        """
        Initialize clothing encoder.
        
        Args:
            text_hidden_size: Text encoder hidden size
            embedding_dim: Output embedding dimension
        """
        super().__init__()
        intermediate_size = max(1024, embedding_dim * 2)
        
        self.clothing_encoder = nn.Sequential(
            nn.Linear(text_hidden_size, intermediate_size),
            nn.LayerNorm(intermediate_size),
            nn.GELU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(intermediate_size, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )
        
        self.residual_proj = nn.Linear(text_hidden_size, embedding_dim)
    
    def forward(self, text_features: torch.Tensor) -> torch.Tensor:
        """
        Encode text features with residual connection.
        
        Args:
            text_features: Text encoder features
            
        Returns:
            Clothing embeddings
        """
        encoded = self.clothing_encoder(text_features)
        residual = self.residual_proj(text_features)
        return encoded + residual


