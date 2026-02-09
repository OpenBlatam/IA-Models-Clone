"""
Character Encoder
=================

Encodes CLIP features into character embeddings with residual connections.
"""

import torch
import torch.nn as nn

from ..constants import (
    CHARACTER_EMBEDDING_DIM,
    MIN_INTERMEDIATE_SIZE_MULTIPLIER,
    DROPOUT_RATE,
)


class CharacterEncoder(nn.Module):
    """Encodes CLIP features into character embeddings with residual connections."""
    
    def __init__(self, clip_hidden_size: int, embedding_dim: int = CHARACTER_EMBEDDING_DIM):
        """
        Initialize character encoder.
        
        Args:
            clip_hidden_size: CLIP hidden size
            embedding_dim: Output embedding dimension
        """
        super().__init__()
        intermediate_size = max(2048, embedding_dim * MIN_INTERMEDIATE_SIZE_MULTIPLIER)
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(clip_hidden_size, intermediate_size),
            nn.LayerNorm(intermediate_size),
            nn.GELU(),
            nn.Dropout(DROPOUT_RATE),
        )
        
        self.character_encoder = nn.Sequential(
            nn.Linear(intermediate_size, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )
        
        self.residual_proj = nn.Linear(clip_hidden_size, embedding_dim)
    
    def forward(self, pooled_features: torch.Tensor) -> torch.Tensor:
        """
        Encode features with residual connection.
        
        Args:
            pooled_features: Pooled CLIP features
            
        Returns:
            Character embeddings
        """
        extracted = self.feature_extractor(pooled_features)
        encoded = self.character_encoder(extracted)
        residual = self.residual_proj(pooled_features)
        return encoded + residual


