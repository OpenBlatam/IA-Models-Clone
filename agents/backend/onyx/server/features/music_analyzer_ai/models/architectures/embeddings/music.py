"""
Music Feature Embedding Module

Specialized embedding for music features (Spotify audio features, etc.).
"""

import logging
import torch
import torch.nn as nn

from .base import FeatureEmbedding

logger = logging.getLogger(__name__)


class MusicFeatureEmbedding(FeatureEmbedding):
    """
    Specialized embedding for music features (Spotify audio features, etc.).
    
    Args:
        input_dim: Input feature dimension.
        embed_dim: Embedding dimension.
        dropout: Dropout probability.
        feature_groups: Optional dictionary mapping feature group names to dimensions.
    """
    
    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        dropout: float = 0.1,
        feature_groups: dict = None
    ):
        super().__init__(input_dim, embed_dim, dropout)
        self.feature_groups = feature_groups or {}
        
        # Optional: separate embeddings for different feature groups
        if feature_groups:
            self.group_embeddings = nn.ModuleDict({
                name: nn.Linear(dim, embed_dim // len(feature_groups))
                for name, dim in feature_groups.items()
            })
        
        logger.debug(f"Initialized MusicFeatureEmbedding with feature_groups={bool(feature_groups)}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with optional group embeddings.
        
        Args:
            x: Input tensor [..., input_dim]
        
        Returns:
            Embedded tensor [..., embed_dim]
        """
        if self.feature_groups and hasattr(self, 'group_embeddings'):
            # Concatenate group embeddings
            group_embeds = []
            start_idx = 0
            for name, dim in self.feature_groups.items():
                group_x = x[..., start_idx:start_idx + dim]
                group_embeds.append(self.group_embeddings[name](group_x))
                start_idx += dim
            x = torch.cat(group_embeds, dim=-1)
        else:
            x = self.embedding(x)
        
        x = self.dropout(x)
        return x



