"""
Modular Embedding Layers
Various embedding strategies for music features
"""

import logging
import math

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available")


class FeatureEmbedding(nn.Module):
    """
    Base feature embedding layer
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
    
    def _reset_parameters(self):
        """Initialize embedding parameters"""
        nn.init.xavier_uniform_(self.embedding.weight)
        if self.embedding.bias is not None:
            nn.init.zeros_(self.embedding.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        x = self.embedding(x)
        x = self.dropout(x)
        return x


class AudioFeatureEmbedding(FeatureEmbedding):
    """
    Specialized embedding for audio features (MFCC, chroma, etc.)
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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with optional normalization"""
        x = self.embedding(x)
        if self.use_layer_norm:
            x = self.norm(x)
        x = self.dropout(x)
        return x


class MusicFeatureEmbedding(FeatureEmbedding):
    """
    Specialized embedding for music features (Spotify audio features, etc.)
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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with optional group embeddings"""
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



