"""
Feature Pooling Utilities
=========================

Utilities for pooling image features from CLIP outputs.
"""

import torch
import torch.nn.functional as F
from typing import Tuple

# Pooling constants
POOLING_METHODS_COUNT = 3  # CLS, mean, attention
POOLING_COMBINE_WEIGHT = 1.0 / POOLING_METHODS_COUNT


class FeaturePooler:
    """Helper class for pooling image features."""
    
    @staticmethod
    def pool_clip_features(image_features: torch.Tensor) -> torch.Tensor:
        """
        Enhanced pooling: CLS token + mean pooling + attention pooling.
        
        Args:
            image_features: CLIP features [batch, seq_len, hidden_size]
            
        Returns:
            Pooled features [batch, hidden_size]
        """
        # CLS token features
        cls_features = image_features[:, 0, :]  # [batch, hidden_size]
        
        # Mean pooling
        mean_features = image_features.mean(dim=1)  # [batch, hidden_size]
        
        # Attention-based pooling (learned attention over sequence)
        seq_len = image_features.size(1)
        attention_weights = F.softmax(
            torch.sum(image_features * cls_features.unsqueeze(1), dim=-1),
            dim=1
        )  # [batch, seq_len]
        
        attn_pooled = torch.sum(
            image_features * attention_weights.unsqueeze(-1),
            dim=1
        )  # [batch, hidden_size]
        
        # Combine pooling methods
        pooled_features = (cls_features + mean_features + attn_pooled) / 3.0
        
        return pooled_features
    
    @staticmethod
    def statistical_aggregation(embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Statistical aggregation of embeddings.
        
        Args:
            embeddings: Stacked embeddings [batch, num_images, embedding_dim]
            
        Returns:
            Tuple of (mean_aggregated, max_aggregated)
        """
        mean_aggregated = embeddings.mean(dim=1)  # [batch, embedding_dim]
        max_aggregated = embeddings.max(dim=1)[0]  # [batch, embedding_dim]
        return mean_aggregated, max_aggregated

