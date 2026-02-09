"""
Feature Pooler
===============

Handles feature pooling from CLIP vision outputs with improved methods.
"""

import torch
import torch.nn.functional as F
from typing import Optional

from ..constants import (
    CLS_POOLING_WEIGHT,
    MEAN_POOLING_WEIGHT,
    MAX_POOLING_WEIGHT,
    ATTN_POOLING_WEIGHT,
)


class FeaturePooler:
    """Handles feature pooling from CLIP vision outputs with improved methods."""
    
    @staticmethod
    def pool_features(image_features: torch.Tensor) -> torch.Tensor:
        """
        Enhanced pooling: CLS token + mean pooling + max pooling + attention pooling.
        
        Args:
            image_features: [batch, seq_len, hidden_size]
            
        Returns:
            Pooled features [batch, hidden_size]
        """
        # CLS token (global context)
        cls_features = image_features[:, 0, :]
        
        # Mean pooling (average features)
        mean_features = image_features.mean(dim=1)
        
        # Max pooling (strongest features)
        max_features = image_features.max(dim=1)[0]
        
        # Attention-based pooling (learned attention)
        attention_weights = F.softmax(
            torch.sum(image_features * cls_features.unsqueeze(1), dim=-1),
            dim=1
        )
        attn_pooled = torch.sum(
            image_features * attention_weights.unsqueeze(-1),
            dim=1
        )
        
        # Weighted combination
        pooled = (
            CLS_POOLING_WEIGHT * cls_features +
            MEAN_POOLING_WEIGHT * mean_features +
            MAX_POOLING_WEIGHT * max_features +
            ATTN_POOLING_WEIGHT * attn_pooled
        )
        
        return pooled


