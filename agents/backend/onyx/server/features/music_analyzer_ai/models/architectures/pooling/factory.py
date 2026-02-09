"""
Pooling Factory Module

Factory for creating pooling layers.
"""

from typing import Optional
import logging
import torch.nn as nn

from .mean import MeanPooling
from .max import MaxPooling
from .attention import AttentionPooling
from .adaptive import AdaptivePooling

logger = logging.getLogger(__name__)


class PoolingFactory:
    """Factory for creating pooling layers."""
    
    @staticmethod
    def create(
        pooling_type: str,
        embed_dim: Optional[int] = None,
        **kwargs
    ) -> nn.Module:
        """
        Create pooling layer.
        
        Args:
            pooling_type: Type of pooling.
            embed_dim: Embedding dimension (for attention pooling).
            **kwargs: Pooling-specific arguments.
        
        Returns:
            Pooling module.
        """
        pooling_type = pooling_type.lower()
        
        if pooling_type == "mean":
            return MeanPooling(**kwargs)
        elif pooling_type == "max":
            return MaxPooling(**kwargs)
        elif pooling_type == "attention":
            if embed_dim is None:
                raise ValueError("embed_dim required for attention pooling")
            return AttentionPooling(embed_dim)
        elif pooling_type == "adaptive":
            return AdaptivePooling(embed_dim=embed_dim, **kwargs)
        else:
            raise ValueError(f"Unknown pooling type: {pooling_type}")


def create_pooling(pooling_type: str, embed_dim: Optional[int] = None, **kwargs) -> nn.Module:
    """Convenience function for creating pooling layers."""
    return PoolingFactory.create(pooling_type, embed_dim, **kwargs)



