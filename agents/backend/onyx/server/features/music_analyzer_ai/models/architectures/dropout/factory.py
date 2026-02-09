"""
Dropout Factory Module

Factory for creating dropout layers.
"""

import logging
import torch.nn as nn

from .standard import StandardDropout
from .spatial import SpatialDropout
from .alpha import AlphaDropout

logger = logging.getLogger(__name__)


class DropoutFactory:
    """Factory for creating dropout layers."""
    
    @staticmethod
    def create(dropout_type: str = "standard", p: float = 0.5, **kwargs) -> nn.Module:
        """
        Create dropout layer.
        
        Args:
            dropout_type: Type of dropout ("standard", "spatial", "alpha").
            p: Dropout probability.
            **kwargs: Dropout-specific arguments.
        
        Returns:
            Dropout module.
        """
        dropout_type = dropout_type.lower()
        
        if dropout_type == "standard":
            return StandardDropout(p=p, **kwargs)
        elif dropout_type == "spatial":
            return SpatialDropout(p=p)
        elif dropout_type == "alpha":
            return AlphaDropout(p=p, **kwargs)
        else:
            raise ValueError(f"Unknown dropout type: {dropout_type}")


def create_dropout(dropout_type: str = "standard", p: float = 0.5, **kwargs) -> nn.Module:
    """Convenience function for creating dropout layers."""
    return DropoutFactory.create(dropout_type, p, **kwargs)



