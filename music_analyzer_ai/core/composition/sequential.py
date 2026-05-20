"""
Sequential Composer Module

Simplified composer for sequential models.
"""

from typing import List
import logging

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available")


class SequentialComposer:
    """
    Simplified composer for sequential models.
    """
    
    def __init__(self):
        self.layers: List[nn.Module] = []
    
    def add(self, layer: nn.Module) -> 'SequentialComposer':
        """
        Add a layer.
        
        Args:
            layer: Layer module.
        
        Returns:
            Self for chaining.
        """
        self.layers.append(layer)
        return self
    
    def build(self) -> nn.Sequential:
        """
        Build sequential model.
        
        Returns:
            Sequential model.
        """
        return nn.Sequential(*self.layers)



