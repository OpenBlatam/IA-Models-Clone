"""
Modular Dropout Layers
Various dropout strategies for regularization
"""

from typing import Optional
import logging

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available")


class StandardDropout(nn.Module):
    """Standard dropout layer"""
    
    def __init__(self, p: float = 0.5, inplace: bool = False):
        super().__init__()
        self.dropout = nn.Dropout(p=p, inplace=inplace)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x)


class SpatialDropout(nn.Module):
    """Spatial dropout for 2D/3D tensors"""
    
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply spatial dropout"""
        if x.dim() == 3:  # [batch, seq_len, features]
            # Drop entire feature vectors
            mask = torch.bernoulli(torch.ones(x.shape[:2]) * (1 - self.p))
            mask = mask.unsqueeze(-1).to(x.device)
            return x * mask / (1 - self.p)
        elif x.dim() == 4:  # [batch, channels, height, width]
            # Drop entire channels
            mask = torch.bernoulli(torch.ones(x.shape[:2]) * (1 - self.p))
            mask = mask.unsqueeze(-1).unsqueeze(-1).to(x.device)
            return x * mask / (1 - self.p)
        else:
            # Fallback to standard dropout
            return nn.functional.dropout(x, p=self.p, training=self.training)


class AlphaDropout(nn.Module):
    """Alpha dropout for self-normalizing networks"""
    
    def __init__(self, p: float = 0.5, alpha: float = -1.7580993408473766):
        super().__init__()
        self.p = p
        self.alpha = alpha
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply alpha dropout"""
        if not self.training:
            return x
        
        # Alpha dropout formula
        alpha = self.alpha
        keep_prob = 1 - self.p
        noise = torch.rand_like(x)
        noise = noise + alpha
        noise = torch.bernoulli(noise)
        noise = noise * (keep_prob + alpha * keep_prob)
        
        return x * noise


class DropoutFactory:
    """Factory for creating dropout layers"""
    
    @staticmethod
    def create(dropout_type: str = "standard", p: float = 0.5, **kwargs) -> nn.Module:
        """
        Create dropout layer
        
        Args:
            dropout_type: Type of dropout ("standard", "spatial", "alpha")
            p: Dropout probability
            **kwargs: Dropout-specific arguments
        
        Returns:
            Dropout module
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
    """Convenience function for creating dropout layers"""
    return DropoutFactory.create(dropout_type, p, **kwargs)



