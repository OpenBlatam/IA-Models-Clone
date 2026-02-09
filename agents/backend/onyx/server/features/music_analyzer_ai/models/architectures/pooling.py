"""
Modular Pooling Layers
Various pooling strategies for sequence and feature aggregation
"""

from typing import Optional
import logging

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available")


class MeanPooling(nn.Module):
    """Mean pooling over sequence dimension"""
    
    def __init__(self, dim: int = 1):
        super().__init__()
        self.dim = dim
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, features]
            mask: Optional mask [batch, seq_len]
        
        Returns:
            [batch, features]
        """
        if mask is not None:
            mask_expanded = (~mask).unsqueeze(-1).float()
            masked_sum = (x * mask_expanded).sum(dim=self.dim)
            mask_count = mask_expanded.sum(dim=self.dim)
            return masked_sum / (mask_count + 1e-8)
        return x.mean(dim=self.dim)


class MaxPooling(nn.Module):
    """Max pooling over sequence dimension"""
    
    def __init__(self, dim: int = 1):
        super().__init__()
        self.dim = dim
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Max pooling with optional mask"""
        if mask is not None:
            # Set masked positions to very negative values
            mask_expanded = (~mask).unsqueeze(-1).float()
            x = x * mask_expanded + (1 - mask_expanded) * float('-inf')
        return x.max(dim=self.dim)[0]


class AttentionPooling(nn.Module):
    """Attention-based pooling"""
    
    def __init__(self, embed_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Tanh(),
            nn.Linear(embed_dim, 1)
        )
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, embed_dim]
            mask: Optional mask [batch, seq_len]
        
        Returns:
            [batch, embed_dim]
        """
        # Compute attention weights
        attn_weights = self.attention(x).squeeze(-1)  # [batch, seq_len]
        
        # Apply mask if provided
        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask, float('-inf'))
        
        # Softmax
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        # Weighted sum
        output = torch.bmm(attn_weights.unsqueeze(1), x).squeeze(1)  # [batch, embed_dim]
        return output


class AdaptivePooling(nn.Module):
    """Adaptive pooling that can switch between strategies"""
    
    def __init__(self, strategy: str = "mean", embed_dim: Optional[int] = None):
        super().__init__()
        self.strategy = strategy
        
        if strategy == "mean":
            self.pooler = MeanPooling()
        elif strategy == "max":
            self.pooler = MaxPooling()
        elif strategy == "attention":
            if embed_dim is None:
                raise ValueError("embed_dim required for attention pooling")
            self.pooler = AttentionPooling(embed_dim)
        else:
            raise ValueError(f"Unknown pooling strategy: {strategy}")
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass"""
        return self.pooler(x, mask)


class PoolingFactory:
    """Factory for creating pooling layers"""
    
    @staticmethod
    def create(
        pooling_type: str,
        embed_dim: Optional[int] = None,
        **kwargs
    ) -> nn.Module:
        """
        Create pooling layer
        
        Args:
            pooling_type: Type of pooling
            embed_dim: Embedding dimension (for attention pooling)
            **kwargs: Pooling-specific arguments
        
        Returns:
            Pooling module
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
    """Convenience function for creating pooling layers"""
    return PoolingFactory.create(pooling_type, embed_dim, **kwargs)



