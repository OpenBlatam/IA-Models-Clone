"""
Attention Pooling Module

Implements attention-based pooling.
"""

from typing import Optional
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class AttentionPooling(nn.Module):
    """
    Attention-based pooling.
    
    Args:
        embed_dim: Embedding dimension.
    """
    
    def __init__(self, embed_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Tanh(),
            nn.Linear(embed_dim, 1)
        )
        logger.debug(f"Initialized AttentionPooling with embed_dim={embed_dim}")
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply attention-based pooling.
        
        Args:
            x: Input tensor [batch, seq_len, embed_dim]
            mask: Optional mask [batch, seq_len]
        
        Returns:
            Pooled tensor [batch, embed_dim]
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



