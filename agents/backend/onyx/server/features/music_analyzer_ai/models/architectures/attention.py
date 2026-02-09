"""
Modular Attention Mechanisms
Implements various attention mechanisms following best practices
"""

from typing import Optional, Tuple
import math
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


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention with proper scaling and masking
    """
    
    def __init__(
        self,
        head_dim: int,
        dropout: float = 0.1,
        scale: Optional[float] = None
    ):
        super().__init__()
        self.head_dim = head_dim
        self.scale = scale if scale is not None else 1.0 / math.sqrt(head_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: [batch, num_heads, seq_len, head_dim]
            key: [batch, num_heads, seq_len, head_dim]
            value: [batch, num_heads, seq_len, head_dim]
            mask: [batch, seq_len, seq_len] or [batch, 1, seq_len, seq_len]
        
        Returns:
            output: [batch, num_heads, seq_len, head_dim]
            attention_weights: [batch, num_heads, seq_len, seq_len]
        """
        # Compute attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        
        # Apply mask if provided
        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)  # [batch, 1, seq_len, seq_len]
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax with numerical stability
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Check for NaN/Inf
        if torch.isnan(attn_weights).any() or torch.isinf(attn_weights).any():
            logger.warning("NaN/Inf in attention weights, applying fix")
            attn_weights = torch.nan_to_num(attn_weights, nan=0.0, posinf=1.0, neginf=0.0)
            attn_weights = attn_weights / (attn_weights.sum(dim=-1, keepdim=True) + 1e-8)
        
        # Apply attention to values
        output = torch.matmul(attn_weights, value)
        
        return output, attn_weights


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention with modular components
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        bias: bool = True,
        use_flash_attention: bool = False
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.use_flash_attention = use_flash_attention
        
        # Projection layers
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        # Attention mechanism
        self.attention = ScaledDotProductAttention(
            head_dim=self.head_dim,
            dropout=dropout
        )
        
        # Initialize weights
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Initialize attention parameters"""
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        
        if self.q_proj.bias is not None:
            nn.init.zeros_(self.q_proj.bias)
            nn.init.zeros_(self.k_proj.bias)
            nn.init.zeros_(self.v_proj.bias)
            nn.init.zeros_(self.out_proj.bias)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            query: [batch, seq_len, embed_dim]
            key: [batch, seq_len, embed_dim]
            value: [batch, seq_len, embed_dim]
            mask: [batch, seq_len, seq_len] or [batch, 1, seq_len, seq_len]
        
        Returns:
            output: [batch, seq_len, embed_dim]
        """
        batch_size, seq_len, _ = query.size()
        
        # Project and reshape for multi-head attention
        Q = self.q_proj(query).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply attention
        attn_output, _ = self.attention(Q, K, V, mask)
        
        # Concatenate heads and reshape
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.embed_dim
        )
        
        # Output projection
        output = self.out_proj(attn_output)
        return output


class AttentionLayer(nn.Module):
    """
    Complete attention layer with residual connection and normalization
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        pre_norm: bool = True
    ):
        super().__init__()
        self.pre_norm = pre_norm
        
        self.attention = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, embed_dim]
            mask: Optional attention mask
        
        Returns:
            output: [batch, seq_len, embed_dim]
        """
        residual = x
        
        if self.pre_norm:
            # Pre-norm: normalize before attention
            x = self.norm(x)
            x = self.attention(x, x, x, mask)
            x = self.dropout(x)
            x = x + residual
        else:
            # Post-norm: normalize after attention
            x = self.attention(x, x, x, mask)
            x = self.dropout(x)
            x = x + residual
            x = self.norm(x)
        
        return x



