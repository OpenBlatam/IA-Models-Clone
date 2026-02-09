"""
Model Architecture Improvements
================================
Improved model architectures with best practices
"""

from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
from torch.nn import functional as F
import structlog
import math

from .config_loader import config_loader

logger = structlog.get_logger()


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention with proper implementation
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        bias: bool = True
    ):
        """
        Initialize multi-head attention
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
            bias: Use bias
        """
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        self.dropout = nn.Dropout(dropout)
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Initialize parameters"""
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            query: Query tensor [batch, seq_len, embed_dim]
            key: Key tensor [batch, seq_len, embed_dim]
            value: Value tensor [batch, seq_len, embed_dim]
            mask: Attention mask (optional)
            
        Returns:
            (output, attention_weights)
        """
        batch_size, seq_len, _ = query.size()
        
        # Project to Q, K, V
        Q = self.q_proj(query).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.embed_dim
        )
        
        output = self.out_proj(attn_output)
        
        return output, attn_weights


class TransformerBlock(nn.Module):
    """
    Transformer block with layer normalization and residual connections
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        ffn_dim: int = None,
        dropout: float = 0.1,
        activation: str = "gelu"
    ):
        """
        Initialize transformer block
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            ffn_dim: Feed-forward dimension (None = 4 * embed_dim)
            dropout: Dropout rate
            activation: Activation function
        """
        super().__init__()
        
        if ffn_dim is None:
            ffn_dim = 4 * embed_dim
        
        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            self._get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout)
        )
    
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function"""
        if activation == "gelu":
            return nn.GELU()
        elif activation == "relu":
            return nn.ReLU()
        elif activation == "swish":
            return nn.SiLU()
        else:
            return nn.GELU()
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor [batch, seq_len, embed_dim]
            mask: Attention mask (optional)
            
        Returns:
            Output tensor
        """
        # Self-attention with residual
        attn_output, _ = self.attention(x, x, x, mask)
        x = self.norm1(x + attn_output)
        
        # Feed-forward with residual
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        
        return x


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformers
    """
    
    def __init__(
        self,
        embed_dim: int,
        max_len: int = 5000,
        dropout: float = 0.1
    ):
        """
        Initialize positional encoding
        
        Args:
            embed_dim: Embedding dimension
            max_len: Maximum sequence length
            dropout: Dropout rate
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        # Create positional encoding
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, embed_dim]
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding
        
        Args:
            x: Input tensor [batch, seq_len, embed_dim]
            
        Returns:
            Tensor with positional encoding
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class ImprovedPersonalityModel(nn.Module):
    """
    Improved personality model with better architecture
    """
    
    def __init__(
        self,
        vocab_size: int = 30522,
        embed_dim: int = 768,
        num_heads: int = 12,
        num_layers: int = 6,
        max_len: int = 512,
        num_traits: int = 5,
        dropout: float = 0.1
    ):
        """
        Initialize improved personality model
        
        Args:
            vocab_size: Vocabulary size
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            max_len: Maximum sequence length
            num_traits: Number of personality traits
            dropout: Dropout rate
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_traits = num_traits
        
        # Embedding layer
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_encoding = PositionalEncoding(embed_dim, max_len, dropout)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        # Classification heads for each trait
        self.trait_heads = nn.ModuleDict({
            "openness": nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim // 2, 1),
                nn.Sigmoid()
            ),
            "conscientiousness": nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim // 2, 1),
                nn.Sigmoid()
            ),
            "extraversion": nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim // 2, 1),
                nn.Sigmoid()
            ),
            "agreeableness": nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim // 2, 1),
                nn.Sigmoid()
            ),
            "neuroticism": nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim // 2, 1),
                nn.Sigmoid()
            )
        })
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            input_ids: Input token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            
        Returns:
            Dictionary of trait predictions
        """
        # Embeddings
        x = self.token_embedding(input_ids)
        x = self.positional_encoding(x)
        
        # Transformer blocks
        for block in self.transformer_blocks:
            x = block(x, mask=attention_mask)
        
        # Pooling (use CLS token or mean pooling)
        if attention_mask is not None:
            x = x * attention_mask.unsqueeze(-1)
            pooled = x.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
        else:
            pooled = x.mean(dim=1)
        
        pooled = self.dropout(pooled)
        
        # Trait predictions
        predictions = {}
        for trait, head in self.trait_heads.items():
            predictions[trait] = head(pooled)
        
        return predictions




