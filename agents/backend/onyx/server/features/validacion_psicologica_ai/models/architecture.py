"""
Model Architecture Components
=============================
Advanced architectural components for models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1
    ):
        """
        Initialize multi-head attention
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.q_lin = nn.Linear(embed_dim, embed_dim)
        self.k_lin = nn.Linear(embed_dim, embed_dim)
        self.v_lin = nn.Linear(embed_dim, embed_dim)
        self.out_lin = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            query: Query tensor [batch, seq_len, embed_dim]
            key: Key tensor [batch, seq_len, embed_dim]
            value: Value tensor [batch, seq_len, embed_dim]
            mask: Attention mask (optional)
            
        Returns:
            Output tensor and attention weights
        """
        batch_size, seq_len, _ = query.shape
        
        # Linear projections
        Q = self.q_lin(query).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_lin(key).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_lin(value).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
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
        
        output = self.out_lin(attn_output)
        return output, attn_weights


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformers
    """
    
    def __init__(self, embed_dim: int, max_len: int = 5000):
        """
        Initialize positional encoding
        
        Args:
            embed_dim: Embedding dimension
            max_len: Maximum sequence length
        """
        super().__init__()
        
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(0.1)
    
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


class TransformerBlock(nn.Module):
    """
    Transformer block with attention and feed-forward
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout: float = 0.1
    ):
        """
        Initialize transformer block
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            ff_dim: Feed-forward dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor [batch, seq_len, embed_dim]
            mask: Attention mask (optional)
            
        Returns:
            Output tensor
        """
        # Self-attention with residual
        attn_out, _ = self.attention(x, x, x, mask)
        x = self.norm1(x + attn_out)
        
        # Feed-forward with residual
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        
        return x


class ImprovedPersonalityModel(nn.Module):
    """
    Improved personality model with custom transformer architecture
    """
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        ff_dim: int = 512,
        num_traits: int = 5,
        max_len: int = 512,
        dropout: float = 0.1
    ):
        """
        Initialize improved personality model
        
        Args:
            vocab_size: Vocabulary size
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            ff_dim: Feed-forward dimension
            num_traits: Number of personality traits
            max_len: Maximum sequence length
            dropout: Dropout rate
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_traits = num_traits
        
        # Embedding and positional encoding
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = PositionalEncoding(embed_dim, max_len)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # Trait heads
        self.trait_heads = nn.ModuleDict({
            "openness": nn.Linear(embed_dim, 1),
            "conscientiousness": nn.Linear(embed_dim, 1),
            "extraversion": nn.Linear(embed_dim, 1),
            "agreeableness": nn.Linear(embed_dim, 1),
            "neuroticism": nn.Linear(embed_dim, 1)
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
            input_ids: Token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            
        Returns:
            Dictionary of trait predictions
        """
        # Embeddings
        x = self.token_embedding(input_ids)
        x = self.pos_encoding(x)
        
        # Transformer blocks
        for block in self.transformer_blocks:
            x = block(x, attention_mask)
        
        # Use [CLS] token (first token)
        cls_embedding = x[:, 0, :]
        cls_embedding = self.dropout(cls_embedding)
        
        # Trait predictions
        predictions = {}
        for trait, head in self.trait_heads.items():
            predictions[trait] = torch.sigmoid(head(cls_embedding))
        
        return predictions




