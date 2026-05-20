from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass
import warnings
from typing import Any, List, Dict, Optional
import asyncio
#!/usr/bin/env python3
"""
Advanced Attention Mechanisms and Positional Encodings
Production-ready attention mechanisms with proper GPU utilization and mixed precision training.
"""


warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


@dataclass
class AttentionConfig:
    """Configuration for attention mechanisms."""
    hidden_size: int = 768
    num_heads: int = 12
    head_dim: int = 64
    dropout: float = 0.1
    attention_dropout: float = 0.1
    max_position_embeddings: int = 512
    layer_norm_eps: float = 1e-12
    initializer_range: float = 0.02
    use_relative_position: bool = True
    relative_attention_num_buckets: int = 32
    relative_attention_max_distance: int = 128
    use_rope: bool = False  # Rotary Position Embedding
    rope_dim: int = 64
    use_alibi: bool = False  # Attention with Linear Biases
    alibi_max_positions: int = 2048


class SinusoidalPositionEmbedding(nn.Module):
    """Sinusoidal positional embedding as described in 'Attention Is All You Need'."""
    
    def __init__(self, hidden_size: int, max_position_embeddings: int = 512, dropout: float = 0.1):
        
    """__init__ function."""
super().__init__()
        self.hidden_size = hidden_size
        self.max_position_embeddings = max_position_embeddings
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_position_embeddings, hidden_size)
        position = torch.arange(0, max_position_embeddings, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_size, 2).float() * 
                           (-math.log(10000.0) / hidden_size))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input embeddings."""
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class LearnedPositionEmbedding(nn.Module):
    """Learned positional embedding."""
    
    def __init__(self, hidden_size: int, max_position_embeddings: int = 512, dropout: float = 0.1):
        
    """__init__ function."""
super().__init__()
        self.hidden_size = hidden_size
        self.max_position_embeddings = max_position_embeddings
        self.dropout = nn.Dropout(p=dropout)
        
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self._initialize_weights()
    
    def _initialize_weights(self) -> Any:
        """Initialize weights using proper techniques."""
        nn.init.normal_(self.position_embeddings.weight, mean=0.0, std=0.02)
    
    def forward(self, x: torch.Tensor, position_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Add learned positional embedding to input embeddings."""
        seq_length = x.size(1)
        
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=x.device)
            position_ids = position_ids.unsqueeze(0).expand(x.size(0), -1)
        
        position_embeddings = self.position_embeddings(position_ids)
        x = x + position_embeddings
        return self.dropout(x)


class RotaryPositionEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) for better position encoding."""
    
    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: int = 10000):
        
    """__init__ function."""
super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        # Generate rotation matrix
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
    
    def forward(self, x: torch.Tensor, seq_len: Optional[int] = None) -> torch.Tensor:
        """Apply rotary position embedding."""
        t = torch.arange(seq_len or x.size(1), device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        # Apply rotation
        cos = emb.cos()
        sin = emb.sin()
        
        # Reshape for broadcasting
        cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, dim)
        sin = sin.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, dim)
        
        # Apply rotation to input
        x_rot = torch.cat([-x[..., self.dim//2:], x[..., :self.dim//2]], dim=-1)
        x = x * cos + x_rot * sin
        
        return x


class ALiBiPositionEmbedding(nn.Module):
    """Attention with Linear Biases (ALiBi) for position encoding."""
    
    def __init__(self, num_heads: int, max_positions: int = 2048):
        
    """__init__ function."""
super().__init__()
        self.num_heads = num_heads
        self.max_positions = max_positions
        
        # Generate ALiBi slopes
        slopes = torch.Tensor(self._get_slopes(num_heads))
        self.register_buffer('slopes', slopes)
    
    def _get_slopes(self, num_heads: int) -> List[float]:
        """Generate slopes for ALiBi."""
        def get_slopes_power_of_2(n) -> Optional[Dict[str, Any]]:
            start = (2**(-2**-(math.log2(n)-3)))
            ratio = start
            return [start*ratio**i for i in range(n)]
        
        if math.log2(num_heads).is_integer():
            return get_slopes_power_of_2(num_heads)
        else:
            closest_power_of_2 = 2**math.floor(math.log2(num_heads))
            return (get_slopes_power_of_2(closest_power_of_2) + 
                   self._get_slopes(2*closest_power_of_2)[0::2][:num_heads-closest_power_of_2])
    
    def forward(self, attention_scores: torch.Tensor) -> torch.Tensor:
        """Add ALiBi bias to attention scores."""
        batch_size, num_heads, seq_len, seq_len = attention_scores.shape
        
        # Create position indices
        position_indices = torch.arange(seq_len, device=attention_scores.device)
        
        # Create ALiBi bias matrix
        alibi_bias = position_indices.unsqueeze(0) - position_indices.unsqueeze(1)
        alibi_bias = alibi_bias.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
        
        # Apply slopes
        slopes = self.slopes.unsqueeze(-1).unsqueeze(-1)  # (num_heads, 1, 1)
        alibi_bias = alibi_bias * slopes
        
        return attention_scores + alibi_bias


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism with proper implementation."""
    
    def __init__(self, config: AttentionConfig):
        
    """__init__ function."""
super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.all_head_size = self.num_heads * self.head_dim
        
        # Linear projections
        self.query_projection = nn.Linear(config.hidden_size, self.all_head_size)
        self.key_projection = nn.Linear(config.hidden_size, self.all_head_size)
        self.value_projection = nn.Linear(config.hidden_size, self.all_head_size)
        self.output_projection = nn.Linear(self.all_head_size, config.hidden_size)
        
        # Dropout layers
        self.attention_dropout = nn.Dropout(config.attention_dropout)
        self.output_dropout = nn.Dropout(config.dropout)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Positional encodings
        self.use_rope = config.use_rope
        self.use_alibi = config.use_alibi
        
        if self.use_rope:
            self.rope = RotaryPositionEmbedding(config.rope_dim)
        
        if self.use_alibi:
            self.alibi = ALiBiPositionEmbedding(config.num_heads, config.alibi_max_positions)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self) -> Any:
        """Initialize weights using proper techniques."""
        for module in [self.query_projection, self.key_projection, 
                      self.value_projection, self.output_projection]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0)
    
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        """Transpose tensor for multi-head attention computation."""
        new_x_shape = x.size()[:-1] + (self.num_heads, self.head_dim)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_length, head_dim)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with scaled dot-product attention."""
        batch_size, seq_length, hidden_size = query.shape
        
        # Linear projections
        query = self.query_projection(query)
        key = self.key_projection(key)
        value = self.value_projection(value)
        
        # Transpose for multi-head attention
        query = self.transpose_for_scores(query)  # (batch_size, num_heads, seq_length, head_dim)
        key = self.transpose_for_scores(key)
        value = self.transpose_for_scores(value)
        
        # Apply RoPE if enabled
        if self.use_rope:
            query = self.rope(query, seq_length)
            key = self.rope(key, seq_length)
        
        # Scaled dot-product attention
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply ALiBi if enabled
        if self.use_alibi:
            attention_scores = self.alibi(attention_scores)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        # Softmax and dropout
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.attention_dropout(attention_probs)
        
        # Apply attention to values
        context = torch.matmul(attention_probs, value)
        
        # Transpose back
        context = context.permute(0, 2, 1, 3).contiguous()
        new_context_shape = context.size()[:-2] + (self.all_head_size,)
        context = context.view(*new_context_shape)
        
        # Output projection
        output = self.output_projection(context)
        output = self.output_dropout(output)
        
        # Residual connection and layer normalization
        output = self.layer_norm(query.view(batch_size, seq_length, hidden_size) + output)
        
        return output


class RelativePositionEmbedding(nn.Module):
    """Relative position embedding for better position encoding."""
    
    def __init__(self, config: AttentionConfig):
        
    """__init__ function."""
super().__init__()
        self.config = config
        self.num_buckets = config.relative_attention_num_buckets
        self.max_distance = config.relative_attention_max_distance
        
        # Relative position embeddings
        self.relative_attention_bias = nn.Embedding(self.num_buckets, config.num_heads)
        self._initialize_weights()
    
    def _initialize_weights(self) -> Any:
        """Initialize weights using proper techniques."""
        nn.init.normal_(self.relative_attention_bias.weight, mean=0.0, std=0.02)
    
    def _relative_position_bucket(self, relative_position: torch.Tensor) -> torch.Tensor:
        """Convert relative position to bucket index."""
        relative_buckets = 0
        relative_position = torch.abs(relative_position)
        
        # Half of the buckets are for exact increments in positions
        max_exact = self.num_buckets // 2
        is_small = relative_position < max_exact
        
        # The other half of the buckets are for logarithmically bigger bins in positions
        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact) / 
            math.log(self.max_distance / max_exact) * (self.num_buckets - max_exact)
        ).to(torch.long)
        
        relative_position_if_large = torch.min(
            relative_position_if_large, torch.full_like(relative_position_if_large, self.num_buckets - 1)
        )
        
        relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
        return relative_buckets
    
    def forward(self, query_length: int, key_length: int) -> torch.Tensor:
        """Compute relative position embeddings."""
        context_position = torch.arange(query_length, dtype=torch.long)[:, None]
        memory_position = torch.arange(key_length, dtype=torch.long)[None, :]
        relative_position = memory_position - context_position
        
        relative_position_bucket = self._relative_position_bucket(relative_position)
        relative_position_bucket = relative_position_bucket.to(self.relative_attention_bias.weight.device)
        
        values = self.relative_attention_bias(relative_position_bucket)
        values = values.permute([2, 0, 1]).unsqueeze(0)  # (1, num_heads, query_length, key_length)
        
        return values


class CrossAttention(nn.Module):
    """Cross-attention mechanism for encoder-decoder architectures."""
    
    def __init__(self, config: AttentionConfig):
        
    """__init__ function."""
super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.all_head_size = self.num_heads * self.head_dim
        
        # Linear projections
        self.query_projection = nn.Linear(config.hidden_size, self.all_head_size)
        self.key_projection = nn.Linear(config.hidden_size, self.all_head_size)
        self.value_projection = nn.Linear(config.hidden_size, self.all_head_size)
        self.output_projection = nn.Linear(self.all_head_size, config.hidden_size)
        
        # Dropout layers
        self.attention_dropout = nn.Dropout(config.attention_dropout)
        self.output_dropout = nn.Dropout(config.dropout)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self) -> Any:
        """Initialize weights using proper techniques."""
        for module in [self.query_projection, self.key_projection, 
                      self.value_projection, self.output_projection]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0)
    
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        """Transpose tensor for multi-head attention computation."""
        new_x_shape = x.size()[:-1] + (self.num_heads, self.head_dim)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with cross-attention."""
        batch_size, query_length, hidden_size = query.shape
        
        # Linear projections
        query = self.query_projection(query)
        key = self.key_projection(key)
        value = self.value_projection(value)
        
        # Transpose for multi-head attention
        query = self.transpose_for_scores(query)
        key = self.transpose_for_scores(key)
        value = self.transpose_for_scores(value)
        
        # Scaled dot-product attention
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        # Softmax and dropout
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.attention_dropout(attention_probs)
        
        # Apply attention to values
        context = torch.matmul(attention_probs, value)
        
        # Transpose back
        context = context.permute(0, 2, 1, 3).contiguous()
        new_context_shape = context.size()[:-2] + (self.all_head_size,)
        context = context.view(*new_context_shape)
        
        # Output projection
        output = self.output_projection(context)
        output = self.output_dropout(output)
        
        # Residual connection and layer normalization
        output = self.layer_norm(query.view(batch_size, query_length, hidden_size) + output)
        
        return output


class AttentionWithPositionalEncoding(nn.Module):
    """Complete attention mechanism with positional encoding."""
    
    def __init__(self, config: AttentionConfig):
        
    """__init__ function."""
super().__init__()
        self.config = config
        
        # Positional encoding
        if config.use_rope:
            self.position_encoding = RotaryPositionEmbedding(config.rope_dim)
        elif config.use_alibi:
            self.position_encoding = ALiBiPositionEmbedding(config.num_heads, config.alibi_max_positions)
        else:
            self.position_encoding = SinusoidalPositionEmbedding(
                config.hidden_size, config.max_position_embeddings, config.dropout
            )
        
        # Attention mechanism
        self.attention = MultiHeadAttention(config)
        
        # Relative position embedding (optional)
        self.use_relative_position = config.use_relative_position
        if self.use_relative_position:
            self.relative_position_embedding = RelativePositionEmbedding(config)
    
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with attention and positional encoding."""
        batch_size, seq_length, hidden_size = x.shape
        
        # Apply positional encoding
        if isinstance(self.position_encoding, SinusoidalPositionEmbedding):
            x = x.transpose(0, 1)  # (seq_length, batch_size, hidden_size)
            x = self.position_encoding(x)
            x = x.transpose(0, 1)  # (batch_size, seq_length, hidden_size)
        elif isinstance(self.position_encoding, LearnedPositionEmbedding):
            x = self.position_encoding(x, position_ids)
        
        # Add relative position bias if enabled
        relative_position_bias = None
        if self.use_relative_position:
            relative_position_bias = self.relative_position_embedding(seq_length, seq_length)
        
        # Apply attention
        if relative_position_bias is not None:
            if attention_mask is None:
                attention_mask = relative_position_bias
            else:
                attention_mask = attention_mask + relative_position_bias
        
        output = self.attention(x, x, x, attention_mask, position_ids)
        
        return output


def create_attention_mechanism(hidden_size: int = 768, num_heads: int = 12,
                              use_rope: bool = False, use_alibi: bool = False) -> AttentionWithPositionalEncoding:
    """Create an attention mechanism with default configuration."""
    config = AttentionConfig(
        hidden_size=hidden_size,
        num_heads=num_heads,
        head_dim=hidden_size // num_heads,
        use_rope=use_rope,
        use_alibi=use_alibi
    )
    return AttentionWithPositionalEncoding(config)


# Example usage
if __name__ == "__main__":
    # Create attention mechanism
    attention_mechanism = create_attention_mechanism(768, 12, use_rope=True)
    
    # Sample input
    batch_size, seq_length, hidden_size = 2, 128, 768
    x = torch.randn(batch_size, seq_length, hidden_size)
    
    # Forward pass
    output = attention_mechanism(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Test different positional encodings
    print("\nTesting different positional encodings:")
    
    # Sinusoidal
    sinusoidal_pe = SinusoidalPositionEmbedding(768)
    print(f"Sinusoidal PE: {sinusoidal_pe(x.transpose(0, 1)).shape}")
    
    # Learned
    learned_pe = LearnedPositionEmbedding(768)
    print(f"Learned PE: {learned_pe(x).shape}")
    
    # RoPE
    rope_pe = RotaryPositionEmbedding(64)
    print(f"RoPE: {rope_pe(x, seq_length).shape}")
    
    # ALiBi
    alibi_pe = ALiBiPositionEmbedding(12)
    attention_scores = torch.randn(2, 12, 128, 128)
    print(f"ALiBi bias: {alibi_pe(attention_scores).shape}") 