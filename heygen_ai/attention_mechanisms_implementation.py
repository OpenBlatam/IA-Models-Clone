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
from typing import Optional, Tuple, Union, List
from dataclasses import dataclass
import logging
from typing import Any, List, Dict, Optional
import asyncio
#!/usr/bin/env python3
"""
Attention Mechanisms and Positional Encodings Implementation
Comprehensive implementation of attention mechanisms and positional encodings for transformers.
"""


logger = logging.getLogger(__name__)

@dataclass
class AttentionConfig:
    """Configuration for attention mechanisms."""
    hidden_size: int = 768
    num_attention_heads: int = 12
    attention_dropout: float = 0.1
    max_position_embeddings: int = 512
    position_embedding_type: str = "absolute"  # absolute, relative, rotary
    use_relative_position: bool = False
    relative_attention_num_buckets: int = 32
    relative_attention_max_distance: int = 128
    rotary_dim: int = 64
    use_scale_attention_weights: bool = True
    attention_softmax_in_fp32: bool = False

class PositionalEncoding(nn.Module):
    """Absolute positional encoding for transformer models."""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        
    """__init__ function."""
super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input embeddings.
        
        Args:
            x: Input tensor of shape (seq_len, batch_size, d_model)
            
        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class RelativePositionalEncoding(nn.Module):
    """Relative positional encoding for transformer models."""
    
    def __init__(self, d_model: int, max_relative_position: int = 32):
        
    """__init__ function."""
super().__init__()
        self.d_model = d_model
        self.max_relative_position = max_relative_position
        
        # Create relative position embeddings
        self.relative_position_embeddings = nn.Embedding(
            2 * max_relative_position + 1, d_model
        )
        
        # Initialize embeddings
        nn.init.normal_(self.relative_position_embeddings.weight, std=0.02)
    
    def forward(self, length: int, device: torch.device) -> torch.Tensor:
        """
        Generate relative position embeddings.
        
        Args:
            length: Sequence length
            device: Device to place embeddings on
            
        Returns:
            Relative position embeddings
        """
        range_vec = torch.arange(length, device=device)
        range_mat = range_vec.unsqueeze(0).repeat(length, 1)
        distance_mat = range_mat - range_mat.transpose(0, 1)
        
        # Clip distances to max_relative_position
        distance_mat_clipped = torch.clamp(
            distance_mat, 
            -self.max_relative_position, 
            self.max_relative_position
        )
        
        # Shift to non-negative indices
        final_mat = distance_mat_clipped + self.max_relative_position
        
        return self.relative_position_embeddings(final_mat)

class RotaryPositionalEncoding(nn.Module):
    """Rotary positional encoding (RoPE) for transformer models."""
    
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
        """
        Apply rotary positional encoding to input.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, num_heads, head_dim)
            seq_len: Sequence length (optional)
            
        Returns:
            Tensor with rotary positional encoding applied
        """
        if seq_len is None:
            seq_len = x.size(1)
        
        # Generate position indices
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        # Apply rotation
        cos = emb.cos()
        sin = emb.sin()
        
        # Reshape for broadcasting
        cos = cos.view(1, seq_len, 1, -1)
        sin = sin.view(1, seq_len, 1, -1)
        
        # Apply rotation to input
        x_rot = torch.cat([-x[..., self.dim//2:], x[..., :self.dim//2]], dim=-1)
        return x * cos + x_rot * sin

class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism with various positional encoding options."""
    
    def __init__(self, config: AttentionConfig):
        
    """__init__ function."""
super().__init__()
        self.config = config
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        # Linear projections
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.output = nn.Linear(config.hidden_size, config.hidden_size)
        
        # Dropout
        self.attention_dropout = nn.Dropout(config.attention_dropout)
        self.output_dropout = nn.Dropout(config.attention_dropout)
        
        # Positional encodings
        self.position_embedding_type = config.position_embedding_type
        if config.position_embedding_type == "absolute":
            self.position_embeddings = nn.Embedding(
                config.max_position_embeddings, config.hidden_size
            )
        elif config.position_embedding_type == "relative":
            self.relative_position_embeddings = RelativePositionalEncoding(
                config.hidden_size, config.relative_attention_max_distance
            )
        elif config.position_embedding_type == "rotary":
            self.rotary_position_embeddings = RotaryPositionalEncoding(
                config.rotary_dim, config.max_position_embeddings
            )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self) -> Any:
        """Initialize attention weights."""
        nn.init.normal_(self.query.weight, std=0.02)
        nn.init.normal_(self.key.weight, std=0.02)
        nn.init.normal_(self.value.weight, std=0.02)
        nn.init.normal_(self.output.weight, std=0.02)
        nn.init.zeros_(self.query.bias)
        nn.init.zeros_(self.key.bias)
        nn.init.zeros_(self.value.bias)
        nn.init.zeros_(self.output.bias)
    
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        """Transpose tensor for multi-head attention."""
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass of multi-head attention.
        
        Args:
            hidden_states: Input hidden states
            attention_mask: Attention mask
            head_mask: Head mask for pruning
            encoder_hidden_states: Encoder hidden states for cross-attention
            encoder_attention_mask: Encoder attention mask
            past_key_value: Past key-value states for caching
            output_attentions: Whether to output attention weights
            position_ids: Position IDs for positional encoding
            
        Returns:
            Tuple of (output, attention_weights, past_key_value)
        """
        # Apply positional encoding
        if self.position_embedding_type == "absolute" and position_ids is not None:
            position_embeddings = self.position_embeddings(position_ids)
            hidden_states = hidden_states + position_embeddings
        
        # Linear transformations
        mixed_query_layer = self.query(hidden_states)
        
        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None
        
        if is_cross_attention:
            mixed_key_layer = self.key(encoder_hidden_states)
            mixed_value_layer = self.value(encoder_hidden_states)
            attention_mask = encoder_attention_mask
        else:
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)
        
        # Transpose for multi-head attention
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        
        # Apply rotary positional encoding if specified
        if self.position_embedding_type == "rotary":
            query_layer = self.rotary_position_embeddings(query_layer)
            key_layer = self.rotary_position_embeddings(key_layer)
        
        # Take the dot product between "query" and "key" to get the raw attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        
        # Scale attention scores
        if self.config.use_scale_attention_weights:
            attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        # Apply attention mask
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        # Normalize the attention scores to probabilities
        if self.config.attention_softmax_in_fp32:
            attention_scores = attention_scores.float()
        
        attention_probs = F.softmax(attention_scores, dim=-1)
        
        # Apply dropout
        attention_probs = self.attention_dropout(attention_probs)
        
        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask
        
        # Compute attention output
        context_layer = torch.matmul(attention_probs, value_layer)
        
        # Transpose back
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        
        # Apply output projection
        attention_output = self.output(context_layer)
        attention_output = self.output_dropout(attention_output)
        
        # Apply layer normalization
        attention_output = self.layer_norm(attention_output + hidden_states)
        
        outputs = (attention_output,)
        if output_attentions:
            outputs += (attention_probs,)
        
        return outputs

class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention mechanism."""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        
    """__init__ function."""
super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.dropout = nn.Dropout(dropout)
        
        # Linear projections
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.scale = math.sqrt(self.d_k)
    
    def forward(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of scaled dot-product attention.
        
        Args:
            query: Query tensor
            key: Key tensor
            value: Value tensor
            mask: Attention mask
            
        Returns:
            Tuple of (output, attention_weights)
        """
        batch_size = query.size(0)
        
        # Linear transformations and reshape
        Q = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        
        # Reshape and apply output projection
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        output = self.w_o(context)
        
        return output, attention_weights

class SelfAttention(nn.Module):
    """Self-attention mechanism with residual connection and layer normalization."""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        
    """__init__ function."""
super().__init__()
        self.attention = ScaledDotProductAttention(d_model, num_heads, dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of self-attention.
        
        Args:
            x: Input tensor
            mask: Attention mask
            
        Returns:
            Tuple of (output, attention_weights)
        """
        residual = x
        
        # Apply attention
        output, attention_weights = self.attention(x, x, x, mask)
        
        # Apply dropout and residual connection
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        
        return output, attention_weights

class CrossAttention(nn.Module):
    """Cross-attention mechanism for encoder-decoder architectures."""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        
    """__init__ function."""
super().__init__()
        self.attention = ScaledDotProductAttention(d_model, num_heads, dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of cross-attention.
        
        Args:
            query: Query tensor (from decoder)
            key: Key tensor (from encoder)
            value: Value tensor (from encoder)
            mask: Attention mask
            
        Returns:
            Tuple of (output, attention_weights)
        """
        residual = query
        
        # Apply attention
        output, attention_weights = self.attention(query, key, value, mask)
        
        # Apply dropout and residual connection
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        
        return output, attention_weights

class AttentionBlock(nn.Module):
    """Complete attention block with feed-forward network."""
    
    def __init__(self, config: AttentionConfig):
        
    """__init__ function."""
super().__init__()
        self.attention = MultiHeadAttention(config)
        self.intermediate = nn.Linear(config.hidden_size, config.hidden_size * 4)
        self.output = nn.Linear(config.hidden_size * 4, config.hidden_size)
        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.attention_dropout)
        
        # Activation function
        self.activation = nn.GELU()
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass of attention block.
        
        Args:
            hidden_states: Input hidden states
            attention_mask: Attention mask
            head_mask: Head mask for pruning
            encoder_hidden_states: Encoder hidden states for cross-attention
            encoder_attention_mask: Encoder attention mask
            past_key_value: Past key-value states for caching
            output_attentions: Whether to output attention weights
            position_ids: Position IDs for positional encoding
            
        Returns:
            Tuple of (output, attention_weights, past_key_value)
        """
        # Self-attention
        attention_outputs = self.attention(
            self.layernorm_before(hidden_states),
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
            position_ids,
        )
        attention_output = attention_outputs[0]
        
        # Residual connection
        hidden_states = hidden_states + attention_output
        
        # Feed-forward network
        ff_output = self.layernorm_after(hidden_states)
        ff_output = self.intermediate(ff_output)
        ff_output = self.activation(ff_output)
        ff_output = self.output(ff_output)
        ff_output = self.dropout(ff_output)
        
        # Residual connection
        hidden_states = hidden_states + ff_output
        
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attention_outputs[1],)
        
        return outputs

# Example usage and demonstration
def demonstrate_attention_mechanisms():
    """Demonstrate various attention mechanisms."""
    
    # Configuration
    config = AttentionConfig(
        hidden_size=768,
        num_attention_heads=12,
        attention_dropout=0.1,
        max_position_embeddings=512,
        position_embedding_type="absolute"
    )
    
    # Create attention block
    attention_block = AttentionBlock(config)
    
    # Create sample input
    batch_size = 2
    seq_len = 128
    hidden_size = config.hidden_size
    
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    attention_mask = torch.ones(batch_size, seq_len)
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
    
    print(f"Input shape: {hidden_states.shape}")
    
    # Forward pass
    with torch.no_grad():
        outputs = attention_block(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=True
        )
    
    print(f"Output shape: {outputs[0].shape}")
    if len(outputs) > 1:
        print(f"Attention weights shape: {outputs[1].shape}")
    
    # Test different positional encodings
    print("\nTesting different positional encodings:")
    
    # Absolute positional encoding
    abs_pe = PositionalEncoding(hidden_size, max_len=512)
    abs_output = abs_pe(hidden_states.transpose(0, 1)).transpose(0, 1)
    print(f"Absolute PE output shape: {abs_output.shape}")
    
    # Relative positional encoding
    rel_pe = RelativePositionalEncoding(hidden_size, max_relative_position=32)
    rel_embeddings = rel_pe(seq_len, hidden_states.device)
    print(f"Relative PE embeddings shape: {rel_embeddings.shape}")
    
    # Rotary positional encoding
    rotary_pe = RotaryPositionalEncoding(dim=64, max_position_embeddings=512)
    rotary_input = torch.randn(batch_size, seq_len, config.num_attention_heads, 64)
    rotary_output = rotary_pe(rotary_input)
    print(f"Rotary PE output shape: {rotary_output.shape}")

if __name__ == "__main__":
    # Set seed for reproducibility
    torch.manual_seed(42)
    
    # Demonstrate attention mechanisms
    demonstrate_attention_mechanisms() 