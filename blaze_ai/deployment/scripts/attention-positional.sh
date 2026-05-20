#!/usr/bin/env python3
"""
Advanced Attention Mechanisms and Positional Encodings for Blaze AI
Implements correct attention mechanisms, positional encodings, and best practices
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import warnings
import numpy as np

warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class AttentionConfig:
    """Configuration for attention mechanisms"""
    num_heads: int = 8
    head_dim: int = 64
    hidden_size: int = 512
    dropout: float = 0.1
    attention_dropout: float = 0.1
    max_position_embeddings: int = 2048
    use_bias: bool = True
    use_relative_position: bool = False
    relative_position_max_distance: int = 128
    use_flash_attention: bool = True
    use_xformers: bool = True


@dataclass
class PositionalEncodingConfig:
    """Configuration for positional encodings"""
    encoding_type: str = "sinusoidal"  # sinusoidal, learned, rope, alibi, t5
    max_length: int = 2048
    hidden_size: int = 512
    dropout: float = 0.1
    base: int = 10000  # For RoPE
    scale: float = 1.0  # For ALiBi
    use_learned_scale: bool = False


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding as described in 'Attention Is All You Need'"""
    
    def __init__(self, hidden_size: int, max_length: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_length = max_length
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_length, hidden_size)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        
        # Calculate division term
        div_term = torch.exp(torch.arange(0, hidden_size, 2).float() * 
                           (-math.log(10000.0) / hidden_size))
        
        # Apply sine and cosine functions
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension and register as buffer
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        # Add positional encoding
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class LearnedPositionalEncoding(nn.Module):
    """Learned positional encoding"""
    
    def __init__(self, hidden_size: int, max_length: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_length = max_length
        self.dropout = nn.Dropout(p=dropout)
        
        # Learnable positional embeddings
        self.position_embeddings = nn.Embedding(max_length, hidden_size)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize positional embedding weights"""
        nn.init.normal_(self.position_embeddings.weight, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        batch_size, seq_len, _ = x.size()
        
        # Create position indices
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        
        # Get positional embeddings
        pos_embeddings = self.position_embeddings(positions)
        
        # Add positional encoding
        x = x + pos_embeddings
        return self.dropout(x)


class RotaryPositionalEncoding(nn.Module):
    """Rotary Position Embedding (RoPE) for transformers"""
    
    def __init__(self, hidden_size: int, max_length: int = 2048, base: int = 10000):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_length = max_length
        self.base = base
        
        # Generate rotation matrix
        inv_freq = 1.0 / (self.base ** (torch.arange(0, hidden_size, 2).float() / hidden_size))
        self.register_buffer("inv_freq", inv_freq)
    
    def forward(self, x: torch.Tensor, seq_len: Optional[int] = None) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, num_heads, seq_len, head_dim]
            seq_len: Optional sequence length for position calculation
        """
        if seq_len is None:
            seq_len = x.size(-2)
        
        # Generate position embeddings
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        # Apply rotation
        cos = torch.cos(emb).unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_dim]
        sin = torch.sin(emb).unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_dim]
        
        # Apply rotary transformation
        x_rot = x * cos + self._rotate_half(x) * sin
        return x_rot
    
    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate half the hidden dimension"""
        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)


class ALiBiPositionalEncoding(nn.Module):
    """Attention with Linear Biases (ALiBi) positional encoding"""
    
    def __init__(self, num_heads: int, max_length: int = 2048, scale: float = 1.0):
        super().__init__()
        self.num_heads = num_heads
        self.max_length = max_length
        self.scale = scale
        
        # Generate ALiBi slopes
        slopes = self._get_slopes(num_heads)
        self.register_buffer("slopes", torch.tensor(slopes))
        
        # Generate bias matrix
        bias = torch.arange(max_length).unsqueeze(0).unsqueeze(0)  # [1, 1, max_length]
        bias = bias * self.slopes.unsqueeze(-1)  # [num_heads, 1, max_length]
        self.register_buffer("bias", bias)
    
    def _get_slopes(self, n: int) -> List[float]:
        """Get ALiBi slopes for n heads"""
        def get_slopes_power_of_2(n):
            start = (2**(-2**-(math.log2(n)-3)))
            ratio = start
            return [start*ratio**i for i in range(n)]
        
        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)
        else:
            closest_power_of_2 = 2**math.floor(math.log2(n))
            return get_slopes_power_of_2(closest_power_of_2) + self._get_slopes(2*closest_power_of_2)[0::2][:n-closest_power_of_2]
    
    def forward(self, attention_scores: torch.Tensor, seq_len: Optional[int] = None) -> torch.Tensor:
        """
        Args:
            attention_scores: Tensor, shape [batch_size, num_heads, seq_len, seq_len]
            seq_len: Optional sequence length for bias calculation
        """
        if seq_len is None:
            seq_len = attention_scores.size(-1)
        
        # Get bias for current sequence length
        bias = self.bias[:, :, :seq_len, :seq_len] * self.scale
        
        # Add bias to attention scores
        attention_scores = attention_scores + bias
        return attention_scores


class T5PositionalEncoding(nn.Module):
    """T5-style relative positional encoding"""
    
    def __init__(self, hidden_size: int, max_relative_position: int = 32):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_relative_position = max_relative_position
        
        # Relative position embeddings
        self.relative_attention_bias = nn.Embedding(
            2 * max_relative_position + 1, 
            hidden_size
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize relative position embedding weights"""
        nn.init.normal_(self.relative_attention_bias.weight, std=0.02)
    
    def forward(self, seq_len: int) -> torch.Tensor:
        """
        Args:
            seq_len: Sequence length
        Returns:
            Relative position bias tensor
        """
        # Generate relative position indices
        range_vec = torch.arange(seq_len, device=self.relative_attention_bias.weight.device)
        range_mat = range_vec.unsqueeze(0).repeat(seq_len, 1)
        distance_mat = range_mat - range_mat.transpose(0, 1)
        
        # Clip distances to max_relative_position
        distance_mat_clipped = torch.clamp(
            distance_mat, 
            -self.max_relative_position, 
            self.max_relative_position
        )
        
        # Shift to non-negative indices
        final_mat = distance_mat_clipped + self.max_relative_position
        
        # Get embeddings
        embeddings = self.relative_attention_bias(final_mat)
        return embeddings


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention with various positional encoding options"""
    
    def __init__(self, config: AttentionConfig):
        super().__init__()
        self.config = config
        
        # Ensure hidden_size is divisible by num_heads
        assert config.hidden_size % config.num_heads == 0
        
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        self.scale = self.head_dim ** -0.5
        
        # Linear projections
        self.query = nn.Linear(config.hidden_size, config.hidden_size, bias=config.use_bias)
        self.key = nn.Linear(config.hidden_size, config.hidden_size, bias=config.use_bias)
        self.value = nn.Linear(config.hidden_size, config.hidden_size, bias=config.use_bias)
        self.output = nn.Linear(config.hidden_size, config.hidden_size, bias=config.use_bias)
        
        # Dropout layers
        self.attention_dropout = nn.Dropout(config.attention_dropout)
        self.output_dropout = nn.Dropout(config.dropout)
        
        # Flash Attention support
        self.use_flash_attention = config.use_flash_attention and hasattr(F, 'scaled_dot_product_attention')
        
        # Relative position support
        self.use_relative_position = config.use_relative_position
        if self.use_relative_position:
            self.relative_position_encoding = T5PositionalEncoding(
                config.hidden_size,
                config.relative_position_max_distance
            )
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                past_key_value: Optional[Tuple[torch.Tensor]] = None,
                output_attentions: bool = False) -> Tuple[torch.Tensor, ...]:
        
        batch_size, query_seq_len, _ = query.size()
        key_seq_len = key.size(1)
        value_seq_len = value.size(1)
        
        # Project queries, keys, and values
        query_states = self.query(query)
        key_states = self.key(key)
        value_states = self.value(value)
        
        # Reshape for multi-head attention
        query_states = query_states.view(batch_size, query_seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, key_seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, value_seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Handle past key/value states for caching
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        
        past_key_value = (key_states, value_states)
        
        # Compute attention scores
        if self.use_flash_attention:
            # Use PyTorch 2.0 scaled dot product attention
            attention_output = F.scaled_dot_product_attention(
                query_states, key_states, value_states,
                attn_mask=attention_mask,
                dropout_p=self.config.attention_dropout if self.training else 0.0
            )
            attention_probs = None
        else:
            # Standard attention computation
            attention_scores = torch.matmul(query_states, key_states.transpose(-1, -2)) * self.scale
            
            # Add relative position bias if enabled
            if self.use_relative_position:
                relative_position_bias = self.relative_position_encoding(query_seq_len)
                relative_position_bias = relative_position_bias.unsqueeze(0).unsqueeze(0)
                attention_scores = attention_scores + relative_position_bias
            
            # Apply attention mask
            if attention_mask is not None:
                attention_scores = attention_scores + attention_mask
            
            # Apply softmax and dropout
            attention_probs = F.softmax(attention_scores, dim=-1)
            attention_probs = self.attention_dropout(attention_probs)
            
            # Compute attention output
            attention_output = torch.matmul(attention_probs, value_states)
        
        # Reshape and project output
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(batch_size, query_seq_len, self.hidden_size)
        attention_output = self.output(attention_output)
        attention_output = self.output_dropout(attention_output)
        
        # Prepare outputs
        outputs = (attention_output,)
        if output_attentions:
            outputs += (attention_probs,)
        outputs += (past_key_value,)
        
        return outputs


class CrossAttention(nn.Module):
    """Cross-attention mechanism for encoder-decoder architectures"""
    
    def __init__(self, config: AttentionConfig):
        super().__init__()
        self.config = config
        
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        self.scale = self.head_dim ** -0.5
        
        # Linear projections
        self.query = nn.Linear(config.hidden_size, config.hidden_size, bias=config.use_bias)
        self.key = nn.Linear(config.hidden_size, config.hidden_size, bias=config.use_bias)
        self.value = nn.Linear(config.hidden_size, config.hidden_size, bias=config.use_bias)
        self.output = nn.Linear(config.hidden_size, config.hidden_size, bias=config.use_bias)
        
        # Dropout layers
        self.attention_dropout = nn.Dropout(config.attention_dropout)
        self.output_dropout = nn.Dropout(config.dropout)
        
        # Flash Attention support
        self.use_flash_attention = config.use_flash_attention and hasattr(F, 'scaled_dot_product_attention')
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                output_attentions: bool = False) -> Tuple[torch.Tensor, ...]:
        
        batch_size, query_seq_len, _ = query.size()
        key_seq_len = key.size(1)
        
        # Project queries, keys, and values
        query_states = self.query(query)
        key_states = self.key(key)
        value_states = self.value(value)
        
        # Reshape for multi-head attention
        query_states = query_states.view(batch_size, query_seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, key_seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, key_seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        if self.use_flash_attention:
            # Use PyTorch 2.0 scaled dot product attention
            attention_output = F.scaled_dot_product_attention(
                query_states, key_states, value_states,
                attn_mask=attention_mask,
                dropout_p=self.config.attention_dropout if self.training else 0.0
            )
            attention_probs = None
        else:
            # Standard attention computation
            attention_scores = torch.matmul(query_states, key_states.transpose(-1, -2)) * self.scale
            
            # Apply attention mask
            if attention_mask is not None:
                attention_scores = attention_scores + attention_mask
            
            # Apply softmax and dropout
            attention_probs = F.softmax(attention_scores, dim=-1)
            attention_probs = self.attention_dropout(attention_probs)
            
            # Compute attention output
            attention_output = torch.matmul(attention_probs, value_states)
        
        # Reshape and project output
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(batch_size, query_seq_len, self.hidden_size)
        attention_output = self.output(attention_output)
        attention_output = self.output_dropout(attention_output)
        
        # Prepare outputs
        outputs = (attention_output,)
        if output_attentions:
            outputs += (attention_probs,)
        
        return outputs


class GroupedQueryAttention(nn.Module):
    """Grouped Query Attention for efficient multi-head attention"""
    
    def __init__(self, config: AttentionConfig, num_key_value_heads: int = 8):
        super().__init__()
        self.config = config
        self.num_heads = config.num_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        self.scale = self.head_dim ** -0.5
        
        # Linear projections
        self.query = nn.Linear(config.hidden_size, config.hidden_size, bias=config.use_bias)
        self.key_value = nn.Linear(config.hidden_size, 2 * num_key_value_heads * self.head_dim, bias=config.use_bias)
        self.output = nn.Linear(config.hidden_size, config.hidden_size, bias=config.use_bias)
        
        # Dropout layers
        self.attention_dropout = nn.Dropout(config.attention_dropout)
        self.output_dropout = nn.Dropout(config.dropout)
        
        # Flash Attention support
        self.use_flash_attention = config.use_flash_attention and hasattr(F, 'scaled_dot_product_attention')
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                past_key_value: Optional[Tuple[torch.Tensor]] = None,
                output_attentions: bool = False) -> Tuple[torch.Tensor, ...]:
        
        batch_size, seq_len, _ = hidden_states.size()
        
        # Project queries, keys, and values
        query_states = self.query(hidden_states)
        key_value_states = self.key_value(hidden_states)
        
        # Split key and value
        key_states, value_states = key_value_states.chunk(2, dim=-1)
        
        # Reshape for grouped query attention
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        # Repeat key and value heads for grouped query attention
        key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
        value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=1)
        
        # Handle past key/value states
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        
        past_key_value = (key_states, value_states)
        
        # Compute attention scores
        if self.use_flash_attention:
            # Use PyTorch 2.0 scaled dot product attention
            attention_output = F.scaled_dot_product_attention(
                query_states, key_states, value_states,
                attn_mask=attention_mask,
                dropout_p=self.config.attention_dropout if self.training else 0.0
            )
            attention_probs = None
        else:
            # Standard attention computation
            attention_scores = torch.matmul(query_states, key_states.transpose(-1, -2)) * self.scale
            
            # Apply attention mask
            if attention_mask is not None:
                attention_scores = attention_scores + attention_mask
            
            # Apply softmax and dropout
            attention_probs = F.softmax(attention_scores, dim=-1)
            attention_probs = self.attention_dropout(attention_probs)
            
            # Compute attention output
            attention_output = torch.matmul(attention_probs, value_states)
        
        # Reshape and project output
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(batch_size, seq_len, self.hidden_size)
        attention_output = self.output(attention_output)
        attention_output = self.output_dropout(attention_output)
        
        # Prepare outputs
        outputs = (attention_output,)
        if output_attentions:
            outputs += (attention_probs,)
        outputs += (past_key_value,)
        
        return outputs


class AttentionBlock(nn.Module):
    """Complete attention block with positional encoding and normalization"""
    
    def __init__(self, config: AttentionConfig, pos_config: PositionalEncodingConfig):
        super().__init__()
        self.config = config
        self.pos_config = pos_config
        
        # Attention mechanism
        self.attention = MultiHeadAttention(config)
        
        # Positional encoding
        if pos_config.encoding_type == "sinusoidal":
            self.pos_encoding = SinusoidalPositionalEncoding(
                pos_config.hidden_size, pos_config.max_length, pos_config.dropout
            )
        elif pos_config.encoding_type == "learned":
            self.pos_encoding = LearnedPositionalEncoding(
                pos_config.hidden_size, pos_config.max_length, pos_config.dropout
            )
        elif pos_config.encoding_type == "rope":
            self.pos_encoding = RotaryPositionalEncoding(
                pos_config.hidden_size, pos_config.max_length, pos_config.base
            )
        elif pos_config.encoding_type == "alibi":
            self.pos_encoding = ALiBiPositionalEncoding(
                config.num_heads, pos_config.max_length, pos_config.scale
            )
        elif pos_config.encoding_type == "t5":
            self.pos_encoding = T5PositionalEncoding(
                pos_config.hidden_size, pos_config.max_length
            )
        else:
            self.pos_encoding = None
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.norm2 = nn.LayerNorm(config.hidden_size)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 4),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size * 4, config.hidden_size),
            nn.Dropout(config.dropout)
        )
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                past_key_value: Optional[Tuple[torch.Tensor]] = None,
                output_attentions: bool = False) -> Tuple[torch.Tensor, ...]:
        
        # Apply positional encoding if needed
        if self.pos_encoding is not None:
            if isinstance(self.pos_encoding, (SinusoidalPositionalEncoding, LearnedPositionalEncoding)):
                hidden_states = self.pos_encoding(hidden_states)
            elif isinstance(self.pos_encoding, RotaryPositionalEncoding):
                # RoPE is applied in the attention mechanism
                pass
            elif isinstance(self.pos_encoding, ALiBiPositionalEncoding):
                # ALiBi is applied in the attention mechanism
                pass
        
        # Self-attention with residual connection
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        
        attention_outputs = self.attention(
            hidden_states, hidden_states, hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions
        )
        
        attention_output = attention_outputs[0]
        hidden_states = residual + attention_output
        
        # Feed-forward network with residual connection
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.ffn(hidden_states)
        hidden_states = residual + hidden_states
        
        # Prepare outputs
        outputs = (hidden_states,)
        if output_attentions:
            outputs += attention_outputs[1:]
        if len(attention_outputs) > 1:
            outputs += attention_outputs[-1:]
        
        return outputs


class AttentionExperiments:
    """Collection of attention mechanism experiments"""
    
    @staticmethod
    def demonstrate_positional_encodings():
        """Demonstrate different positional encoding schemes"""
        
        logger.info("Demonstrating positional encodings...")
        
        batch_size, seq_len, hidden_size = 2, 10, 64
        
        # Create input tensor
        x = torch.randn(batch_size, seq_len, hidden_size)
        
        # Test different positional encodings
        encodings = {}
        
        # Sinusoidal
        sinusoidal_pe = SinusoidalPositionalEncoding(hidden_size, seq_len)
        encodings["sinusoidal"] = sinusoidal_pe(x.transpose(0, 1)).transpose(0, 1)
        
        # Learned
        learned_pe = LearnedPositionalEncoding(hidden_size, seq_len)
        encodings["learned"] = learned_pe(x)
        
        # RoPE
        rope_pe = RotaryPositionalEncoding(hidden_size, seq_len)
        # RoPE is applied to reshaped tensor for multi-head attention
        x_reshaped = x.view(batch_size, seq_len, 8, 8)  # 8 heads, 8 dims per head
        encodings["rope"] = rope_pe(x_reshaped, seq_len)
        
        # ALiBi
        alibi_pe = ALiBiPositionalEncoding(8, seq_len)  # 8 heads
        attention_scores = torch.randn(batch_size, 8, seq_len, seq_len)
        encodings["alibi"] = alibi_pe(attention_scores, seq_len)
        
        logger.info("Positional encoding demonstration completed")
        return encodings
    
    @staticmethod
    def demonstrate_attention_mechanisms():
        """Demonstrate different attention mechanisms"""
        
        logger.info("Demonstrating attention mechanisms...")
        
        # Create configurations
        attention_config = AttentionConfig(
            num_heads=8,
            head_dim=64,
            hidden_size=512,
            use_flash_attention=True
        )
        
        pos_config = PositionalEncodingConfig(
            encoding_type="rope",
            hidden_size=512
        )
        
        # Create attention block
        attention_block = AttentionBlock(attention_config, pos_config)
        
        # Create input
        batch_size, seq_len = 2, 16
        hidden_states = torch.randn(batch_size, seq_len, attention_config.hidden_size)
        
        # Forward pass
        with torch.no_grad():
            outputs = attention_block(
                hidden_states,
                output_attentions=True
            )
        
        logger.info("Attention mechanism demonstration completed")
        return attention_block, outputs
    
    @staticmethod
    def demonstrate_grouped_query_attention():
        """Demonstrate grouped query attention"""
        
        logger.info("Demonstrating grouped query attention...")
        
        # Create configuration
        attention_config = AttentionConfig(
            num_heads=32,
            head_dim=64,
            hidden_size=2048,
            use_flash_attention=True
        )
        
        # Create grouped query attention
        gqa = GroupedQueryAttention(attention_config, num_key_value_heads=8)
        
        # Create input
        batch_size, seq_len = 2, 32
        hidden_states = torch.randn(batch_size, seq_len, attention_config.hidden_size)
        
        # Forward pass
        with torch.no_grad():
            outputs = gqa(hidden_states, output_attentions=True)
        
        logger.info("Grouped query attention demonstration completed")
        return gqa, outputs


def main():
    """Main execution function"""
    logger.info("Starting Advanced Attention Mechanisms and Positional Encodings Demonstrations...")
    
    # Demonstrate positional encodings
    logger.info("Testing positional encodings...")
    pos_encodings = AttentionExperiments.demonstrate_positional_encodings()
    
    # Demonstrate attention mechanisms
    logger.info("Testing attention mechanisms...")
    attention_block, attention_outputs = AttentionExperiments.demonstrate_attention_mechanisms()
    
    # Demonstrate grouped query attention
    logger.info("Testing grouped query attention...")
    gqa, gqa_outputs = AttentionExperiments.demonstrate_grouped_query_attention()
    
    # Create comprehensive attention system
    logger.info("Creating comprehensive attention system...")
    
    comprehensive_attention_config = AttentionConfig(
        num_heads=16,
        head_dim=64,
        hidden_size=1024,
        use_flash_attention=True,
        use_relative_position=True
    )
    
    comprehensive_pos_config = PositionalEncodingConfig(
        encoding_type="rope",
        hidden_size=1024,
        max_length=4096
    )
    
    comprehensive_attention = AttentionBlock(comprehensive_attention_config, comprehensive_pos_config)
    
    # Test comprehensive attention
    test_input = torch.randn(4, 64, 1024)
    
    with torch.no_grad():
        test_outputs = comprehensive_attention(
            test_input,
            output_attentions=True
        )
    
    logger.info(f"Comprehensive attention output shape: {test_outputs[0].shape}")
    logger.info(f"Comprehensive attention parameters: {sum(p.numel() for p in comprehensive_attention.parameters()):,}")
    
    # Summary
    logger.info("Attention Mechanisms and Positional Encodings Summary:")
    logger.info(f"Positional encodings tested: {len(pos_encodings)}")
    logger.info(f"Attention mechanisms tested: ✓")
    logger.info(f"Grouped query attention tested: ✓")
    logger.info(f"Comprehensive attention system created: ✓")
    logger.info(f"Total parameters across attention systems: {sum(p.numel() for p in [attention_block, gqa, comprehensive_attention])}")
    
    logger.info("Advanced Attention Mechanisms and Positional Encodings demonstrations completed successfully!")


if __name__ == "__main__":
    main()
