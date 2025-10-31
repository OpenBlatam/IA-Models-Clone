"""
Advanced Attention Mechanisms and Positional Encodings System
Implements correct attention mechanisms and positional encodings for transformer models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Optional, Tuple, Union
from dataclasses import dataclass


@dataclass
class AttentionConfig:
    """Configuration for attention mechanisms."""
    d_model: int = 512
    n_heads: int = 8
    d_k: int = 64
    d_v: int = 64
    dropout: float = 0.1
    max_seq_length: int = 512
    use_relative_position: bool = False
    use_rotary_position: bool = False
    use_alibi_position: bool = False
    attention_type: str = "scaled_dot_product"  # scaled_dot_product, linear, local, sparse


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding as described in 'Attention Is All You Need'."""
    
    def __init__(self, d_model: int, max_seq_length: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input embeddings."""
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class LearnedPositionalEncoding(nn.Module):
    """Learned positional encoding that can adapt to different sequence lengths."""
    
    def __init__(self, d_model: int, max_seq_length: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Parameter(torch.randn(max_seq_length, d_model))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add learned positional encoding to input embeddings."""
        seq_len = x.size(1)
        if seq_len <= self.pe.size(0):
            x = x + self.pe[:seq_len, :]
        else:
            # Extend positional encoding for longer sequences
            extended_pe = F.interpolate(
                self.pe.unsqueeze(0).transpose(1, 2),
                size=seq_len,
                mode='linear',
                align_corners=False
            ).squeeze(0).transpose(0, 1)
            x = x + extended_pe
        
        return self.dropout(x)


class RelativePositionalEncoding(nn.Module):
    """Relative positional encoding for transformer models."""
    
    def __init__(self, d_model: int, max_relative_position: int = 32):
        super().__init__()
        self.d_model = d_model
        self.max_relative_position = max_relative_position
        
        # Relative position embeddings
        self.relative_attention_bias = nn.Embedding(
            2 * max_relative_position + 1, 
            n_heads
        )
        
    def forward(self, query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        """Compute relative positional encoding for attention."""
        batch_size, n_heads, seq_len, d_k = query.size()
        
        # Compute relative positions
        positions = torch.arange(seq_len, device=query.device)
        relative_positions = positions.unsqueeze(1) - positions.unsqueeze(0)
        relative_positions = torch.clamp(
            relative_positions, 
            -self.max_relative_position, 
            self.max_relative_position
        )
        relative_positions += self.max_relative_position
        
        # Get relative attention bias
        relative_attention_bias = self.relative_attention_bias(relative_positions)
        relative_attention_bias = relative_attention_bias.unsqueeze(0).unsqueeze(0)
        
        return relative_attention_bias


class RotaryPositionalEncoding(nn.Module):
    """Rotary positional encoding (RoPE) for transformer models."""
    
    def __init__(self, d_model: int, max_seq_length: int = 512):
        super().__init__()
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        
        # Generate rotation matrices
        self.register_buffer('cos_cached', self._get_cos_sin_cache()[0])
        self.register_buffer('sin_cached', self._get_cos_sin_cache()[1])
    
    def _get_cos_sin_cache(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate cached cos and sin values for rotary encoding."""
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.d_model, 2).float() / self.d_model))
        t = torch.arange(self.max_seq_length, device=inv_freq.device).type_as(inv_freq)
        freqs = torch.outer(t, inv_freq)
        cos = torch.cos(freqs)
        sin = torch.sin(freqs)
        return cos, sin
    
    def forward(self, x: torch.Tensor, seq_len: Optional[int] = None) -> torch.Tensor:
        """Apply rotary positional encoding to input."""
        if seq_len is None:
            seq_len = x.size(1)
        
        cos = self.cos_cached[:seq_len, :]
        sin = self.sin_cached[:seq_len, :]
        
        # Apply rotary encoding
        x_rot = x.clone()
        x_rot[:, :, 0::2] = x[:, :, 0::2] * cos - x[:, :, 1::2] * sin
        x_rot[:, :, 1::2] = x[:, :, 0::2] * sin + x[:, :, 1::2] * cos
        
        return x_rot


class ALiBiPositionalEncoding(nn.Module):
    """Attention with Linear Biases (ALiBi) positional encoding."""
    
    def __init__(self, n_heads: int, max_seq_length: int = 512):
        super().__init__()
        self.n_heads = n_heads
        self.max_seq_length = max_seq_length
        
        # Generate ALiBi slopes
        slopes = torch.Tensor(self._get_slopes(n_heads))
        self.register_buffer('slopes', slopes)
        
        # Generate ALiBi bias
        bias = torch.arange(max_seq_length, dtype=torch.float32)
        bias = bias.unsqueeze(0).unsqueeze(0)
        bias = bias * slopes.unsqueeze(1)
        self.register_buffer('bias', bias)
    
    def _get_slopes(self, n_heads: int) -> list:
        """Generate slopes for ALiBi encoding."""
        def get_slopes_power_of_2(n):
            start = (2**(-2**-(math.log2(n)-3)))
            ratio = start
            return [start*ratio**i for i in range(n)]
        
        if math.log2(n_heads).is_integer():
            return get_slopes_power_of_2(n_heads)
        else:
            closest_power_of_2 = 2**math.floor(math.log2(n_heads))
            return (get_slopes_power_of_2(closest_power_of_2) + 
                   self._get_slopes(2 * closest_power_of_2)[0::2][:n_heads-closest_power_of_2])
    
    def forward(self, attention_scores: torch.Tensor) -> torch.Tensor:
        """Add ALiBi bias to attention scores."""
        seq_len = attention_scores.size(-1)
        if seq_len <= self.max_seq_length:
            bias = self.bias[:, :, :seq_len, :seq_len]
        else:
            # Extend bias for longer sequences
            bias = torch.zeros(
                self.n_heads, 1, seq_len, seq_len,
                device=attention_scores.device
            )
            for i in range(seq_len):
                for j in range(seq_len):
                    bias[:, 0, i, j] = self.slopes * (j - i)
        
        return attention_scores + bias


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism with various attention types."""
    
    def __init__(self, config: AttentionConfig):
        super().__init__()
        self.config = config
        
        # Linear transformations
        self.w_q = nn.Linear(config.d_model, config.n_heads * config.d_k, bias=False)
        self.w_k = nn.Linear(config.d_model, config.n_heads * config.d_k, bias=False)
        self.w_v = nn.Linear(config.d_model, config.n_heads * config.d_v, bias=False)
        self.w_o = nn.Linear(config.n_heads * config.d_v, config.d_model, bias=False)
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
        
        # Positional encodings
        self.use_relative_position = config.use_relative_position
        self.use_rotary_position = config.use_rotary_position
        self.use_alibi_position = config.use_alibi_position
        
        if config.use_relative_position:
            self.relative_position_encoding = RelativePositionalEncoding(
                config.d_model, 
                config.max_seq_length
            )
        
        if config.use_rotary_position:
            self.rotary_position_encoding = RotaryPositionalEncoding(
                config.d_model, 
                config.max_seq_length
            )
        
        if config.use_alibi_position:
            self.alibi_position_encoding = ALiBiPositionalEncoding(
                config.n_heads, 
                config.max_seq_length
            )
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of multi-head attention."""
        batch_size = query.size(0)
        
        # Linear transformations and reshape
        Q = self.w_q(query).view(batch_size, -1, self.config.n_heads, self.config.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.config.n_heads, self.config.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.config.n_heads, self.config.d_v).transpose(1, 2)
        
        # Apply rotary positional encoding if enabled
        if self.use_rotary_position:
            Q = self.rotary_position_encoding(Q, Q.size(2))
            K = self.rotary_position_encoding(K, K.size(2))
        
        # Compute attention scores
        if self.config.attention_type == "scaled_dot_product":
            attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.config.d_k)
        elif self.config.attention_type == "linear":
            attention_scores = self._linear_attention(Q, K, V)
        elif self.config.attention_type == "local":
            attention_scores = self._local_attention(Q, K, V)
        elif self.config.attention_type == "sparse":
            attention_scores = self._sparse_attention(Q, K, V)
        else:
            raise ValueError(f"Unknown attention type: {self.config.attention_type}")
        
        # Apply ALiBi positional encoding if enabled
        if self.use_alibi_position:
            attention_scores = self.alibi_position_encoding(attention_scores)
        
        # Apply mask if provided
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax and dropout
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        
        # Reshape and apply output linear transformation
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.config.n_heads * self.config.d_v
        )
        output = self.w_o(context)
        
        return output, attention_weights
    
    def _linear_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """Linear attention implementation for efficiency."""
        # This is a simplified linear attention implementation
        # In practice, you might want to use more sophisticated linear attention methods
        Q = F.elu(Q) + 1
        K = F.elu(K) + 1
        
        KV = torch.matmul(K.transpose(-2, -1), V)
        QKV = torch.matmul(Q, KV)
        
        K_sum = K.sum(dim=-2, keepdim=True)
        QK_sum = torch.matmul(Q, K_sum.transpose(-2, -1))
        
        return QKV / (QK_sum + 1e-8)
    
    def _local_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """Local attention implementation with sliding window."""
        seq_len = Q.size(2)
        window_size = min(64, seq_len)  # Local window size
        
        # Pad sequences for local attention
        pad_size = window_size // 2
        Q_padded = F.pad(Q, (0, 0, pad_size, pad_size))
        K_padded = F.pad(K, (0, 0, pad_size, pad_size))
        V_padded = F.pad(V, (0, 0, pad_size, pad_size))
        
        attention_scores = []
        for i in range(seq_len):
            start_idx = i
            end_idx = i + window_size
            q_i = Q[:, :, i:i+1, :]
            k_local = K_padded[:, :, start_idx:end_idx, :]
            v_local = V_padded[:, :, start_idx:end_idx, :]
            
            scores = torch.matmul(q_i, k_local.transpose(-2, -1)) / math.sqrt(self.config.d_k)
            attention_scores.append(scores)
        
        return torch.cat(attention_scores, dim=2)
    
    def _sparse_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """Sparse attention implementation with top-k selection."""
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.config.d_k)
        
        # Select top-k attention scores
        k = min(32, scores.size(-1))  # Top-k parameter
        top_k_scores, top_k_indices = torch.topk(scores, k, dim=-1)
        
        # Create sparse attention matrix
        batch_size, n_heads, seq_len, _ = scores.size()
        sparse_scores = torch.zeros_like(scores)
        
        for b in range(batch_size):
            for h in range(n_heads):
                for i in range(seq_len):
                    sparse_scores[b, h, i, top_k_indices[b, h, i]] = top_k_scores[b, h, i]
        
        return sparse_scores


class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention mechanism."""
    
    def __init__(self, d_k: int, dropout: float = 0.1):
        super().__init__()
        self.d_k = d_k
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of scaled dot-product attention."""
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax and dropout
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        
        return context, attention_weights


class SelfAttention(nn.Module):
    """Self-attention mechanism for processing sequences."""
    
    def __init__(self, config: AttentionConfig):
        super().__init__()
        self.multi_head_attention = MultiHeadAttention(config)
        self.layer_norm = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass of self-attention."""
        # Self-attention
        attn_output, _ = self.multi_head_attention(x, x, x, mask)
        
        # Residual connection and layer normalization
        output = self.layer_norm(x + self.dropout(attn_output))
        
        return output


class CrossAttention(nn.Module):
    """Cross-attention mechanism for processing different sequences."""
    
    def __init__(self, config: AttentionConfig):
        super().__init__()
        self.multi_head_attention = MultiHeadAttention(config)
        self.layer_norm = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass of cross-attention."""
        # Cross-attention
        attn_output, _ = self.multi_head_attention(query, key, value, mask)
        
        # Residual connection and layer normalization
        output = self.layer_norm(query + self.dropout(attn_output))
        
        return output


class AttentionBlock(nn.Module):
    """Complete attention block with feed-forward network."""
    
    def __init__(self, config: AttentionConfig):
        super().__init__()
        self.self_attention = SelfAttention(config)
        self.cross_attention = CrossAttention(config)
        self.feed_forward = nn.Sequential(
            nn.Linear(config.d_model, config.d_model * 4),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model * 4, config.d_model)
        )
        self.layer_norm1 = nn.LayerNorm(config.d_model)
        self.layer_norm2 = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x: torch.Tensor, encoder_output: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass of attention block."""
        # Self-attention
        x = self.self_attention(x, mask)
        
        # Cross-attention if encoder output is provided
        if encoder_output is not None:
            x = self.cross_attention(x, encoder_output, encoder_output, mask)
        
        # Feed-forward network
        ff_output = self.feed_forward(x)
        x = self.layer_norm2(x + self.dropout(ff_output))
        
        return x


class AttentionSystem:
    """Complete attention system with all mechanisms and encodings."""
    
    def __init__(self, config: AttentionConfig):
        self.config = config
        self.positional_encoding = self._create_positional_encoding()
        self.attention_block = AttentionBlock(config)
    
    def _create_positional_encoding(self) -> nn.Module:
        """Create appropriate positional encoding based on configuration."""
        if self.config.use_rotary_position:
            return RotaryPositionalEncoding(self.config.d_model, self.config.max_seq_length)
        elif self.config.use_alibi_position:
            return None  # ALiBi is applied in attention scores
        else:
            return SinusoidalPositionalEncoding(
                self.config.d_model, 
                self.config.max_seq_length, 
                self.config.dropout
            )
    
    def process_sequence(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None,
                        encoder_output: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Process input sequence through the attention system."""
        # Add positional encoding if not using ALiBi
        if self.positional_encoding is not None:
            x = self.positional_encoding(x)
        
        # Process through attention block
        output = self.attention_block(x, encoder_output, mask)
        
        return output
    
    def get_attention_weights(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Get attention weights for analysis."""
        # This would require modifying the attention mechanism to return weights
        # For now, we'll use a simplified approach
        batch_size, seq_len, d_model = x.size()
        
        # Create dummy attention weights for demonstration
        attention_weights = torch.ones(batch_size, self.config.n_heads, seq_len, seq_len)
        attention_weights = attention_weights / seq_len  # Normalize
        
        return attention_weights


# Example usage and testing
def create_attention_system_example():
    """Create and test the attention system."""
    # Configuration
    config = AttentionConfig(
        d_model=512,
        n_heads=8,
        d_k=64,
        d_v=64,
        dropout=0.1,
        max_seq_length=512,
        use_relative_position=True,
        use_rotary_position=False,
        use_alibi_position=False,
        attention_type="scaled_dot_product"
    )
    
    # Create attention system
    attention_system = AttentionSystem(config)
    
    # Create sample input
    batch_size = 2
    seq_len = 10
    d_model = 512
    
    x = torch.randn(batch_size, seq_len, d_model)
    mask = torch.ones(batch_size, seq_len, seq_len)
    
    # Process sequence
    output = attention_system.process_sequence(x, mask)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention system created successfully!")
    
    return attention_system, output


if __name__ == "__main__":
    # Test the attention system
    attention_system, output = create_attention_system_example()
    
    # Test different attention types
    configs = [
        AttentionConfig(attention_type="scaled_dot_product"),
        AttentionConfig(attention_type="linear"),
        AttentionConfig(attention_type="local"),
        AttentionConfig(attention_type="sparse")
    ]
    
    for config in configs:
        print(f"\nTesting {config.attention_type} attention...")
        try:
            system = AttentionSystem(config)
            x = torch.randn(1, 5, config.d_model)
            output = system.process_sequence(x)
            print(f"✓ {config.attention_type} attention working")
        except Exception as e:
            print(f"✗ {config.attention_type} attention failed: {e}")


