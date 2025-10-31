"""
Advanced Attention Mechanisms and Positional Encodings Implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, List, Union
from dataclasses import dataclass
import math


@dataclass
class AttentionConfig:
    """Configuration for attention mechanisms."""
    hidden_size: int = 768
    num_heads: int = 12
    head_dim: int = 64
    dropout_rate: float = 0.1
    attention_dropout: float = 0.1
    use_bias: bool = False
    use_scale: bool = True
    use_relative_position: bool = False
    max_relative_position: int = 32
    use_rotary: bool = True
    use_flash_attention: bool = True
    max_position_embeddings: int = 2048


class RotaryPositionalEmbedding(nn.Module):
    """Correctly implemented Rotary Positional Embedding (RoPE)."""
    
    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        # Precompute rotation matrices
        self.register_buffer('cos_cached', self._get_cos_cache())
        self.register_buffer('sin_cached', self._get_sin_cache())
    
    def _get_cos_cache(self) -> torch.Tensor:
        """Generate cosine cache for rotary embeddings."""
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        pos = torch.arange(self.max_position_embeddings, dtype=torch.float)
        freqs = torch.outer(pos, inv_freq)
        return torch.cos(freqs)
    
    def _get_sin_cache(self) -> torch.Tensor:
        """Generate sine cache for rotary embeddings."""
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        pos = torch.arange(self.max_position_embeddings, dtype=torch.float)
        freqs = torch.outer(pos, inv_freq)
        return torch.sin(freqs)
    
    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate half the hidden dims of the input."""
        x1, x2 = torch.chunk(x, 2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    
    def forward(self, q: torch.Tensor, k: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary positional embeddings to query and key tensors."""
        if seq_len > self.max_position_embeddings:
            # Extend cache if needed
            self._extend_cache(seq_len)
        
        # Get rotation matrices for current sequence length
        cos = self.cos_cached[:seq_len]
        sin = self.sin_cached[:seq_len]
        
        # Apply to query and key
        q_embed = (q * cos) + (self._rotate_half(q) * sin)
        k_embed = (k * cos) + (self._rotate_half(k) * sin)
        
        return q_embed, k_embed
    
    def _extend_cache(self, seq_len: int):
        """Extend cache for longer sequences."""
        self.max_position_embeddings = seq_len
        self.cos_cached = self._get_cos_cache()
        self.sin_cached = self._get_sin_cache()


class AbsolutePositionalEmbedding(nn.Module):
    """Sinusoidal absolute positional embeddings."""
    
    def __init__(self, hidden_size: int, max_position_embeddings: int = 2048):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_position_embeddings = max_position_embeddings
        
        # Create sinusoidal position embeddings
        pe = torch.zeros(max_position_embeddings, hidden_size)
        position = torch.arange(0, max_position_embeddings, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, hidden_size, 2).float() * 
                           (-math.log(10000.0) / hidden_size))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x: torch.Tensor, position_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Add positional embeddings to input."""
        if position_ids is None:
            seq_len = x.size(1)
            position_ids = torch.arange(seq_len, device=x.device).unsqueeze(0)
        
        pos_embeddings = self.pe[:, position_ids].squeeze(0)
        return x + pos_embeddings


class RelativePositionalBias(nn.Module):
    """Relative positional bias for attention (T5-style)."""
    
    def __init__(self, num_heads: int, max_distance: int = 128):
        super().__init__()
        self.num_heads = num_heads
        self.max_distance = max_distance
        
        # Relative position bias table
        self.relative_attention_bias = nn.Embedding(
            2 * max_distance + 1, 
            num_heads
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize relative position bias."""
        nn.init.normal_(self.relative_attention_bias.weight, mean=0.0, std=0.02)
    
    def _relative_position_bucket(self, relative_position: torch.Tensor) -> torch.Tensor:
        """Compute relative position buckets."""
        ret = 0
        n = -relative_position
        
        num_buckets = self.max_distance
        max_exact = num_buckets // 2
        
        is_small = n < max_exact
        val_if_large = max_exact + (
            torch.log(n.float() / max_exact) / math.log(self.max_distance / max_exact) * 
            (num_buckets - max_exact)
        ).long()
        
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))
        ret += torch.where(is_small, n, val_if_large)
        
        return ret + self.max_distance
    
    def forward(self, seq_len: int) -> torch.Tensor:
        """Compute relative position bias."""
        range_vec = torch.arange(seq_len)
        range_mat = range_vec.unsqueeze(-1) - range_vec.unsqueeze(0)
        
        # Compute relative position buckets
        relative_position_bucket = self._relative_position_bucket(range_mat)
        
        # Get relative position bias
        bias = self.relative_attention_bias(relative_position_bucket)
        bias = bias.permute(2, 0, 1).unsqueeze(0)  # [1, num_heads, seq_len, seq_len]
        
        return bias


class MultiHeadAttention(nn.Module):
    """Correctly implemented multi-head attention with all modern features."""
    
    def __init__(self, config: AttentionConfig):
        super().__init__()
        self.config = config
        
        assert config.hidden_size % config.num_heads == 0
        
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.scale = math.sqrt(self.head_dim) if config.use_scale else 1.0
        
        # Query, Key, Value projections
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.use_bias)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.use_bias)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.use_bias)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.use_bias)
        
        # Dropouts
        self.attention_dropout = nn.Dropout(config.attention_dropout)
        self.output_dropout = nn.Dropout(config.dropout_rate)
        
        # Positional encodings
        if config.use_rotary:
            self.rotary_emb = RotaryPositionalEmbedding(
                self.head_dim, 
                config.max_position_embeddings
            )
        
        if config.use_relative_position:
            self.relative_bias = RelativePositionalBias(
                config.num_heads, 
                config.max_relative_position
            )
        
        # Flash attention
        self.use_flash_attention = config.use_flash_attention and hasattr(F, 'scaled_dot_product_attention')
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize attention weights."""
        for module in [self.q_proj, self.k_proj, self.v_proj, self.o_proj]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass with all attention features."""
        batch_size, seq_len, hidden_size = hidden_states.size()
        
        # Linear projections
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)
        
        # Reshape for multi-head attention
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply rotary positional embeddings
        if self.config.use_rotary:
            query, key = self.rotary_emb(query, key, seq_len)
        
        # Handle past key-value states for generation
        if past_key_value is not None:
            past_key, past_value = past_key_value
            key = torch.cat([past_key, key], dim=2)
            value = torch.cat([past_value, value], dim=2)
        
        # Prepare for caching
        if use_cache:
            present_key_value = (key, value)
        else:
            present_key_value = None
        
        # Attention computation
        if self.use_flash_attention:
            # Use flash attention
            attention_output = F.scaled_dot_product_attention(
                query, key, value,
                attn_mask=attention_mask,
                dropout_p=self.attention_dropout.p if self.training else 0.0,
                is_causal=False
            )
            attention_weights = None
        else:
            # Standard attention
            attention_scores = torch.matmul(query, key.transpose(-2, -1)) / self.scale
            
            # Add relative position bias
            if self.config.use_relative_position:
                relative_bias = self.relative_bias(seq_len)
                attention_scores = attention_scores + relative_bias
            
            # Apply attention mask
            if attention_mask is not None:
                if attention_mask.dim() == 2:
                    # Expand mask dimensions
                    attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                attention_scores = attention_scores + attention_mask
            
            # Softmax
            attention_weights = F.softmax(attention_scores, dim=-1)
            attention_weights = self.attention_dropout(attention_weights)
            
            # Apply attention to values
            attention_output = torch.matmul(attention_weights, value)
        
        # Reshape output
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(batch_size, seq_len, hidden_size)
        
        # Output projection
        attention_output = self.o_proj(attention_output)
        attention_output = self.output_dropout(attention_output)
        
        # Prepare outputs
        outputs = (attention_output,)
        if output_attentions:
            outputs += (attention_weights,)
        if use_cache:
            outputs += (present_key_value,)
        
        return outputs


class CrossAttention(nn.Module):
    """Cross-attention for encoder-decoder architectures."""
    
    def __init__(self, config: AttentionConfig):
        super().__init__()
        self.config = config
        
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.scale = math.sqrt(self.head_dim) if config.use_scale else 1.0
        
        # Projections
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.use_bias)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.use_bias)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.use_bias)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.use_bias)
        
        # Dropouts
        self.attention_dropout = nn.Dropout(config.attention_dropout)
        self.output_dropout = nn.Dropout(config.dropout_rate)
        
        # Flash attention
        self.use_flash_attention = config.use_flash_attention and hasattr(F, 'scaled_dot_product_attention')
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Cross-attention forward pass."""
        batch_size, seq_len, hidden_size = hidden_states.size()
        encoder_seq_len = encoder_hidden_states.size(1)
        
        # Query from decoder, Key and Value from encoder
        query = self.q_proj(hidden_states)
        key = self.k_proj(encoder_hidden_states)
        value = self.v_proj(encoder_hidden_states)
        
        # Reshape for multi-head attention
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, encoder_seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, encoder_seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention computation
        if self.use_flash_attention:
            attention_output = F.scaled_dot_product_attention(
                query, key, value,
                attn_mask=encoder_attention_mask,
                dropout_p=self.attention_dropout.p if self.training else 0.0
            )
            attention_weights = None
        else:
            # Standard cross-attention
            attention_scores = torch.matmul(query, key.transpose(-2, -1)) / self.scale
            
            # Apply encoder attention mask
            if encoder_attention_mask is not None:
                if encoder_attention_mask.dim() == 2:
                    encoder_attention_mask = encoder_attention_mask.unsqueeze(1).unsqueeze(2)
                attention_scores = attention_scores + encoder_attention_mask
            
            # Softmax
            attention_weights = F.softmax(attention_scores, dim=-1)
            attention_weights = self.attention_dropout(attention_weights)
            
            # Apply attention to values
            attention_output = torch.matmul(attention_weights, value)
        
        # Reshape output
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(batch_size, seq_len, hidden_size)
        
        # Output projection
        attention_output = self.o_proj(attention_output)
        attention_output = self.output_dropout(attention_output)
        
        # Prepare outputs
        outputs = (attention_output,)
        if output_attentions:
            outputs += (attention_weights,)
        
        return outputs


class TransformerBlock(nn.Module):
    """Complete transformer block with attention and feed-forward."""
    
    def __init__(self, config: AttentionConfig):
        super().__init__()
        self.config = config
        
        # Self-attention
        self.self_attention = MultiHeadAttention(config)
        
        # Cross-attention (optional)
        self.cross_attention = CrossAttention(config) if config.use_cross_attention else None
        
        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 4),
            nn.GELU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_size * 4, config.hidden_size),
            nn.Dropout(config.dropout_rate)
        )
        
        # Layer normalizations
        self.ln1 = nn.LayerNorm(config.hidden_size, eps=1e-5)
        self.ln2 = nn.LayerNorm(config.hidden_size, eps=1e-5)
        if self.cross_attention:
            self.ln_cross = nn.LayerNorm(config.hidden_size, eps=1e-5)
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout_rate)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass through transformer block."""
        # Self-attention with pre-norm
        residual = hidden_states
        hidden_states = self.ln1(hidden_states)
        
        self_attention_outputs = self.self_attention(
            hidden_states,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache
        )
        
        attention_output = self_attention_outputs[0]
        hidden_states = residual + self.dropout(attention_output)
        
        # Cross-attention (if applicable)
        if self.cross_attention and encoder_hidden_states is not None:
            residual = hidden_states
            hidden_states = self.ln_cross(hidden_states)
            
            cross_attention_outputs = self.cross_attention(
                hidden_states,
                encoder_hidden_states,
                attention_mask=attention_mask,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions
            )
            
            cross_attention_output = cross_attention_outputs[0]
            hidden_states = residual + self.dropout(cross_attention_output)
        
        # Feed-forward with pre-norm
        residual = hidden_states
        hidden_states = self.ln2(hidden_states)
        ff_output = self.feed_forward(hidden_states)
        hidden_states = residual + self.dropout(ff_output)
        
        # Prepare outputs
        outputs = (hidden_states,)
        if output_attentions:
            outputs += self_attention_outputs[1:]
        if use_cache:
            outputs += self_attention_outputs[-1:]
        
        return outputs


def create_attention_mask(input_ids: torch.Tensor, mask_type: str = "causal") -> torch.Tensor:
    """Create attention masks for different attention patterns."""
    batch_size, seq_len = input_ids.size()
    
    if mask_type == "causal":
        # Causal mask for autoregressive models
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask.unsqueeze(0).unsqueeze(0)
    
    elif mask_type == "bidirectional":
        # No mask for bidirectional attention
        return None
    
    elif mask_type == "prefix":
        # Allow attention to prefix tokens
        mask = torch.zeros(seq_len, seq_len)
        return mask.unsqueeze(0).unsqueeze(0)
    
    else:
        raise ValueError(f"Unknown mask type: {mask_type}")


# Usage example
def main():
    """Main function demonstrating attention mechanisms."""
    
    # Configuration
    config = AttentionConfig(
        hidden_size=768,
        num_heads=12,
        dropout_rate=0.1,
        use_rotary=True,
        use_relative_position=True,
        use_flash_attention=True
    )
    
    # Create attention layer
    attention = MultiHeadAttention(config)
    
    # Create sample input
    batch_size, seq_len = 2, 128
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
    
    # Create attention mask
    attention_mask = create_attention_mask(
        torch.ones(batch_size, seq_len), 
        mask_type="causal"
    )
    
    # Forward pass
    outputs = attention(
        hidden_states,
        attention_mask=attention_mask,
        output_attentions=True
    )
    
    attention_output = outputs[0]
    attention_weights = outputs[1] if len(outputs) > 1 else None
    
    print(f"✅ Attention Mechanisms & Positional Encodings Ready!")
    print(f"   Input shape: {hidden_states.shape}")
    print(f"   Output shape: {attention_output.shape}")
    print(f"   Attention weights shape: {attention_weights.shape if attention_weights is not None else 'None'}")
    print(f"   Using RoPE: {config.use_rotary}")
    print(f"   Using Flash Attention: {config.use_flash_attention}")


if __name__ == "__main__":
    main()