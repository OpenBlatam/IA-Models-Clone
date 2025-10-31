#!/usr/bin/env python3
"""
Advanced Transformer Models - State-of-the-art transformer implementations
Implements multi-head attention, positional encoding, and transformer architectures
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
import numpy as np
from pathlib import Path
import json
import uuid
from datetime import datetime, timezone

@dataclass
class TransformerConfig:
    """Configuration for transformer models."""
    # Model architecture
    vocab_size: int = 50257
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 2048
    dropout: float = 0.1
    activation: str = "relu"
    
    # Positional encoding
    max_seq_length: int = 512
    use_learned_pos_encoding: bool = True
    
    # Attention
    attention_dropout: float = 0.1
    use_flash_attention: bool = True
    use_multi_query_attention: bool = False
    
    # Training
    use_gradient_checkpointing: bool = True
    use_mixed_precision: bool = True
    
    # Advanced features
    use_rotary_embeddings: bool = True
    use_relative_position_bias: bool = False
    use_scale_attention: bool = True
    attention_scale_factor: float = 1.0

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models."""
    
    def __init__(self, d_model: int, max_length: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_length, d_model)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply positional encoding to input."""
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class RotaryEmbedding(nn.Module):
    """Rotary positional embeddings (RoPE)."""
    
    def __init__(self, d_model: int, max_length: int = 5000):
        super().__init__()
        self.d_model = d_model
        self.max_length = max_length
        
        # Precompute rotation matrices
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer('inv_freq', inv_freq)
    
    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary embeddings."""
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()[None, :, :]
        sin = emb.sin()[None, :, :]
        return cos, sin

class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism with advanced features."""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1,
                 use_flash_attention: bool = True, use_rotary_embeddings: bool = True,
                 attention_scale_factor: float = 1.0):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.use_flash_attention = use_flash_attention
        self.use_rotary_embeddings = use_rotary_embeddings
        self.attention_scale_factor = attention_scale_factor
        
        # Linear projections
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Rotary embeddings
        if use_rotary_embeddings:
            self.rotary_emb = RotaryEmbedding(d_model)
        
        # Scale factor
        self.scale = math.sqrt(self.d_k) * attention_scale_factor
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass of multi-head attention."""
        batch_size, seq_len = query.size(0), query.size(1)
        
        # Linear projections
        Q = self.w_q(query).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Apply rotary embeddings if enabled
        if self.use_rotary_embeddings:
            cos, sin = self.rotary_emb(query, seq_len)
            Q = self._apply_rotary_embeddings(Q, cos, sin)
            K = self._apply_rotary_embeddings(K, cos, sin)
        
        # Use flash attention if available and enabled
        if self.use_flash_attention and hasattr(F, 'scaled_dot_product_attention'):
            try:
                attn_output = F.scaled_dot_product_attention(
                    Q, K, V, attn_mask=mask, dropout_p=self.dropout.p if self.training else 0.0
                )
            except Exception:
                # Fallback to manual attention
                attn_output = self._manual_attention(Q, K, V, mask)
        else:
            attn_output = self._manual_attention(Q, K, V, mask)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        return self.w_o(attn_output)
    
    def _apply_rotary_embeddings(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Apply rotary embeddings to input."""
        x1, x2 = x[..., ::2], x[..., 1::2]
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    
    def _manual_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                         mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Manual attention computation."""
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)
        
        return attn_output

class FeedForward(nn.Module):
    """Feed-forward network with advanced features."""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1, activation: str = "relu"):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        
        # Linear layers
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        
        # Activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "swish":
            self.activation = nn.SiLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of feed-forward network."""
        return self.linear2(self.dropout(self.activation(self.linear1(x))))

class TransformerBlock(nn.Module):
    """Transformer block with advanced features."""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1,
                 activation: str = "relu", use_flash_attention: bool = True,
                 use_rotary_embeddings: bool = True, attention_scale_factor: float = 1.0):
        super().__init__()
        
        # Multi-head attention
        self.attention = MultiHeadAttention(
            d_model, n_heads, dropout, use_flash_attention, 
            use_rotary_embeddings, attention_scale_factor
        )
        
        # Feed-forward network
        self.feed_forward = FeedForward(d_model, d_ff, dropout, activation)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass of transformer block."""
        # Self-attention with residual connection
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class TransformerModel(nn.Module):
    """Advanced transformer model with state-of-the-art features."""
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Embedding layers
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        
        # Positional encoding
        if config.use_learned_pos_encoding:
            self.pos_embedding = nn.Embedding(config.max_seq_length, config.d_model)
        else:
            self.pos_encoding = PositionalEncoding(config.d_model, config.max_seq_length, config.dropout)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                config.d_model, config.n_heads, config.d_ff, config.dropout,
                config.activation, config.use_flash_attention,
                config.use_rotary_embeddings, config.attention_scale_factor
            )
            for _ in range(config.n_layers)
        ])
        
        # Output layers
        self.layer_norm = nn.LayerNorm(config.d_model)
        self.output_projection = nn.Linear(config.d_model, config.vocab_size)
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass of transformer model."""
        batch_size, seq_len = input_ids.size()
        
        # Token embeddings
        x = self.token_embedding(input_ids)
        
        # Positional encoding
        if self.config.use_learned_pos_encoding:
            if position_ids is None:
                position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
            x = x + self.pos_embedding(position_ids)
        else:
            x = self.pos_encoding(x.transpose(0, 1)).transpose(0, 1)
        
        # Apply dropout
        x = self.dropout(x)
        
        # Create attention mask
        if attention_mask is not None:
            # Convert attention mask to attention mask for attention layers
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = attention_mask.expand(batch_size, 1, seq_len, seq_len)
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x, attention_mask)
        
        # Layer normalization
        x = self.layer_norm(x)
        
        # Output projection
        logits = self.output_projection(x)
        
        return logits
    
    def generate(self, input_ids: torch.Tensor, max_length: int = 100,
                temperature: float = 1.0, top_k: int = 50, top_p: float = 0.9,
                do_sample: bool = True) -> torch.Tensor:
        """Generate text using the transformer model."""
        self.eval()
        
        with torch.no_grad():
            for _ in range(max_length - input_ids.size(1)):
                # Get logits
                logits = self.forward(input_ids)
                
                # Get next token logits
                next_token_logits = logits[:, -1, :] / temperature
                
                # Apply top-k filtering
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                    next_token_logits = torch.full_like(next_token_logits, -float('inf'))
                    next_token_logits.scatter_(1, top_k_indices, top_k_logits)
                
                # Apply top-p filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = -float('inf')
                
                # Sample next token
                if do_sample:
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # Append to input
                input_ids = torch.cat([input_ids, next_token], dim=1)
        
        return input_ids
    
    def get_attention_weights(self, input_ids: torch.Tensor, 
                            attention_mask: Optional[torch.Tensor] = None) -> List[torch.Tensor]:
        """Get attention weights from all layers."""
        self.eval()
        attention_weights = []
        
        with torch.no_grad():
            batch_size, seq_len = input_ids.size()
            
            # Token embeddings
            x = self.token_embedding(input_ids)
            
            # Positional encoding
            if self.config.use_learned_pos_encoding:
                position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
                x = x + self.pos_embedding(position_ids)
            else:
                x = self.pos_encoding(x.transpose(0, 1)).transpose(0, 1)
            
            # Create attention mask
            if attention_mask is not None:
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                attention_mask = attention_mask.expand(batch_size, 1, seq_len, seq_len)
            
            # Apply transformer blocks and collect attention weights
            for block in self.transformer_blocks:
                # This would need to be modified to return attention weights
                # For now, we'll just pass through
                x = block(x, attention_mask)
        
        return attention_weights
    
    def count_parameters(self) -> int:
        """Count total number of parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_size(self) -> Dict[str, Any]:
        """Get model size information."""
        total_params = self.count_parameters()
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'non_trainable_parameters': total_params - trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
            'layers': self.config.n_layers,
            'heads': self.config.n_heads,
            'hidden_size': self.config.d_model
        }
