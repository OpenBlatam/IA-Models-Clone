"""
Blatam AI - Advanced Attention Mechanisms and Positional Encodings v6.0.0
Ultra-optimized PyTorch-based attention mechanisms and positional encodings
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# ADVANCED ATTENTION MECHANISMS
# ============================================================================

class MultiHeadAttention(nn.Module):
    """Advanced multi-head attention mechanism with optimizations."""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1,
                 use_flash_attention: bool = False, use_xformers: bool = False,
                 attention_type: str = "scaled_dot_product"):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.dropout = dropout
        self.use_flash_attention = use_flash_attention
        self.use_xformers = use_xformers
        self.attention_type = attention_type
        
        # Linear projections
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        
        # Dropout layers
        self.dropout_layer = nn.Dropout(dropout)
        self.attention_dropout = nn.Dropout(dropout)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize attention weights with proper scaling."""
        nn.init.xavier_uniform_(self.w_q.weight)
        nn.init.xavier_uniform_(self.w_k.weight)
        nn.init.xavier_uniform_(self.w_v.weight)
        nn.init.xavier_uniform_(self.w_o.weight)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None, return_attention: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass with advanced attention mechanisms."""
        batch_size, seq_len, _ = query.size()
        
        # Linear projections and reshape
        Q = self.w_q(query).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Compute attention based on type
        if self.attention_type == "scaled_dot_product":
            attention_output, attention_weights = self._scaled_dot_product_attention(Q, K, V, mask)
        elif self.attention_type == "additive":
            attention_output, attention_weights = self._additive_attention(Q, K, V, mask)
        elif self.attention_type == "multiplicative":
            attention_output, attention_weights = self._multiplicative_attention(Q, K, V, mask)
        else:
            raise ValueError(f"Unknown attention type: {self.attention_type}")
            
        # Reshape and project output
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        output = self.w_o(attention_output)
        
        # Residual connection and layer normalization
        output = self.layer_norm(query + self.dropout_layer(output))
        
        if return_attention:
            return output, attention_weights
        return output
        
    def _scaled_dot_product_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                                     mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Scaled dot-product attention with optimizations."""
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply mask if provided
        if mask is not None:
            # Expand mask for multi-head attention
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
            
        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.attention_dropout(attention_weights)
        
        # Apply attention to values
        attention_output = torch.matmul(attention_weights, V)
        
        return attention_output, attention_weights
        
    def _additive_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                           mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Additive attention mechanism."""
        # Project queries and keys to attention space
        attention_dim = self.d_k // 2
        Q_proj = self.w_q.weight[:attention_dim, :].unsqueeze(0).unsqueeze(0)
        K_proj = self.w_k.weight[:attention_dim, :].unsqueeze(0).unsqueeze(0)
        
        # Compute additive attention scores
        Q_expanded = Q.unsqueeze(-2)  # [batch, heads, seq_len, 1, d_k]
        K_expanded = K.unsqueeze(-3)  # [batch, heads, 1, seq_len, d_k]
        
        # Additive attention: tanh(W_q * Q + W_k * K)
        attention_scores = torch.tanh(
            torch.matmul(Q_expanded, Q_proj.transpose(-2, -1)) +
            torch.matmul(K_expanded, K_proj.transpose(-2, -1))
        ).squeeze(-1).squeeze(-1)
        
        # Apply mask if provided
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
            
        # Apply softmax
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.attention_dropout(attention_weights)
        
        # Apply attention to values
        attention_output = torch.matmul(attention_weights, V)
        
        return attention_output, attention_weights
        
    def _multiplicative_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                                 mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Multiplicative attention mechanism."""
        # Learnable attention parameter
        if not hasattr(self, 'attention_param'):
            self.register_parameter('attention_param', nn.Parameter(torch.randn(self.d_k)))
            
        # Compute multiplicative attention scores
        Q_scaled = Q * self.attention_param.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        scores = torch.matmul(Q_scaled, K.transpose(-2, -1))
        
        # Apply mask if provided
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
            
        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.attention_dropout(attention_weights)
        
        # Apply attention to values
        attention_output = torch.matmul(attention_weights, V)
        
        return attention_output, attention_weights

class RelativePositionalEncoding(nn.Module):
    """Relative positional encoding for better sequence modeling."""
    
    def __init__(self, d_model: int, max_relative_position: int = 32):
        super().__init__()
        self.d_model = d_model
        self.max_relative_position = max_relative_position
        
        # Relative position embeddings
        self.rel_pos_embeddings = nn.Embedding(
            2 * max_relative_position + 1, d_model
        )
        
        # Initialize embeddings
        self._init_embeddings()
        
    def _init_embeddings(self):
        """Initialize relative position embeddings."""
        nn.init.normal_(self.rel_pos_embeddings.weight, mean=0.0, std=0.02)
        
    def forward(self, length: int, device: torch.device) -> torch.Tensor:
        """Generate relative positional encodings."""
        range_vec = torch.arange(length, device=device)
        range_mat = range_vec.unsqueeze(0).repeat(length, 1)
        distance_mat = range_mat - range_mat.transpose(0, 1)
        
        # Clip distances to max_relative_position
        distance_mat = torch.clamp(
            distance_mat, 
            -self.max_relative_position, 
            self.max_relative_position
        )
        
        # Shift to non-negative indices
        distance_mat = distance_mat + self.max_relative_position
        
        # Get embeddings
        rel_pos_embeddings = self.rel_pos_embeddings(distance_mat)
        
        return rel_pos_embeddings

class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding as in the original Transformer paper."""
    
    def __init__(self, d_model: int, max_seq_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.dropout = nn.Dropout(dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len).unsqueeze(1).float()
        
        # Compute division term
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(math.log(10000.0) / d_model))
        
        # Apply sine and cosine functions
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer (not parameter)
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add sinusoidal positional encoding to input."""
        seq_len = x.size(1)
        
        if seq_len <= self.max_seq_len:
            pos_encoding = self.pe[:, :seq_len, :]
        else:
            # For longer sequences, compute on-the-fly
            pos_encoding = self._compute_positional_encoding(seq_len, x.device)
            
        x = x + pos_encoding
        return self.dropout(x)
        
    def _compute_positional_encoding(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Compute positional encoding for sequences longer than max_seq_len."""
        pe = torch.zeros(seq_len, self.d_model, device=device)
        position = torch.arange(0, seq_len, device=device).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, self.d_model, 2, device=device).float() * 
                           -(math.log(10000.0) / self.d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)

class LearnablePositionalEncoding(nn.Module):
    """Learnable positional encoding with initialization."""
    
    def __init__(self, d_model: int, max_seq_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.dropout = nn.Dropout(dropout)
        
        # Learnable positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, max_seq_len, d_model))
        
        # Initialize with sinusoidal encoding
        self._init_with_sinusoidal()
        
    def _init_with_sinusoidal(self):
        """Initialize learnable encoding with sinusoidal values."""
        pe = torch.zeros(self.max_seq_len, self.d_model)
        position = torch.arange(0, self.max_seq_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * 
                           -(math.log(10000.0) / self.d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.pos_encoding.data = pe.unsqueeze(0)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add learnable positional encoding to input."""
        seq_len = x.size(1)
        
        if seq_len <= self.max_seq_len:
            pos_encoding = self.pos_encoding[:, :seq_len, :]
        else:
            # For longer sequences, interpolate or extend
            pos_encoding = self._extend_positional_encoding(seq_len)
            
        x = x + pos_encoding
        return self.dropout(x)
        
    def _extend_positional_encoding(self, seq_len: int) -> torch.Tensor:
        """Extend positional encoding for longer sequences."""
        # Interpolate existing encoding
        pos_encoding = F.interpolate(
            self.pos_encoding.transpose(1, 2),
            size=seq_len,
            mode='linear',
            align_corners=False
        ).transpose(1, 2)
        
        return pos_encoding

class RotaryPositionalEncoding(nn.Module):
    """Rotary positional encoding (RoPE) for better sequence modeling."""
    
    def __init__(self, d_model: int, max_seq_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Generate rotation matrices
        self.register_buffer('cos_cached', self._precompute_cos())
        self.register_buffer('sin_cached', self._precompute_sin())
        
    def _precompute_cos(self) -> torch.Tensor:
        """Precompute cosine values for rotary encoding."""
        position = torch.arange(0, self.max_seq_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * 
                           -(math.log(10000.0) / self.d_model))
        
        cos = torch.cos(position * div_term)
        return cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, d_model//2]
        
    def _precompute_sin(self) -> torch.Tensor:
        """Precompute sine values for rotary encoding."""
        position = torch.arange(0, self.max_seq_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * 
                           -(math.log(10000.0) / self.d_model))
        
        sin = torch.sin(position * div_term)
        return sin.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, d_model//2]
        
    def forward(self, x: torch.Tensor, seq_len: Optional[int] = None) -> torch.Tensor:
        """Apply rotary positional encoding."""
        if seq_len is None:
            seq_len = x.size(1)
            
        if seq_len <= self.max_seq_len:
            cos = self.cos_cached[:, :, :seq_len, :]
            sin = self.sin_cached[:, :, :seq_len, :]
        else:
            # Compute for longer sequences
            cos, sin = self._compute_rotary_encoding(seq_len, x.device)
            
        # Apply rotary encoding
        x_rot = self._apply_rotary_encoding(x, cos, sin)
        
        return x_rot
        
    def _compute_rotary_encoding(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute rotary encoding for longer sequences."""
        position = torch.arange(0, seq_len, device=device).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, self.d_model, 2, device=device).float() * 
                           -(math.log(10000.0) / self.d_model))
        
        cos = torch.cos(position * div_term).unsqueeze(0).unsqueeze(0)
        sin = torch.sin(position * div_term).unsqueeze(0).unsqueeze(0)
        
        return cos, sin
        
    def _apply_rotary_encoding(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Apply rotary encoding to input tensor."""
        # Reshape for rotary encoding
        x_reshaped = x.view(*x.shape[:-1], -1, 2)
        
        # Apply rotation
        x_rot = torch.cat([
            x_reshaped[..., 0:1] * cos - x_reshaped[..., 1:2] * sin,
            x_reshaped[..., 0:1] * sin + x_reshaped[..., 1:2] * cos
        ], dim=-1)
        
        return x_rot.view(*x.shape)

class AttentionWithRelativePositioning(nn.Module):
    """Attention mechanism with relative positional encoding."""
    
    def __init__(self, d_model: int, n_heads: int, max_relative_position: int = 32,
                 dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.max_relative_position = max_relative_position
        self.dropout = dropout
        
        # Standard attention components
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        
        # Relative position components
        self.rel_pos_k = nn.Linear(d_model, d_model, bias=False)
        self.rel_pos_v = nn.Linear(d_model, d_model, bias=False)
        
        # Relative position embeddings
        self.rel_pos_embeddings_k = nn.Embedding(2 * max_relative_position + 1, self.d_k)
        self.rel_pos_embeddings_v = nn.Embedding(2 * max_relative_position + 1, self.d_k)
        
        # Dropout and normalization
        self.dropout_layer = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize attention weights."""
        nn.init.xavier_uniform_(self.w_q.weight)
        nn.init.xavier_uniform_(self.w_k.weight)
        nn.init.xavier_uniform_(self.w_v.weight)
        nn.init.xavier_uniform_(self.w_o.weight)
        nn.init.xavier_uniform_(self.rel_pos_k.weight)
        nn.init.xavier_uniform_(self.rel_pos_v.weight)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with relative positional encoding."""
        batch_size, seq_len, _ = query.size()
        
        # Linear projections
        Q = self.w_q(query).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Compute relative positions
        rel_pos_k = self._get_relative_positions(seq_len, K.device)
        rel_pos_v = self._get_relative_positions(seq_len, V.device)
        
        # Compute attention with relative positioning
        attention_output = self._compute_relative_attention(Q, K, V, rel_pos_k, rel_pos_v, mask)
        
        # Reshape and project output
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        output = self.w_o(attention_output)
        
        # Residual connection and layer normalization
        output = self.layer_norm(query + self.dropout_layer(output))
        
        return output
        
    def _get_relative_positions(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Get relative position indices."""
        range_vec = torch.arange(seq_len, device=device)
        range_mat = range_vec.unsqueeze(0).repeat(seq_len, 1)
        distance_mat = range_mat - range_mat.transpose(0, 1)
        
        # Clip distances
        distance_mat = torch.clamp(
            distance_mat, 
            -self.max_relative_position, 
            self.max_relative_position
        )
        
        # Shift to non-negative indices
        distance_mat = distance_mat + self.max_relative_position
        
        return distance_mat
        
    def _compute_relative_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                                   rel_pos_k: torch.Tensor, rel_pos_v: torch.Tensor,
                                   mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute attention with relative positional encoding."""
        # Get relative position embeddings
        rel_pos_k_emb = self.rel_pos_embeddings_k(rel_pos_k)
        rel_pos_v_emb = self.rel_pos_embeddings_v(rel_pos_v)
        
        # Compute content-based attention
        content_scores = torch.matmul(Q, K.transpose(-2, -1))
        
        # Compute position-based attention
        pos_scores = torch.matmul(Q, rel_pos_k_emb.transpose(-2, -1))
        
        # Combine scores
        scores = (content_scores + pos_scores) / math.sqrt(self.d_k)
        
        # Apply mask if provided
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
            
        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout_layer(attention_weights)
        
        # Apply attention to values and relative positions
        content_output = torch.matmul(attention_weights, V)
        pos_output = torch.matmul(attention_weights, rel_pos_v_emb)
        
        # Combine outputs
        attention_output = content_output + pos_output
        
        return attention_output

# ============================================================================
# USAGE EXAMPLES
# ============================================================================

def main():
    """Main examples for attention mechanisms and positional encodings."""
    # Test multi-head attention
    attention = MultiHeadAttention(d_model=512, n_heads=8, dropout=0.1)
    
    # Test inputs
    batch_size, seq_len, d_model = 2, 128, 512
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Forward pass
    output = attention(x, x, x)
    logger.info(f"Multi-head attention output shape: {output.shape}")
    
    # Test sinusoidal positional encoding
    pos_encoding = SinusoidalPositionalEncoding(d_model=512, max_seq_len=1000)
    encoded = pos_encoding(x)
    logger.info(f"Sinusoidal encoding output shape: {encoded.shape}")
    
    # Test learnable positional encoding
    learnable_pos = LearnablePositionalEncoding(d_model=512, max_seq_len=1000)
    encoded_learnable = learnable_pos(x)
    logger.info(f"Learnable encoding output shape: {encoded_learnable.shape}")
    
    # Test rotary positional encoding
    rotary_pos = RotaryPositionalEncoding(d_model=512, max_seq_len=1000)
    encoded_rotary = rotary_pos(x)
    logger.info(f"Rotary encoding output shape: {encoded_rotary.shape}")
    
    # Test attention with relative positioning
    rel_attention = AttentionWithRelativePositioning(d_model=512, n_heads=8)
    rel_output = rel_attention(x, x, x)
    logger.info(f"Relative attention output shape: {rel_output.shape}")
    
    print("Attention mechanisms and positional encodings ready!")

if __name__ == "__main__":
    main()

