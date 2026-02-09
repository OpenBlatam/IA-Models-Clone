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
from typing import Optional, Tuple, Dict, Any, List, Union
import logging
from typing import Any, List, Dict, Optional
import asyncio
#!/usr/bin/env python3
"""
Advanced Attention Mechanisms and Positional Encodings
Comprehensive implementation of attention mechanisms and positional encodings for transformer models
"""


logger = logging.getLogger(__name__)

class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding as described in "Attention Is All You Need"
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        
    """__init__ function."""
super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Apply sine and cosine functions
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        # Register as buffer (not parameter)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input embeddings
        
        Args:
            x: Input embeddings [seq_len, batch_size, d_model]
            
        Returns:
            Embeddings with positional encoding added
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class LearnedPositionalEncoding(nn.Module):
    """
    Learned positional encoding that can be trained
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        
    """__init__ function."""
super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Parameter(torch.randn(max_len, d_model))
        
        # Initialize with small values
        nn.init.normal_(self.pe, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add learned positional encoding to input embeddings
        
        Args:
            x: Input embeddings [seq_len, batch_size, d_model]
            
        Returns:
            Embeddings with learned positional encoding added
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class RelativePositionalEncoding(nn.Module):
    """
    Relative positional encoding for better handling of sequence relationships
    """
    
    def __init__(self, d_model: int, max_relative_position: int = 32, dropout: float = 0.1):
        
    """__init__ function."""
super().__init__()
        self.d_model = d_model
        self.max_relative_position = max_relative_position
        self.dropout = nn.Dropout(p=dropout)
        
        # Relative position embeddings
        self.relative_position_embeddings = nn.Parameter(
            torch.randn(2 * max_relative_position + 1, d_model)
        )
        
        # Initialize with small values
        nn.init.normal_(self.relative_position_embeddings, std=0.02)
    
    def forward(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        """
        Add relative positional encoding
        
        Args:
            x: Input embeddings [batch_size, seq_len, d_model]
            seq_len: Length of the sequence
            
        Returns:
            Embeddings with relative positional encoding added
        """
        batch_size = x.size(0)
        
        # Create relative position indices
        range_vec = torch.arange(seq_len, device=x.device)
        range_mat = range_vec.unsqueeze(0).repeat(seq_len, 1)
        distance_mat = range_mat - range_mat.T
        
        # Clip distances to max_relative_position
        distance_mat_clipped = torch.clamp(
            distance_mat, 
            -self.max_relative_position, 
            self.max_relative_position
        )
        
        # Shift indices to be non-negative
        final_mat = distance_mat_clipped + self.max_relative_position
        
        # Get relative position embeddings
        embeddings = self.relative_position_embeddings[final_mat]
        
        # Add to input embeddings
        x = x + embeddings
        return self.dropout(x)

class RotaryPositionalEncoding(nn.Module):
    """
    Rotary Positional Encoding (RoPE) for better position modeling
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        
    """__init__ function."""
super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.dropout = nn.Dropout(p=dropout)
        
        # Ensure d_model is even
        assert d_model % 2 == 0, "d_model must be even for RoPE"
        
        # Create rotation matrices
        self._create_rotation_matrices()
    
    def _create_rotation_matrices(self) -> Any:
        """Create rotation matrices for RoPE"""
        position = torch.arange(0, self.max_len, dtype=torch.float)
        freqs = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float) * 
            (-math.log(10000.0) / self.d_model)
        )
        
        angles = position.unsqueeze(1) * freqs.unsqueeze(0)
        self.register_buffer('cos', torch.cos(angles))
        self.register_buffer('sin', torch.sin(angles))
    
    def forward(self, x: torch.Tensor, seq_len: Optional[int] = None) -> torch.Tensor:
        """
        Apply rotary positional encoding
        
        Args:
            x: Input embeddings [batch_size, seq_len, d_model]
            seq_len: Length of the sequence (optional)
            
        Returns:
            Embeddings with rotary positional encoding applied
        """
        if seq_len is None:
            seq_len = x.size(1)
        
        # Get rotation matrices for current sequence length
        cos = self.cos[:seq_len, :]  # [seq_len, d_model//2]
        sin = self.sin[:seq_len, :]  # [seq_len, d_model//2]
        
        # Reshape input for rotation
        x_rot = x.view(*x.shape[:-1], -1, 2)  # [batch_size, seq_len, d_model//2, 2]
        
        # Apply rotation
        x_rotated = torch.stack([
            x_rot[..., 0] * cos - x_rot[..., 1] * sin,
            x_rot[..., 0] * sin + x_rot[..., 1] * cos
        ], dim=-1)
        
        # Reshape back
        x_rotated = x_rotated.view(*x.shape)
        
        return self.dropout(x_rotated)

class MultiHeadAttention(nn.Module):
    """
    Standard Multi-Head Attention mechanism with proper implementation
    """
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1, 
                 bias: bool = True, attention_type: str = "scaled_dot_product"):
        
    """__init__ function."""
super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.attention_type = attention_type
        
        # Linear transformations
        self.w_q = nn.Linear(d_model, d_model, bias=bias)
        self.w_k = nn.Linear(d_model, d_model, bias=bias)
        self.w_v = nn.Linear(d_model, d_model, bias=bias)
        self.w_o = nn.Linear(d_model, d_model, bias=bias)
        
        # Dropout layers
        self.dropout = nn.Dropout(dropout)
        self.attention_dropout = nn.Dropout(dropout)
        
        # Scaling factor
        self.scale = math.sqrt(self.d_k)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self) -> Any:
        """Initialize attention weights properly"""
        nn.init.xavier_uniform_(self.w_q.weight)
        nn.init.xavier_uniform_(self.w_k.weight)
        nn.init.xavier_uniform_(self.w_v.weight)
        nn.init.xavier_uniform_(self.w_o.weight)
        
        if self.w_q.bias is not None:
            nn.init.zeros_(self.w_q.bias)
        if self.w_k.bias is not None:
            nn.init.zeros_(self.w_k.bias)
        if self.w_v.bias is not None:
            nn.init.zeros_(self.w_v.bias)
        if self.w_o.bias is not None:
            nn.init.zeros_(self.w_o.bias)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None, 
                key_padding_mask: Optional[torch.Tensor] = None,
                attn_mask: Optional[torch.Tensor] = None,
                need_weights: bool = True) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of multi-head attention
        
        Args:
            query: Query tensor [batch_size, seq_len, d_model]
            key: Key tensor [batch_size, seq_len, d_model]
            value: Value tensor [batch_size, seq_len, d_model]
            mask: Attention mask [batch_size, seq_len, seq_len]
            key_padding_mask: Key padding mask [batch_size, seq_len]
            attn_mask: Attention mask [seq_len, seq_len]
            need_weights: Whether to return attention weights
            
        Returns:
            Output tensor and attention weights (optional)
        """
        batch_size, seq_len, _ = query.size()
        
        # Linear transformations
        Q = self.w_q(query).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Apply attention masks
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, -1e9)
        
        if key_padding_mask is not None:
            # Expand key_padding_mask to match attention scores shape
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(key_padding_mask == 0, -1e9)
        
        # Apply attention weights
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.attention_dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        
        # Reshape and apply output projection
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        output = self.w_o(context)
        
        if need_weights:
            return output, attention_weights
        else:
            return output, None

class LocalAttention(nn.Module):
    """
    Local attention mechanism for efficient processing of long sequences
    """
    
    def __init__(self, d_model: int, num_heads: int, window_size: int = 128, 
                 dropout: float = 0.1):
        
    """__init__ function."""
super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.window_size = window_size
        self.dropout = nn.Dropout(dropout)
        
        # Linear transformations
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.scale = math.sqrt(self.d_k)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Local attention forward pass
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            mask: Attention mask [batch_size, seq_len, seq_len]
            
        Returns:
            Output tensor
        """
        batch_size, seq_len, _ = x.size()
        
        # Linear transformations
        Q = self.w_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Create local attention mask
        local_mask = self._create_local_mask(seq_len, self.window_size, x.device)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Apply local mask
        scores = scores.masked_fill(local_mask == 0, -1e9)
        
        # Apply additional mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply attention
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        return self.w_o(context)
    
    def _create_local_mask(self, seq_len: int, window_size: int, device: torch.device) -> torch.Tensor:
        """Create local attention mask"""
        mask = torch.zeros(seq_len, seq_len, device=device)
        
        for i in range(seq_len):
            start = max(0, i - window_size // 2)
            end = min(seq_len, i + window_size // 2 + 1)
            mask[i, start:end] = 1
        
        return mask

class SparseAttention(nn.Module):
    """
    Sparse attention mechanism for efficient processing
    """
    
    def __init__(self, d_model: int, num_heads: int, num_landmarks: int = 64,
                 dropout: float = 0.1):
        
    """__init__ function."""
super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.num_landmarks = num_landmarks
        self.dropout = nn.Dropout(dropout)
        
        # Linear transformations
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        # Landmark projections
        self.landmark_q = nn.Linear(d_model, d_model)
        self.landmark_k = nn.Linear(d_model, d_model)
        
        self.scale = math.sqrt(self.d_k)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Sparse attention forward pass
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            mask: Attention mask [batch_size, seq_len, seq_len]
            
        Returns:
            Output tensor
        """
        batch_size, seq_len, _ = x.size()
        
        # Select landmarks
        landmark_indices = torch.linspace(0, seq_len - 1, self.num_landmarks, dtype=torch.long)
        landmarks = x[:, landmark_indices, :]
        
        # Compute attention with landmarks
        Q = self.w_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K_landmarks = self.landmark_k(landmarks).view(batch_size, self.num_landmarks, self.num_heads, self.d_k).transpose(1, 2)
        V_landmarks = self.w_v(landmarks).view(batch_size, self.num_landmarks, self.num_heads, self.d_k).transpose(1, 2)
        
        # Compute attention scores with landmarks
        scores = torch.matmul(Q, K_landmarks.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            mask_landmarks = mask[:, :, landmark_indices]
            scores = scores.masked_fill(mask_landmarks == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        context = torch.matmul(attention_weights, V_landmarks)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        return self.w_o(context)

class AttentionWithRelativePositions(nn.Module):
    """
    Attention mechanism with relative position embeddings
    """
    
    def __init__(self, d_model: int, num_heads: int, max_relative_position: int = 32,
                 dropout: float = 0.1):
        
    """__init__ function."""
super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.max_relative_position = max_relative_position
        self.dropout = nn.Dropout(dropout)
        
        # Standard attention components
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        # Relative position embeddings
        self.relative_position_embeddings = nn.Parameter(
            torch.randn(2 * max_relative_position + 1, self.d_k)
        )
        
        self.scale = math.sqrt(self.d_k)
        
        # Initialize weights
        nn.init.normal_(self.relative_position_embeddings, std=0.02)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with relative positions
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            mask: Attention mask [batch_size, seq_len, seq_len]
            
        Returns:
            Output tensor
        """
        batch_size, seq_len, _ = x.size()
        
        # Linear transformations
        Q = self.w_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Compute content-based attention scores
        content_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Compute relative position scores
        relative_scores = self._compute_relative_scores(Q, seq_len)
        
        # Combine scores
        scores = content_scores + relative_scores
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        return self.w_o(context)
    
    def _compute_relative_scores(self, Q: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Compute relative position attention scores"""
        batch_size, num_heads, _, d_k = Q.size()
        
        # Create relative position indices
        range_vec = torch.arange(seq_len, device=Q.device)
        range_mat = range_vec.unsqueeze(0).repeat(seq_len, 1)
        distance_mat = range_mat - range_mat.T
        
        # Clip distances
        distance_mat_clipped = torch.clamp(
            distance_mat, 
            -self.max_relative_position, 
            self.max_relative_position
        )
        
        # Shift indices
        final_mat = distance_mat_clipped + self.max_relative_position
        
        # Get relative position embeddings
        relative_embeddings = self.relative_position_embeddings[final_mat]
        
        # Compute relative scores
        Q_reshaped = Q.view(batch_size * num_heads, seq_len, d_k)
        relative_scores = torch.matmul(Q_reshaped, relative_embeddings.transpose(-2, -1))
        relative_scores = relative_scores.view(batch_size, num_heads, seq_len, seq_len)
        
        return relative_scores

class AttentionFactory:
    """
    Factory class for creating different types of attention mechanisms
    """
    
    @staticmethod
    def create_attention(attention_type: str, d_model: int, num_heads: int, 
                        **kwargs) -> nn.Module:
        """
        Create attention mechanism based on type
        
        Args:
            attention_type: Type of attention mechanism
            d_model: Model dimension
            num_heads: Number of attention heads
            **kwargs: Additional arguments
            
        Returns:
            Attention mechanism module
        """
        if attention_type == "standard":
            return MultiHeadAttention(d_model, num_heads, **kwargs)
        elif attention_type == "local":
            window_size = kwargs.get("window_size", 128)
            return LocalAttention(d_model, num_heads, window_size, **kwargs)
        elif attention_type == "sparse":
            num_landmarks = kwargs.get("num_landmarks", 64)
            return SparseAttention(d_model, num_heads, num_landmarks, **kwargs)
        elif attention_type == "relative":
            max_relative_position = kwargs.get("max_relative_position", 32)
            return AttentionWithRelativePositions(d_model, num_heads, max_relative_position, **kwargs)
        else:
            raise ValueError(f"Unsupported attention type: {attention_type}")

class PositionalEncodingFactory:
    """
    Factory class for creating different types of positional encodings
    """
    
    @staticmethod
    def create_positional_encoding(encoding_type: str, d_model: int, 
                                 max_len: int = 5000, **kwargs) -> nn.Module:
        """
        Create positional encoding based on type
        
        Args:
            encoding_type: Type of positional encoding
            d_model: Model dimension
            max_len: Maximum sequence length
            **kwargs: Additional arguments
            
        Returns:
            Positional encoding module
        """
        if encoding_type == "sinusoidal":
            return PositionalEncoding(d_model, max_len, **kwargs)
        elif encoding_type == "learned":
            return LearnedPositionalEncoding(d_model, max_len, **kwargs)
        elif encoding_type == "relative":
            max_relative_position = kwargs.get("max_relative_position", 32)
            return RelativePositionalEncoding(d_model, max_relative_position, **kwargs)
        elif encoding_type == "rotary":
            return RotaryPositionalEncoding(d_model, max_len, **kwargs)
        else:
            raise ValueError(f"Unsupported positional encoding type: {encoding_type}")

def create_attention_mask(seq_len: int, device: torch.device, 
                         causal: bool = False) -> torch.Tensor:
    """
    Create attention mask
    
    Args:
        seq_len: Sequence length
        device: Device to create tensor on
        causal: Whether to create causal mask
        
    Returns:
        Attention mask tensor
    """
    if causal:
        # Create causal mask (lower triangular)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    else:
        # Create full attention mask
        return torch.ones(seq_len, seq_len, device=device)

def create_padding_mask(padding_mask: torch.Tensor, 
                       seq_len: int) -> torch.Tensor:
    """
    Create padding mask for attention
    
    Args:
        padding_mask: Boolean padding mask [batch_size, seq_len]
        seq_len: Sequence length
        
    Returns:
        Attention padding mask [batch_size, 1, 1, seq_len]
    """
    # Expand padding mask for attention
    attention_mask = padding_mask.unsqueeze(1).unsqueeze(2)
    return attention_mask 