"""
Attention Layers for TruthGPT API
=================================

TensorFlow-like attention layer implementations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Union, Tuple


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention layer.
    
    Similar to tf.keras.layers.MultiHeadAttention, this layer
    implements multi-head self-attention mechanism.
    """
    
    def __init__(self, 
                 num_heads: int = 8,
                 key_dim: int = 64,
                 value_dim: Optional[int] = None,
                 dropout: float = 0.0,
                 use_bias: bool = True,
                 output_shape: Optional[Tuple[int, ...]] = None,
                 attention_axes: Optional[Tuple[int, ...]] = None,
                 kernel_initializer: str = 'glorot_uniform',
                 bias_initializer: str = 'zeros',
                 name: Optional[str] = None):
        """
        Initialize MultiHeadAttention layer.
        
        Args:
            num_heads: Number of attention heads
            key_dim: Dimension of key/query vectors
            value_dim: Dimension of value vectors
            dropout: Dropout rate
            use_bias: Whether to use bias
            output_shape: Output shape
            attention_axes: Axes to apply attention
            kernel_initializer: Weight initialization method
            bias_initializer: Bias initialization method
            name: Optional name for the layer
        """
        super().__init__()
        
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.value_dim = value_dim or key_dim
        self.dropout = dropout
        self.use_bias = use_bias
        self.output_shape = output_shape
        self.attention_axes = attention_axes
        self.name = name or f"multi_head_attention_{num_heads}"
        
        # Create PyTorch multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=key_dim * num_heads,
            num_heads=num_heads,
            dropout=dropout,
            bias=use_bias,
            batch_first=True
        )
        
        # Linear projections
        self.query_projection = nn.Linear(key_dim, key_dim * num_heads, bias=use_bias)
        self.key_projection = nn.Linear(key_dim, key_dim * num_heads, bias=use_bias)
        self.value_projection = nn.Linear(self.value_dim, self.value_dim * num_heads, bias=use_bias)
        self.output_projection = nn.Linear(self.value_dim * num_heads, key_dim, bias=use_bias)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize layer weights."""
        if hasattr(self, 'query_projection'):
            nn.init.xavier_uniform_(self.query_projection.weight)
            if self.use_bias:
                nn.init.zeros_(self.query_projection.bias)
        
        if hasattr(self, 'key_projection'):
            nn.init.xavier_uniform_(self.key_projection.weight)
            if self.use_bias:
                nn.init.zeros_(self.key_projection.bias)
        
        if hasattr(self, 'value_projection'):
            nn.init.xavier_uniform_(self.value_projection.weight)
            if self.use_bias:
                nn.init.zeros_(self.value_projection.bias)
        
        if hasattr(self, 'output_projection'):
            nn.init.xavier_uniform_(self.output_projection.weight)
            if self.use_bias:
                nn.init.zeros_(self.output_projection.bias)
    
    def call(self, 
             query: torch.Tensor,
             value: torch.Tensor,
             key: Optional[torch.Tensor] = None,
             attention_mask: Optional[torch.Tensor] = None,
             training: bool = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            query: Query tensor
            value: Value tensor
            key: Key tensor (optional)
            attention_mask: Attention mask
            training: Whether in training mode
            
        Returns:
            Tuple of (output, attention_weights)
        """
        if key is None:
            key = value
        
        # Project inputs
        query_proj = self.query_projection(query)
        key_proj = self.key_projection(key)
        value_proj = self.value_projection(value)
        
        # Reshape for multi-head attention
        batch_size, seq_len, _ = query.shape
        query_proj = query_proj.view(batch_size, seq_len, self.num_heads, self.key_dim)
        key_proj = key_proj.view(batch_size, -1, self.num_heads, self.key_dim)
        value_proj = value_proj.view(batch_size, -1, self.num_heads, self.value_dim)
        
        # Transpose for attention computation
        query_proj = query_proj.transpose(1, 2)  # (batch, heads, seq_len, key_dim)
        key_proj = key_proj.transpose(1, 2)      # (batch, heads, seq_len, key_dim)
        value_proj = value_proj.transpose(1, 2)  # (batch, heads, seq_len, value_dim)
        
        # Compute attention
        attention_output, attention_weights = self._compute_attention(
            query_proj, key_proj, value_proj, attention_mask
        )
        
        # Reshape and project output
        attention_output = attention_output.transpose(1, 2)  # (batch, seq_len, heads, value_dim)
        attention_output = attention_output.contiguous().view(
            batch_size, seq_len, self.value_dim * self.num_heads
        )
        output = self.output_projection(attention_output)
        
        return output, attention_weights
    
    def _compute_attention(self, 
                          query: torch.Tensor,
                          key: torch.Tensor,
                          value: torch.Tensor,
                          attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute attention weights and output."""
        # Compute attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.key_dim)
        
        # Apply attention mask
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, -1e9)
        
        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)
        
        # Apply dropout
        if self.dropout > 0 and self.training:
            attention_weights = F.dropout(attention_weights, p=self.dropout, training=self.training)
        
        # Compute weighted sum
        output = torch.matmul(attention_weights, value)
        
        return output, attention_weights
    
    def forward(self, 
                query: torch.Tensor,
                value: torch.Tensor,
                key: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """PyTorch forward method."""
        return self.call(query, value, key, attention_mask, training=self.training)
    
    def __repr__(self):
        return f"MultiHeadAttention(num_heads={self.num_heads}, key_dim={self.key_dim}, value_dim={self.value_dim})"


class SelfAttention(nn.Module):
    """
    Self-attention layer.
    
    Similar to tf.keras.layers.SelfAttention, this layer
    implements self-attention mechanism.
    """
    
    def __init__(self, 
                 attention_axes: Tuple[int, ...] = (1,),
                 dropout: float = 0.0,
                 use_bias: bool = True,
                 kernel_initializer: str = 'glorot_uniform',
                 bias_initializer: str = 'zeros',
                 name: Optional[str] = None):
        """
        Initialize SelfAttention layer.
        
        Args:
            attention_axes: Axes to apply attention
            dropout: Dropout rate
            use_bias: Whether to use bias
            kernel_initializer: Weight initialization method
            bias_initializer: Bias initialization method
            name: Optional name for the layer
        """
        super().__init__()
        
        self.attention_axes = attention_axes
        self.dropout = dropout
        self.use_bias = use_bias
        self.name = name or "self_attention"
        
        # Linear projections for query, key, value
        self.query_projection = None
        self.key_projection = None
        self.value_projection = None
        self.output_projection = None
        
        self._built = False
    
    def build(self, input_shape: tuple):
        """Build the layer with given input shape."""
        if self._built:
            return
        
        input_dim = input_shape[-1]
        
        # Create linear projections
        self.query_projection = nn.Linear(input_dim, input_dim, bias=self.use_bias)
        self.key_projection = nn.Linear(input_dim, input_dim, bias=self.use_bias)
        self.value_projection = nn.Linear(input_dim, input_dim, bias=self.use_bias)
        self.output_projection = nn.Linear(input_dim, input_dim, bias=self.use_bias)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.query_projection.weight)
        nn.init.xavier_uniform_(self.key_projection.weight)
        nn.init.xavier_uniform_(self.value_projection.weight)
        nn.init.xavier_uniform_(self.output_projection.weight)
        
        if self.use_bias:
            nn.init.zeros_(self.query_projection.bias)
            nn.init.zeros_(self.key_projection.bias)
            nn.init.zeros_(self.value_projection.bias)
            nn.init.zeros_(self.output_projection.bias)
        
        self._built = True
    
    def call(self, 
             inputs: torch.Tensor,
             attention_mask: Optional[torch.Tensor] = None,
             training: bool = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            inputs: Input tensor
            attention_mask: Attention mask
            training: Whether in training mode
            
        Returns:
            Tuple of (output, attention_weights)
        """
        if not self._built:
            self.build(inputs.shape)
        
        # Project inputs
        query = self.query_projection(inputs)
        key = self.key_projection(inputs)
        value = self.value_projection(inputs)
        
        # Compute attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(inputs.size(-1))
        
        # Apply attention mask
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, -1e9)
        
        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)
        
        # Apply dropout
        if self.dropout > 0 and self.training:
            attention_weights = F.dropout(attention_weights, p=self.dropout, training=self.training)
        
        # Compute weighted sum
        output = torch.matmul(attention_weights, value)
        
        # Project output
        output = self.output_projection(output)
        
        return output, attention_weights
    
    def forward(self, 
                inputs: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """PyTorch forward method."""
        return self.call(inputs, attention_mask, training=self.training)
    
    def __repr__(self):
        return f"SelfAttention(attention_axes={self.attention_axes}, dropout={self.dropout})"









