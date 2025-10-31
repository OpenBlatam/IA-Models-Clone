"""
Transformer Layers for TruthGPT API
===================================

TensorFlow-like transformer layer implementations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Union, Tuple
from .attention import MultiHeadAttention


class TransformerEncoder(nn.Module):
    """
    Transformer encoder layer.
    
    Similar to tf.keras.layers.TransformerEncoder, this layer
    implements a transformer encoder block.
    """
    
    def __init__(self, 
                 num_heads: int = 8,
                 intermediate_dim: int = 2048,
                 dropout: float = 0.1,
                 activation: str = 'relu',
                 layer_norm_epsilon: float = 1e-5,
                 name: Optional[str] = None):
        """
        Initialize TransformerEncoder layer.
        
        Args:
            num_heads: Number of attention heads
            intermediate_dim: Dimension of intermediate layer
            dropout: Dropout rate
            activation: Activation function
            layer_norm_epsilon: Layer normalization epsilon
            name: Optional name for the layer
        """
        super().__init__()
        
        self.num_heads = num_heads
        self.intermediate_dim = intermediate_dim
        self.dropout = dropout
        self.activation = activation
        self.layer_norm_epsilon = layer_norm_epsilon
        self.name = name or f"transformer_encoder_{num_heads}"
        
        # Multi-head attention
        self.attention = MultiHeadAttention(
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Feed-forward network
        self.intermediate_dense = nn.Linear(1, intermediate_dim)  # Will be set in build
        self.output_dense = nn.Linear(intermediate_dim, 1)  # Will be set in build
        
        # Layer normalization
        self.attention_layer_norm = nn.LayerNorm(1, eps=layer_norm_epsilon)
        self.output_layer_norm = nn.LayerNorm(1, eps=layer_norm_epsilon)
        
        # Dropout
        self.attention_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)
        
        self._built = False
    
    def build(self, input_shape: tuple):
        """Build the layer with given input shape."""
        if self._built:
            return
        
        input_dim = input_shape[-1]
        
        # Update linear layers
        self.intermediate_dense = nn.Linear(input_dim, self.intermediate_dim)
        self.output_dense = nn.Linear(self.intermediate_dim, input_dim)
        
        # Update layer normalization
        self.attention_layer_norm = nn.LayerNorm(input_dim, eps=self.layer_norm_epsilon)
        self.output_layer_norm = nn.LayerNorm(input_dim, eps=self.layer_norm_epsilon)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.intermediate_dense.weight)
        nn.init.xavier_uniform_(self.output_dense.weight)
        nn.init.zeros_(self.intermediate_dense.bias)
        nn.init.zeros_(self.output_dense.bias)
        
        self._built = True
    
    def call(self, 
             inputs: torch.Tensor,
             attention_mask: Optional[torch.Tensor] = None,
             training: bool = None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            inputs: Input tensor
            attention_mask: Attention mask
            training: Whether in training mode
            
        Returns:
            Output tensor
        """
        if not self._built:
            self.build(inputs.shape)
        
        # Self-attention
        attention_output, attention_weights = self.attention(
            query=inputs,
            value=inputs,
            key=inputs,
            attention_mask=attention_mask,
            training=training
        )
        
        # Add & Norm (attention)
        attention_output = self.attention_dropout(attention_output)
        attention_output = self.attention_layer_norm(inputs + attention_output)
        
        # Feed-forward network
        intermediate_output = self.intermediate_dense(attention_output)
        if self.activation == 'relu':
            intermediate_output = F.relu(intermediate_output)
        elif self.activation == 'gelu':
            intermediate_output = F.gelu(intermediate_output)
        
        output = self.output_dense(intermediate_output)
        
        # Add & Norm (output)
        output = self.output_dropout(output)
        output = self.output_layer_norm(attention_output + output)
        
        return output
    
    def forward(self, 
                inputs: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """PyTorch forward method."""
        return self.call(inputs, attention_mask, training=self.training)
    
    def __repr__(self):
        return f"TransformerEncoder(num_heads={self.num_heads}, intermediate_dim={self.intermediate_dim})"


class TransformerDecoder(nn.Module):
    """
    Transformer decoder layer.
    
    Similar to tf.keras.layers.TransformerDecoder, this layer
    implements a transformer decoder block.
    """
    
    def __init__(self, 
                 num_heads: int = 8,
                 intermediate_dim: int = 2048,
                 dropout: float = 0.1,
                 activation: str = 'relu',
                 layer_norm_epsilon: float = 1e-5,
                 name: Optional[str] = None):
        """
        Initialize TransformerDecoder layer.
        
        Args:
            num_heads: Number of attention heads
            intermediate_dim: Dimension of intermediate layer
            dropout: Dropout rate
            activation: Activation function
            layer_norm_epsilon: Layer normalization epsilon
            name: Optional name for the layer
        """
        super().__init__()
        
        self.num_heads = num_heads
        self.intermediate_dim = intermediate_dim
        self.dropout = dropout
        self.activation = activation
        self.layer_norm_epsilon = layer_norm_epsilon
        self.name = name or f"transformer_decoder_{num_heads}"
        
        # Self-attention
        self.self_attention = MultiHeadAttention(
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Cross-attention
        self.cross_attention = MultiHeadAttention(
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Feed-forward network
        self.intermediate_dense = nn.Linear(1, intermediate_dim)  # Will be set in build
        self.output_dense = nn.Linear(intermediate_dim, 1)  # Will be set in build
        
        # Layer normalization
        self.self_attention_layer_norm = nn.LayerNorm(1, eps=layer_norm_epsilon)
        self.cross_attention_layer_norm = nn.LayerNorm(1, eps=layer_norm_epsilon)
        self.output_layer_norm = nn.LayerNorm(1, eps=layer_norm_epsilon)
        
        # Dropout
        self.self_attention_dropout = nn.Dropout(dropout)
        self.cross_attention_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)
        
        self._built = False
    
    def build(self, input_shape: tuple):
        """Build the layer with given input shape."""
        if self._built:
            return
        
        input_dim = input_shape[-1]
        
        # Update linear layers
        self.intermediate_dense = nn.Linear(input_dim, self.intermediate_dim)
        self.output_dense = nn.Linear(self.intermediate_dim, input_dim)
        
        # Update layer normalization
        self.self_attention_layer_norm = nn.LayerNorm(input_dim, eps=self.layer_norm_epsilon)
        self.cross_attention_layer_norm = nn.LayerNorm(input_dim, eps=self.layer_norm_epsilon)
        self.output_layer_norm = nn.LayerNorm(input_dim, eps=self.layer_norm_epsilon)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.intermediate_dense.weight)
        nn.init.xavier_uniform_(self.output_dense.weight)
        nn.init.zeros_(self.intermediate_dense.bias)
        nn.init.zeros_(self.output_dense.bias)
        
        self._built = True
    
    def call(self, 
             inputs: torch.Tensor,
             encoder_outputs: torch.Tensor,
             self_attention_mask: Optional[torch.Tensor] = None,
             cross_attention_mask: Optional[torch.Tensor] = None,
             training: bool = None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            inputs: Input tensor
            encoder_outputs: Encoder outputs
            self_attention_mask: Self-attention mask
            cross_attention_mask: Cross-attention mask
            training: Whether in training mode
            
        Returns:
            Output tensor
        """
        if not self._built:
            self.build(inputs.shape)
        
        # Self-attention
        self_attention_output, _ = self.self_attention(
            query=inputs,
            value=inputs,
            key=inputs,
            attention_mask=self_attention_mask,
            training=training
        )
        
        # Add & Norm (self-attention)
        self_attention_output = self.self_attention_dropout(self_attention_output)
        self_attention_output = self.self_attention_layer_norm(inputs + self_attention_output)
        
        # Cross-attention
        cross_attention_output, _ = self.cross_attention(
            query=self_attention_output,
            value=encoder_outputs,
            key=encoder_outputs,
            attention_mask=cross_attention_mask,
            training=training
        )
        
        # Add & Norm (cross-attention)
        cross_attention_output = self.cross_attention_dropout(cross_attention_output)
        cross_attention_output = self.cross_attention_layer_norm(
            self_attention_output + cross_attention_output
        )
        
        # Feed-forward network
        intermediate_output = self.intermediate_dense(cross_attention_output)
        if self.activation == 'relu':
            intermediate_output = F.relu(intermediate_output)
        elif self.activation == 'gelu':
            intermediate_output = F.gelu(intermediate_output)
        
        output = self.output_dense(intermediate_output)
        
        # Add & Norm (output)
        output = self.output_dropout(output)
        output = self.output_layer_norm(cross_attention_output + output)
        
        return output
    
    def forward(self, 
                inputs: torch.Tensor,
                encoder_outputs: torch.Tensor,
                self_attention_mask: Optional[torch.Tensor] = None,
                cross_attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """PyTorch forward method."""
        return self.call(inputs, encoder_outputs, self_attention_mask, cross_attention_mask, training=self.training)
    
    def __repr__(self):
        return f"TransformerDecoder(num_heads={self.num_heads}, intermediate_dim={self.intermediate_dim})"


class PositionalEncoding(nn.Module):
    """
    Positional encoding layer.
    
    Similar to tf.keras.layers.PositionalEncoding, this layer
    adds positional encoding to input sequences.
    """
    
    def __init__(self, 
                 max_length: int = 5000,
                 d_model: int = 512,
                 name: Optional[str] = None):
        """
        Initialize PositionalEncoding layer.
        
        Args:
            max_length: Maximum sequence length
            d_model: Model dimension
            name: Optional name for the layer
        """
        super().__init__()
        
        self.max_length = max_length
        self.d_model = d_model
        self.name = name or f"positional_encoding_{max_length}"
        
        # Create positional encoding
        pe = torch.zeros(max_length, d_model)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def call(self, inputs: torch.Tensor, training: bool = None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            inputs: Input tensor
            training: Whether in training mode
            
        Returns:
            Output tensor with positional encoding
        """
        seq_len = inputs.size(1)
        return inputs + self.pe[:, :seq_len]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """PyTorch forward method."""
        return self.call(x, training=self.training)
    
    def __repr__(self):
        return f"PositionalEncoding(max_length={self.max_length}, d_model={self.d_model})"


