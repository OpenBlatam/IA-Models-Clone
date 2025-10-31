"""
Core Transformer Components

This module contains the core transformer implementation with
enhanced modularity and extensibility.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, List
from ..base import BaseTransformerComponent, BaseAttentionMechanism, BaseTransformerBlock
from ...transformer_config import TransformerConfig


class EnhancedMultiHeadAttention(BaseAttentionMechanism):
    """Enhanced multi-head attention mechanism with advanced features."""
    
    def __init__(self, config: TransformerConfig):
        super().__init__(config)
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scale = 1.0 / (self.head_dim ** 0.5)
        
        # Linear projections
        self.query_projection = nn.Linear(config.hidden_size, config.hidden_size)
        self.key_projection = nn.Linear(config.hidden_size, config.hidden_size)
        self.value_projection = nn.Linear(config.hidden_size, config.hidden_size)
        self.output_projection = nn.Linear(config.hidden_size, config.hidden_size)
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
        
        # Attention enhancement parameters
        self.attention_temperature = nn.Parameter(torch.tensor(1.0))
        self.attention_bias = nn.Parameter(torch.zeros(1, 1, 1, self.head_dim))
    
    def forward(self, 
                query: torch.Tensor, 
                key: torch.Tensor, 
                value: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Enhanced attention forward pass."""
        batch_size, seq_len, _ = query.size()
        
        # Project to query, key, value
        q = self.query_projection(query)
        k = self.key_projection(key)
        v = self.value_projection(value)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores with temperature scaling
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale / self.attention_temperature
        
        # Add attention bias
        scores = scores + self.attention_bias
        
        # Apply attention mask
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, -1e9)
        
        # Apply softmax with dropout
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, v)
        
        # Reshape back
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        
        # Output projection
        context = self.output_projection(context)
        
        return context, attn_weights


class EnhancedFeedForwardNetwork(nn.Module):
    """Enhanced feed-forward network with advanced features."""
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        
        # Main feed-forward layers
        self.linear1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.linear2 = nn.Linear(config.intermediate_size, config.hidden_size)
        
        # Activation functions
        self.activation = nn.GELU()
        self.activation_dropout = nn.Dropout(config.dropout)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
        
        # Enhancement parameters
        self.gate_linear = nn.Linear(config.hidden_size, config.intermediate_size)
        self.gate_activation = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Enhanced feed-forward forward pass."""
        # Gated activation
        gate = self.gate_activation(self.gate_linear(x))
        
        # Main feed-forward
        hidden = self.linear1(x)
        hidden = self.activation(hidden)
        hidden = self.activation_dropout(hidden)
        
        # Apply gate
        hidden = hidden * gate
        
        # Output projection
        output = self.linear2(hidden)
        output = self.dropout(output)
        
        return output


class EnhancedTransformerBlock(BaseTransformerBlock):
    """Enhanced transformer block with advanced features."""
    
    def __init__(self, config: TransformerConfig):
        super().__init__(config)
        self.attention = EnhancedMultiHeadAttention(config)
        self.ffn = EnhancedFeedForwardNetwork(config)
        
        # Layer normalization
        self.attention_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.ffn_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
        
        # Residual scaling
        self.residual_scaling = nn.Parameter(torch.ones(1))
    
    def forward(self, 
                x: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Enhanced transformer block forward pass."""
        # Self-attention with residual connection
        attn_output, attn_weights = self.attention(x, x, x, attention_mask)
        x = self.attention_norm(x + self.dropout(attn_output) * self.residual_scaling)
        
        # Feed-forward with residual connection
        ffn_output = self.ffn(x)
        x = self.ffn_norm(x + ffn_output * self.residual_scaling)
        
        return x, attn_weights


class EnhancedTransformerModel(nn.Module):
    """Enhanced transformer model with advanced features."""
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_layers = config.num_layers
        self.vocab_size = config.vocab_size
        
        # Embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embedding = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            EnhancedTransformerBlock(config) for _ in range(config.num_layers)
        ])
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Output head
        self.output_head = nn.Linear(config.hidden_size, config.vocab_size)
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, 
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Enhanced transformer forward pass."""
        batch_size, seq_len = input_ids.size()
        
        # Create position ids if not provided
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        
        # Token embeddings
        token_embeddings = self.token_embedding(input_ids)
        
        # Position embeddings
        position_embeddings = self.position_embedding(position_ids)
        
        # Combine embeddings
        embeddings = token_embeddings + position_embeddings
        embeddings = self.dropout(embeddings)
        
        # Pass through transformer blocks
        hidden_states = embeddings
        attention_weights = []
        
        for block in self.transformer_blocks:
            hidden_states, attn_weights = block(hidden_states, attention_mask)
            attention_weights.append(attn_weights)
        
        # Final layer normalization
        hidden_states = self.layer_norm(hidden_states)
        
        # Output logits
        logits = self.output_head(hidden_states)
        
        return {
            'logits': logits,
            'hidden_states': hidden_states,
            'attention_weights': attention_weights
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'vocab_size': self.vocab_size,
            'memory_mb': self._estimate_memory_usage()
        }
    
    def _estimate_memory_usage(self) -> float:
        """Estimate memory usage in MB."""
        param_size = sum(p.numel() * p.element_size() for p in self.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.buffers())
        return (param_size + buffer_size) / 1024 / 1024

