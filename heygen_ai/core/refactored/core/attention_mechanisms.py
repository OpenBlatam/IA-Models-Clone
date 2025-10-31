"""
Advanced Attention Mechanisms

This module contains various advanced attention mechanisms
for the refactored transformer architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict, Any, List
from ..base import BaseAttentionMechanism
from ...transformer_config import TransformerConfig


class SparseAttention(BaseAttentionMechanism):
    """Sparse attention mechanism with configurable sparsity patterns."""
    
    def __init__(self, config: TransformerConfig, sparsity_pattern: str = "strided"):
        super().__init__(config)
        self.sparsity_pattern = sparsity_pattern
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
    
    def _create_sparse_mask(self, seq_len: int, batch_size: int) -> torch.Tensor:
        """Create sparse attention mask."""
        mask = torch.zeros(batch_size, self.num_heads, seq_len, seq_len, device=self.device)
        
        if self.sparsity_pattern == "strided":
            # Strided pattern
            stride = max(1, seq_len // 8)
            for i in range(0, seq_len, stride):
                mask[:, :, i, :] = 1
                mask[:, :, :, i] = 1
        elif self.sparsity_pattern == "local":
            # Local attention
            window_size = min(64, seq_len)
            for i in range(seq_len):
                start = max(0, i - window_size // 2)
                end = min(seq_len, i + window_size // 2)
                mask[:, :, i, start:end] = 1
        elif self.sparsity_pattern == "random":
            # Random sparse pattern
            sparsity = 0.1
            random_mask = torch.rand(batch_size, self.num_heads, seq_len, seq_len) < sparsity
            mask = random_mask.float()
        
        return mask
    
    def forward(self, 
                query: torch.Tensor, 
                key: torch.Tensor, 
                value: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sparse attention forward pass."""
        batch_size, seq_len, _ = query.size()
        
        # Project to query, key, value
        q = self.query_projection(query)
        k = self.key_projection(key)
        v = self.value_projection(value)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Create sparse mask
        sparse_mask = self._create_sparse_mask(seq_len, batch_size)
        
        # Apply masks
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
            scores = scores.masked_fill(attention_mask == 0, -1e9)
        
        scores = scores.masked_fill(sparse_mask == 0, -1e9)
        
        # Apply softmax
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, v)
        
        # Reshape back
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        
        # Output projection
        context = self.output_projection(context)
        
        return context, attn_weights


class LinearAttention(BaseAttentionMechanism):
    """Linear attention mechanism with O(n) complexity."""
    
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
        
        # Feature maps for linear attention
        self.feature_map = nn.Sequential(
            nn.Linear(self.head_dim, self.head_dim * 2),
            nn.ReLU(),
            nn.Linear(self.head_dim * 2, self.head_dim)
        )
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, 
                query: torch.Tensor, 
                key: torch.Tensor, 
                value: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Linear attention forward pass."""
        batch_size, seq_len, _ = query.size()
        
        # Project to query, key, value
        q = self.query_projection(query)
        k = self.key_projection(key)
        v = self.value_projection(value)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply feature map
        q = self.feature_map(q)
        k = self.feature_map(k)
        
        # Linear attention computation
        # QK^T = Q @ K^T
        qk = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply attention mask
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
            qk = qk.masked_fill(attention_mask == 0, 0)
        
        # Normalize
        qk_sum = qk.sum(dim=-1, keepdim=True)
        qk_normalized = qk / (qk_sum + 1e-8)
        
        # Apply to values
        context = torch.matmul(qk_normalized, v)
        
        # Reshape back
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        
        # Output projection
        context = self.output_projection(context)
        
        # Create dummy attention weights for compatibility
        attn_weights = qk_normalized
        
        return context, attn_weights


class AdaptiveAttention(BaseAttentionMechanism):
    """Adaptive attention mechanism that adjusts based on input."""
    
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
        
        # Adaptive components
        self.attention_gate = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 4),
            nn.ReLU(),
            nn.Linear(config.hidden_size // 4, self.num_heads),
            nn.Sigmoid()
        )
        
        self.temperature_network = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 4),
            nn.ReLU(),
            nn.Linear(config.hidden_size // 4, 1),
            nn.Softplus()
        )
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, 
                query: torch.Tensor, 
                key: torch.Tensor, 
                value: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Adaptive attention forward pass."""
        batch_size, seq_len, _ = query.size()
        
        # Project to query, key, value
        q = self.query_projection(query)
        k = self.key_projection(key)
        v = self.value_projection(value)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Adaptive temperature
        temperature = self.temperature_network(query.mean(dim=1))  # [batch_size, 1]
        temperature = temperature.unsqueeze(1).unsqueeze(1)  # [batch_size, 1, 1, 1]
        scores = scores / temperature
        
        # Adaptive gating
        attention_gates = self.attention_gate(query)  # [batch_size, seq_len, num_heads]
        attention_gates = attention_gates.transpose(1, 2).unsqueeze(-1)  # [batch_size, num_heads, seq_len, 1]
        scores = scores * attention_gates
        
        # Apply attention mask
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
            scores = scores.masked_fill(attention_mask == 0, -1e9)
        
        # Apply softmax
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, v)
        
        # Reshape back
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        
        # Output projection
        context = self.output_projection(context)
        
        return context, attn_weights


class CausalAttention(BaseAttentionMechanism):
    """Causal attention mechanism for autoregressive generation."""
    
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
    
    def _create_causal_mask(self, seq_len: int, batch_size: int) -> torch.Tensor:
        """Create causal attention mask."""
        mask = torch.tril(torch.ones(seq_len, seq_len, device=self.device))
        mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
        mask = mask.expand(batch_size, self.num_heads, -1, -1)
        return mask
    
    def forward(self, 
                query: torch.Tensor, 
                key: torch.Tensor, 
                value: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Causal attention forward pass."""
        batch_size, seq_len, _ = query.size()
        
        # Project to query, key, value
        q = self.query_projection(query)
        k = self.key_projection(key)
        v = self.value_projection(value)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Create causal mask
        causal_mask = self._create_causal_mask(seq_len, batch_size)
        scores = scores.masked_fill(causal_mask == 0, -1e9)
        
        # Apply additional attention mask
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
            scores = scores.masked_fill(attention_mask == 0, -1e9)
        
        # Apply softmax
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, v)
        
        # Reshape back
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        
        # Output projection
        context = self.output_projection(context)
        
        return context, attn_weights

