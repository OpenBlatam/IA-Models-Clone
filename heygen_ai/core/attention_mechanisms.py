"""
Enhanced Attention Mechanisms Module

This module contains various advanced attention mechanisms including
sparse attention, linear attention, memory-efficient attention, and more.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict, Any, List
from .transformer_config import TransformerConfig


class MultiHeadAttention(nn.Module):
    """Standard multi-head attention mechanism."""
    
    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        # Linear projections
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, 
                query: torch.Tensor, 
                key: torch.Tensor, 
                value: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of multi-head attention."""
        batch_size, seq_len = query.size(0), query.size(1)
        
        # Linear projections
        Q = self.q_proj(query).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        # Apply attention mask
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask.unsqueeze(1) == 0, -1e9)
        
        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, V)
        
        # Reshape and project
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.hidden_size
        )
        
        output = self.out_proj(context)
        
        return output, attn_weights


class SparseAttention(nn.Module):
    """Sparse attention mechanism for efficient computation."""
    
    def __init__(self, 
                 hidden_size: int, 
                 num_heads: int, 
                 attention_type: str = "strided",
                 local_window_size: int = 64,
                 global_window_size: int = 256):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.attention_type = attention_type
        self.local_window_size = local_window_size
        self.global_window_size = global_window_size
        
        # Linear projections
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        
        self.dropout = nn.Dropout(0.1)
    
    def _create_attention_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create sparse attention mask."""
        if self.attention_type == "strided":
            # Strided attention pattern
            mask = torch.zeros(seq_len, seq_len, device=device)
            for i in range(seq_len):
                # Local attention
                start = max(0, i - self.local_window_size // 2)
                end = min(seq_len, i + self.local_window_size // 2)
                mask[i, start:end] = 1
                
                # Strided attention
                stride = max(1, seq_len // self.global_window_size)
                mask[i, ::stride] = 1
                
        elif self.attention_type == "local_global":
            # Local + global attention pattern
            mask = torch.zeros(seq_len, seq_len, device=device)
            for i in range(seq_len):
                # Local attention
                start = max(0, i - self.local_window_size // 2)
                end = min(seq_len, i + self.local_window_size // 2)
                mask[i, start:end] = 1
                
                # Global attention to first and last tokens
                mask[i, 0] = 1
                mask[i, -1] = 1
                
        else:
            # Full attention
            mask = torch.ones(seq_len, seq_len, device=device)
        
        return mask
    
    def forward(self, 
                query: torch.Tensor, 
                key: torch.Tensor, 
                value: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of sparse attention."""
        batch_size, seq_len = query.size(0), query.size(1)
        
        # Linear projections
        Q = self.q_proj(query).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        # Create sparse attention mask
        sparse_mask = self._create_attention_mask(seq_len, query.device)
        sparse_mask = sparse_mask.unsqueeze(0).unsqueeze(0)  # Add batch and head dimensions
        
        # Apply masks
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)  # Add head dimension
            combined_mask = sparse_mask & attention_mask
        else:
            combined_mask = sparse_mask
        
        scores = scores.masked_fill(combined_mask == 0, -1e9)
        
        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, V)
        
        # Reshape and project
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.hidden_size
        )
        
        output = self.out_proj(context)
        
        return output, attn_weights


class LinearAttention(nn.Module):
    """Linear attention mechanism with O(n) complexity."""
    
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        # Linear projections
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        
        # Feature maps for linear attention
        self.feature_map = nn.Sequential(
            nn.Linear(self.head_dim, self.head_dim * 2),
            nn.ReLU(),
            nn.Linear(self.head_dim * 2, self.head_dim)
        )
        
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, 
                query: torch.Tensor, 
                key: torch.Tensor, 
                value: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of linear attention."""
        batch_size, seq_len = query.size(0), query.size(1)
        
        # Linear projections
        Q = self.q_proj(query).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply feature map
        Q = self.feature_map(Q)
        K = self.feature_map(K)
        
        # Linear attention computation
        # Q: [batch, heads, seq_len, head_dim]
        # K: [batch, heads, seq_len, head_dim]
        # V: [batch, heads, seq_len, head_dim]
        
        # Compute KV^T
        KV = torch.matmul(K.transpose(-2, -1), V)  # [batch, heads, head_dim, head_dim]
        
        # Compute QKV
        QKV = torch.matmul(Q, KV)  # [batch, heads, seq_len, head_dim]
        
        # Normalize by sum of K
        K_sum = torch.sum(K, dim=-2, keepdim=True)  # [batch, heads, 1, head_dim]
        QKV = QKV / (torch.sum(K_sum, dim=-1, keepdim=True) + 1e-8)
        
        # Reshape and project
        context = QKV.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.hidden_size
        )
        
        output = self.out_proj(context)
        
        # Create dummy attention weights for compatibility
        attn_weights = torch.ones(batch_size, self.num_heads, seq_len, seq_len, device=query.device)
        attn_weights = attn_weights / seq_len
        
        return output, attn_weights


class MemoryEfficientAttention(nn.Module):
    """Memory-efficient attention with gradient checkpointing."""
    
    def __init__(self, hidden_size: int, num_heads: int, chunk_size: int = 64):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.chunk_size = chunk_size
        
        # Linear projections
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        
        self.dropout = nn.Dropout(0.1)
    
    def _attention_chunk(self, 
                        Q: torch.Tensor, 
                        K: torch.Tensor, 
                        V: torch.Tensor,
                        attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Process attention in chunks for memory efficiency."""
        batch_size, num_heads, seq_len, head_dim = Q.size()
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        # Apply attention mask
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, -1e9)
        
        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, V)
        
        return context
    
    def forward(self, 
                query: torch.Tensor, 
                key: torch.Tensor, 
                value: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of memory-efficient attention."""
        batch_size, seq_len = query.size(0), query.size(1)
        
        # Linear projections
        Q = self.q_proj(query).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Process in chunks
        context_chunks = []
        for i in range(0, seq_len, self.chunk_size):
            end_i = min(i + self.chunk_size, seq_len)
            Q_chunk = Q[:, :, i:end_i, :]
            
            # Create attention mask for this chunk
            chunk_mask = None
            if attention_mask is not None:
                chunk_mask = attention_mask[:, i:end_i, :].unsqueeze(1)
            
            # Process chunk
            context_chunk = self._attention_chunk(Q_chunk, K, V, chunk_mask)
            context_chunks.append(context_chunk)
        
        # Concatenate chunks
        context = torch.cat(context_chunks, dim=2)
        
        # Reshape and project
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.hidden_size
        )
        
        output = self.out_proj(context)
        
        # Create dummy attention weights for compatibility
        attn_weights = torch.ones(batch_size, self.num_heads, seq_len, seq_len, device=query.device)
        attn_weights = attn_weights / seq_len
        
        return output, attn_weights


class AdaptiveAttention(nn.Module):
    """Adaptive attention mechanism that learns optimal attention patterns."""
    
    def __init__(self, hidden_size: int, num_heads: int, num_patterns: int = 8):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.num_patterns = num_patterns
        
        # Linear projections
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        
        # Pattern selection network
        self.pattern_selector = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, num_patterns),
            nn.Softmax(dim=-1)
        )
        
        # Learnable attention patterns
        self.attention_patterns = nn.Parameter(
            torch.randn(num_patterns, num_heads, 1, 1) * 0.1
        )
        
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, 
                query: torch.Tensor, 
                key: torch.Tensor, 
                value: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of adaptive attention."""
        batch_size, seq_len = query.size(0), query.size(1)
        
        # Linear projections
        Q = self.q_proj(query).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        # Select attention patterns
        pattern_weights = self.pattern_selector(query.mean(dim=1))  # [batch, num_patterns]
        
        # Apply patterns
        pattern_scores = torch.matmul(
            pattern_weights.unsqueeze(1).unsqueeze(1),  # [batch, 1, 1, num_patterns]
            self.attention_patterns.unsqueeze(0)  # [1, num_patterns, num_heads, 1, 1]
        ).squeeze(1)  # [batch, num_heads, 1, 1]
        
        scores = scores * pattern_scores
        
        # Apply attention mask
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask.unsqueeze(1) == 0, -1e9)
        
        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, V)
        
        # Reshape and project
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.hidden_size
        )
        
        output = self.out_proj(context)
        
        return output, attn_weights


class CausalAttention(nn.Module):
    """Causal attention mechanism for autoregressive models."""
    
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        # Linear projections
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, 
                query: torch.Tensor, 
                key: torch.Tensor, 
                value: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of causal attention."""
        batch_size, seq_len = query.size(0), query.size(1)
        
        # Linear projections
        Q = self.q_proj(query).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        # Create causal mask
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=query.device))
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # Add batch and head dimensions
        
        # Apply masks
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1)  # Add head dimension
            combined_mask = causal_mask & attention_mask
        else:
            combined_mask = causal_mask
        
        scores = scores.masked_fill(combined_mask == 0, -1e9)
        
        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, V)
        
        # Reshape and project
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.hidden_size
        )
        
        output = self.out_proj(context)
        
        return output, attn_weights


class SymbolicAttention(nn.Module):
    """Symbolic attention mechanism for symbolic reasoning."""
    
    def __init__(self, hidden_size: int, num_heads: int, num_symbols: int = 100):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.num_symbols = num_symbols
        
        # Linear projections
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        
        # Symbolic reasoning components
        self.symbol_embedding = nn.Embedding(num_symbols, hidden_size)
        self.symbol_selector = nn.Linear(hidden_size, num_symbols)
        self.symbol_combiner = nn.Linear(hidden_size * 2, hidden_size)
        
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, 
                query: torch.Tensor, 
                key: torch.Tensor, 
                value: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of symbolic attention."""
        batch_size, seq_len = query.size(0), query.size(1)
        
        # Linear projections
        Q = self.q_proj(query).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        # Apply attention mask
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask.unsqueeze(1) == 0, -1e9)
        
        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, V)
        
        # Symbolic reasoning
        symbol_logits = self.symbol_selector(query)  # [batch, seq_len, num_symbols]
        symbol_weights = F.softmax(symbol_logits, dim=-1)  # [batch, seq_len, num_symbols]
        
        # Get symbol embeddings
        symbol_indices = torch.argmax(symbol_weights, dim=-1)  # [batch, seq_len]
        symbol_embeddings = self.symbol_embedding(symbol_indices)  # [batch, seq_len, hidden_size]
        
        # Combine with attention output
        combined_context = torch.cat([context, symbol_embeddings], dim=-1)
        context = self.symbol_combiner(combined_context)
        
        # Reshape and project
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.hidden_size
        )
        
        output = self.out_proj(context)
        
        return output, attn_weights


