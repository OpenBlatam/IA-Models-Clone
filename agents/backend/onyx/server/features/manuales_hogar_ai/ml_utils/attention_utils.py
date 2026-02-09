"""
Attention Utils - Utilidades de Mecanismos de Atención
=======================================================

Utilidades para mecanismos de atención personalizados.
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention.
    """
    
    def __init__(self, dropout: float = 0.1):
        """
        Inicializar atención.
        
        Args:
            dropout: Dropout rate
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            query: Query [batch, seq_len, d_k]
            key: Key [batch, seq_len, d_k]
            value: Value [batch, seq_len, d_v]
            mask: Mask (opcional)
            
        Returns:
            Tupla (output, attention_weights)
        """
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        output = torch.matmul(attention_weights, value)
        
        return output, attention_weights


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention.
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        """
        Inicializar multi-head attention.
        
        Args:
            d_model: Dimensión del modelo
            num_heads: Número de heads
            dropout: Dropout rate
        """
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.attention = ScaledDotProductAttention(dropout)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            query: Query
            key: Key
            value: Value
            mask: Mask (opcional)
            
        Returns:
            Tupla (output, attention_weights)
        """
        batch_size = query.size(0)
        
        # Linear transformations
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Attention
        attn_output, attn_weights = self.attention(Q, K, V, mask)
        
        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        # Output projection
        output = self.W_o(attn_output)
        
        return output, attn_weights


class SelfAttention(nn.Module):
    """
    Self-Attention.
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        """
        Inicializar self-attention.
        
        Args:
            d_model: Dimensión del modelo
            num_heads: Número de heads
            dropout: Dropout rate
        """
        super().__init__()
        self.multi_head_attn = MultiHeadAttention(d_model, num_heads, dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input
            mask: Mask (opcional)
            
        Returns:
            Tupla (output, attention_weights)
        """
        return self.multi_head_attn(x, x, x, mask)


class CrossAttention(nn.Module):
    """
    Cross-Attention.
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        """
        Inicializar cross-attention.
        
        Args:
            d_model: Dimensión del modelo
            num_heads: Número de heads
            dropout: Dropout rate
        """
        super().__init__()
        self.multi_head_attn = MultiHeadAttention(d_model, num_heads, dropout)
    
    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            query: Query
            key_value: Key/Value
            mask: Mask (opcional)
            
        Returns:
            Tupla (output, attention_weights)
        """
        return self.multi_head_attn(query, key_value, key_value, mask)


class SparseAttention(nn.Module):
    """
    Sparse Attention (simplificado).
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        local_window: int = 50,
        dropout: float = 0.1
    ):
        """
        Inicializar sparse attention.
        
        Args:
            d_model: Dimensión del modelo
            num_heads: Número de heads
            local_window: Ventana local
            dropout: Dropout rate
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.local_window = local_window
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass con atención sparse.
        
        Args:
            query: Query
            key: Key
            value: Value
            
        Returns:
            Output
        """
        batch_size, seq_len = query.size(0), query.size(1)
        
        Q = self.W_q(query).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Sparse attention: solo atender a ventana local
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Crear mask para ventana local
        mask = torch.ones(seq_len, seq_len, device=scores.device)
        for i in range(seq_len):
            start = max(0, i - self.local_window // 2)
            end = min(seq_len, i + self.local_window // 2)
            mask[i, :start] = 0
            mask[i, end:] = 0
        
        scores = scores.masked_fill(mask == 0, -1e9)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        output = torch.matmul(attn_weights, V)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.W_o(output)
        
        return output




