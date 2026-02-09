"""
Custom Attention Mechanisms - Mecanismos de atención personalizados
====================================================================
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention estándar"""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = query.size(0)
        
        # Linear projections
        Q = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        context = torch.matmul(attention_weights, V)
        
        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        output = self.w_o(context)
        return output, attention_weights


class SparseAttention(nn.Module):
    """Sparse Attention para eficiencia"""
    
    def __init__(self, d_model: int, num_heads: int, sparsity: float = 0.5, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.sparsity = sparsity
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Aplicar sparsity mask
        if mask is None:
            seq_len = query.size(1)
            # Crear máscara sparse (ejemplo: solo atender a tokens cercanos)
            sparse_mask = torch.ones(seq_len, seq_len, device=query.device)
            # Implementar lógica de sparsity específica
            # Por ejemplo, solo atender a ventana local
            window_size = int(seq_len * (1 - self.sparsity))
            for i in range(seq_len):
                start = max(0, i - window_size // 2)
                end = min(seq_len, i + window_size // 2)
                sparse_mask[i, :start] = 0
                sparse_mask[i, end:] = 0
            
            if mask is None:
                mask = sparse_mask.unsqueeze(0).unsqueeze(0)
            else:
                mask = mask * sparse_mask.unsqueeze(0).unsqueeze(0)
        
        return self.attention(query, key, value, mask)


class CrossAttention(nn.Module):
    """Cross Attention para modelos multi-modales"""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Cross attention: query de una modalidad, key/value de otra
        return self.attention(query, key, value, mask)


class AttentionFactory:
    """Factory para crear mecanismos de atención"""
    
    @staticmethod
    def create_attention(
        attention_type: str,
        d_model: int,
        num_heads: int,
        **kwargs
    ) -> nn.Module:
        """Crea un mecanismo de atención"""
        if attention_type == "multi_head":
            return MultiHeadAttention(d_model, num_heads, kwargs.get("dropout", 0.1))
        elif attention_type == "sparse":
            return SparseAttention(
                d_model,
                num_heads,
                kwargs.get("sparsity", 0.5),
                kwargs.get("dropout", 0.1)
            )
        elif attention_type == "cross":
            return CrossAttention(d_model, num_heads, kwargs.get("dropout", 0.1))
        else:
            raise ValueError(f"Tipo de atención {attention_type} no soportado")




