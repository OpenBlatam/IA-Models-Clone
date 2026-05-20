"""
Attention Layers
================

Capas de atención avanzadas para modelos.
"""

import logging
import math
from typing import Optional, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    F = None

logger = logging.getLogger(__name__)


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism.
    
    Implementación eficiente de atención multi-head.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        bias: bool = True
    ):
        """
        Inicializar atención multi-head.
        
        Args:
            embed_dim: Dimensión de embedding
            num_heads: Número de heads
            dropout: Tasa de dropout
            bias: Usar bias
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required")
        
        super().__init__()
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Proyecciones Q, K, V
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        # Proyección de salida
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        self.dropout = nn.Dropout(dropout)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Inicializar pesos."""
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
    
    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            query: Query tensor [batch, seq_len, embed_dim]
            key: Key tensor (opcional, usa query si None)
            value: Value tensor (opcional, usa query si None)
            mask: Attention mask (opcional)
            
        Returns:
            Tupla (output, attention_weights)
        """
        batch_size, seq_len, _ = query.shape
        
        if key is None:
            key = query
        if value is None:
            value = query
        
        # Proyecciones
        Q = self.q_proj(query)
        K = self.k_proj(key)
        V = self.v_proj(value)
        
        # Reshape para multi-head
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Aplicar atención a valores
        attended = torch.matmul(attention_weights, V)
        
        # Concatenar heads
        attended = attended.transpose(1, 2).contiguous()
        attended = attended.view(batch_size, seq_len, self.embed_dim)
        
        # Proyección de salida
        output = self.out_proj(attended)
        
        return output, attention_weights


class SelfAttention(nn.Module):
    """
    Self-Attention layer.
    
    Wrapper para MultiHeadAttention con query=key=value.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        """Inicializar self-attention."""
        super().__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch, seq_len, embed_dim]
            mask: Attention mask (opcional)
            
        Returns:
            Output tensor
        """
        output, _ = self.attention(x, x, x, mask)
        return output


class CrossAttention(nn.Module):
    """
    Cross-Attention layer.
    
    Atención entre dos secuencias diferentes.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        """Inicializar cross-attention."""
        super().__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout)
    
    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            query: Query tensor [batch, seq_len_q, embed_dim]
            key_value: Key/Value tensor [batch, seq_len_kv, embed_dim]
            mask: Attention mask (opcional)
            
        Returns:
            Output tensor
        """
        output, _ = self.attention(query, key_value, key_value, mask)
        return output

