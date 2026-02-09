"""
Custom Architectures - Arquitecturas Personalizadas
====================================================

Arquitecturas de modelos personalizadas para tareas específicas.
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math

logger = logging.getLogger(__name__)


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention optimizada"""
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        bias: bool = True
    ):
        """
        Inicializar Multi-Head Attention
        
        Args:
            embed_dim: Dimensión de embedding
            num_heads: Número de heads
            dropout: Dropout rate
            bias: Usar bias
        """
        super().__init__()
        assert embed_dim % num_heads == 0
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            query: Query tensor [batch, seq_len, embed_dim]
            key: Key tensor
            value: Value tensor
            mask: Attention mask
            
        Returns:
            Tuple (output, attention_weights)
        """
        batch_size, seq_len, _ = query.size()
        
        # Proyectar y reshape
        Q = self.q_proj(query).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.embed_dim
        )
        
        output = self.out_proj(attn_output)
        
        return output, attn_weights


class TransformerBlock(nn.Module):
    """Bloque Transformer completo"""
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        ffn_dim: int = 2048,
        dropout: float = 0.1,
        activation: str = "gelu"
    ):
        """
        Inicializar bloque Transformer
        
        Args:
            embed_dim: Dimensión de embedding
            num_heads: Número de heads
            ffn_dim: Dimensión de feed-forward
            dropout: Dropout rate
            activation: Función de activación
        """
        super().__init__()
        
        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU() if activation == "gelu" else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass con residual connections"""
        # Self-attention
        attn_out, _ = self.attention(x, x, x, mask)
        x = self.norm1(x + attn_out)
        
        # Feed-forward
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x


class SocialMediaClassifier(nn.Module):
    """Clasificador personalizado para redes sociales"""
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        num_classes: int = 3,
        max_seq_len: int = 512
    ):
        """
        Inicializar clasificador
        
        Args:
            vocab_size: Tamaño del vocabulario
            embed_dim: Dimensión de embedding
            num_layers: Número de capas
            num_heads: Número de heads
            num_classes: Número de clases
            max_seq_len: Longitud máxima de secuencia
        """
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = PositionalEncoding(embed_dim, max_seq_len)
        
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads)
            for _ in range(num_layers)
        ])
        
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 2, num_classes)
        )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass"""
        # Embedding + positional encoding
        x = self.embedding(input_ids)
        x = self.pos_encoding(x)
        
        # Transformer blocks
        for layer in self.layers:
            x = layer(x, mask=attention_mask)
        
        # Pooling (CLS token o mean pooling)
        if attention_mask is not None:
            x = x * attention_mask.unsqueeze(-1)
            x = x.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
        else:
            x = x.mean(dim=1)
        
        # Classification
        logits = self.classifier(x)
        
        return logits


class PositionalEncoding(nn.Module):
    """Positional Encoding para transformers"""
    
    def __init__(self, embed_dim: int, max_len: int = 5000):
        """
        Inicializar positional encoding
        
        Args:
            embed_dim: Dimensión de embedding
            max_len: Longitud máxima
        """
        super().__init__()
        
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * 
                           (-math.log(10000.0) / embed_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Aplicar positional encoding"""
        return x + self.pe[:, :x.size(1)]


def init_weights(module: nn.Module):
    """Inicialización de pesos"""
    if isinstance(module, nn.Linear):
        torch.nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

