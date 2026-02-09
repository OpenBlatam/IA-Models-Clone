"""
Transformer Model - Transformer Encoder
=======================================

Transformer encoder for NLP tasks with positional encoding.
"""

import math
import torch
import torch.nn as nn
from typing import Optional
from .base_model import BaseModel


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformers.
    
    Implements sinusoidal positional encoding as described in
    "Attention Is All You Need" (Vaswani et al., 2017).
    """
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        """
        Initialize positional encoding.
        
        Args:
            d_model: Model dimension
            dropout: Dropout probability
            max_len: Maximum sequence length
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.
        
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
        
        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerEncoder(BaseModel):
    """
    Transformer encoder for NLP tasks.
    
    Architecture:
    - Embedding layer
    - Positional encoding
    - Transformer encoder layers
    - Classification head
    """
    
    def __init__(
        self,
        vocab_size: int = 30000,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        num_classes: int = 2,
        max_length: int = 512,
        device: Optional[torch.device] = None
    ):
        """
        Initialize transformer encoder.
        
        Args:
            vocab_size: Vocabulary size
            d_model: Model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Feedforward dimension
            dropout: Dropout probability
            num_classes: Number of output classes
            max_length: Maximum sequence length
            device: Target device
        """
        super().__init__(device)
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_length)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, num_classes)
        )
        
        self._initialize_weights("xavier_uniform")
        self.to(self.device)
        self._initialized = True
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, seq_len)
            mask: Attention mask (optional)
        
        Returns:
            Output logits of shape (batch, num_classes)
        """
        # Embedding and positional encoding
        x = self.embedding(x) * math.sqrt(self.embedding.embedding_dim)
        x = self.pos_encoder(x)
        
        # Transformer encoding
        x = self.transformer_encoder(x, src_key_padding_mask=mask)
        
        # Use [CLS] token (first token) or mean pooling
        x = x[:, 0, :]  # [CLS] token
        
        # Classification
        x = self.classifier(x)
        return x



