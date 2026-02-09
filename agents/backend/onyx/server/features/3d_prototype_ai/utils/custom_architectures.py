"""
Custom Architectures - Arquitecturas de modelos personalizadas
================================================================
Arquitecturas custom para prototipos 3D
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


class PrototypeEncoder(nn.Module):
    """Encoder para descripciones de prototipos"""
    
    def __init__(
        self,
        vocab_size: int = 10000,
        embed_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        self.dropout = nn.Dropout(dropout)
        self.output_dim = hidden_dim * 2  # Bidirectional
    
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        # Usar último hidden state
        output = torch.cat([hidden[-2], hidden[-1]], dim=1)
        return self.dropout(output)


class PrototypeDecoder(nn.Module):
    """Decoder para generar especificaciones de prototipos"""
    
    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 512,
        output_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        self.fc_layers = nn.ModuleList([
            nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim)
            for i in range(num_layers)
        ])
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
    
    def forward(self, x):
        for fc in self.fc_layers:
            x = self.activation(fc(x))
            x = self.dropout(x)
        output = self.output_layer(x)
        return output


class PrototypeGeneratorModel(nn.Module):
    """Modelo completo para generación de prototipos"""
    
    def __init__(
        self,
        vocab_size: int = 10000,
        embed_dim: int = 256,
        encoder_hidden: int = 512,
        decoder_hidden: int = 512,
        output_dim: int = 256,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        self.encoder = PrototypeEncoder(
            vocab_size, embed_dim, encoder_hidden, num_encoder_layers, dropout
        )
        self.decoder = PrototypeDecoder(
            self.encoder.output_dim, decoder_hidden, output_dim, num_decoder_layers, dropout
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class AttentionLayer(nn.Module):
    """Capa de atención multi-head"""
    
    def __init__(self, embed_dim: int = 256, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        attn_out, _ = self.attention(x, x, x)
        out = self.norm(x + self.dropout(attn_out))
        return out


class TransformerPrototypeModel(nn.Module):
    """Modelo Transformer para prototipos"""
    
    def __init__(
        self,
        vocab_size: int = 10000,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        ff_dim: int = 1024,
        max_seq_len: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = nn.Parameter(torch.randn(1, max_seq_len, embed_dim))
        
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                embed_dim, num_heads, ff_dim, dropout, batch_first=True
            )
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask: Optional[torch.Tensor] = None):
        seq_len = x.size(1)
        x = self.embedding(x) + self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x)
        
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=mask)
        
        x = self.norm(x)
        return x


class PrototypeClassifier(nn.Module):
    """Clasificador de prototipos"""
    
    def __init__(
        self,
        input_dim: int = 256,
        num_classes: int = 10,
        hidden_dims: List[int] = [512, 256],
        dropout: float = 0.1
    ):
        super().__init__()
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        self.classifier = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.classifier(x)




