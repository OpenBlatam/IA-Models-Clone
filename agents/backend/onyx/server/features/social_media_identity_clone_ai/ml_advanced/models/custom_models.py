"""
Modelos personalizados con nn.Module
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class IdentityStyleEncoder(nn.Module):
    """Encoder de estilo de identidad"""
    
    def __init__(
        self,
        vocab_size: int = 50257,
        d_model: int = 768,
        nhead: int = 12,
        num_layers: int = 6,
        dim_feedforward: int = 3072,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.style_projection = nn.Linear(d_model, 256)
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()
    
    def _init_weights(self):
        """Inicializa pesos"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        """
        Forward pass
        
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
        """
        # Embedding
        x = self.embedding(input_ids) * (self.embedding.embedding_dim ** 0.5)
        x = self.pos_encoder(x)
        
        # Transformer
        if attention_mask is not None:
            # Convertir attention mask para transformer
            attn_mask = attention_mask == 0
        else:
            attn_mask = None
        
        x = self.transformer(x, src_key_padding_mask=attn_mask)
        
        # Pooling (mean over sequence)
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).expand(x.size()).float()
            x = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1e-9)
        else:
            x = x.mean(dim=1)
        
        # Projection
        x = self.style_projection(x)
        x = self.dropout(x)
        
        return x


class PositionalEncoding(nn.Module):
    """Positional encoding para transformers"""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class ContentGeneratorModel(nn.Module):
    """Modelo generador de contenido personalizado"""
    
    def __init__(
        self,
        style_dim: int = 256,
        vocab_size: int = 50257,
        d_model: int = 768,
        num_layers: int = 12,
        nhead: int = 12
    ):
        super().__init__()
        
        self.style_encoder = IdentityStyleEncoder()
        self.content_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 4,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=num_layers
        )
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        self.style_projection = nn.Linear(256, d_model)
        
    def forward(
        self,
        input_ids: torch.Tensor,
        style_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ):
        """
        Args:
            input_ids: [batch_size, seq_len]
            style_features: [batch_size, style_dim]
            attention_mask: [batch_size, seq_len]
        """
        # Embedding
        x = self.embedding(input_ids) * (self.embedding.embedding_dim ** 0.5)
        x = self.pos_encoder(x)
        
        # Style conditioning
        style_cond = self.style_projection(style_features)
        style_cond = style_cond.unsqueeze(1)  # [batch_size, 1, d_model]
        
        # Decoder
        memory = style_cond.expand(-1, x.size(1), -1)
        x = self.content_decoder(x, memory)
        
        # Output projection
        logits = self.output_projection(x)
        
        return logits




