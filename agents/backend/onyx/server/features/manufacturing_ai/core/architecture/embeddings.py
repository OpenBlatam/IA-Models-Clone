"""
Embeddings and Positional Encodings
===================================

Embeddings y positional encodings avanzados.
"""

import logging
import math
from typing import Optional

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None

logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    """
    Positional encoding sinusoidal.
    
    Encoding posicional para transformers.
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        """
        Inicializar positional encoding.
        
        Args:
            d_model: Dimensión del modelo
            max_len: Longitud máxima
            dropout: Dropout
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required")
        
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Crear matriz de positional encoding
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input [batch, seq_len, d_model]
            
        Returns:
            Input con positional encoding
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class LearnablePositionalEncoding(nn.Module):
    """
    Positional encoding aprendible.
    
    Embeddings posicionales que se aprenden durante el entrenamiento.
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        """
        Inicializar positional encoding aprendible.
        
        Args:
            d_model: Dimensión del modelo
            max_len: Longitud máxima
            dropout: Dropout
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required")
        
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pos_embedding = nn.Embedding(max_len, d_model)
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Inicializar pesos."""
        nn.init.normal_(self.pos_embedding.weight, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input [batch, seq_len, d_model]
            
        Returns:
            Input con positional encoding
        """
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        pos_emb = self.pos_embedding(positions)
        x = x + pos_emb
        return self.dropout(x)


class TokenEmbedding(nn.Module):
    """
    Token embedding con dropout.
    
    Embedding de tokens con normalización.
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        dropout: float = 0.1,
        padding_idx: Optional[int] = None
    ):
        """
        Inicializar token embedding.
        
        Args:
            vocab_size: Tamaño del vocabulario
            d_model: Dimensión del modelo
            dropout: Dropout
            padding_idx: Índice de padding
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required")
        
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Inicializar pesos."""
        nn.init.normal_(self.embedding.weight, mean=0, std=self.d_model ** -0.5)
        if self.embedding.padding_idx is not None:
            nn.init.zeros_(self.embedding.weight[self.embedding.padding_idx])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input [batch, seq_len]
            
        Returns:
            Embeddings [batch, seq_len, d_model]
        """
        # Escalar embeddings
        embeddings = self.embedding(x) * math.sqrt(self.d_model)
        return self.dropout(embeddings)


class FeatureEmbedding(nn.Module):
    """
    Embedding para características numéricas.
    
    Convierte características numéricas en embeddings.
    """
    
    def __init__(
        self,
        num_features: int,
        d_model: int,
        dropout: float = 0.1
    ):
        """
        Inicializar feature embedding.
        
        Args:
            num_features: Número de características
            d_model: Dimensión del modelo
            dropout: Dropout
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required")
        
        super().__init__()
        self.projection = nn.Linear(num_features, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Inicializar pesos."""
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input [batch, num_features]
            
        Returns:
            Embeddings [batch, d_model]
        """
        x = self.projection(x)
        return self.dropout(x)

