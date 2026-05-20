"""
Architecture Utils - Utilidades de Arquitecturas de Modelos
============================================================

Utilidades para construir arquitecturas comunes de modelos.
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Any
import math

logger = logging.getLogger(__name__)

# Intentar importar transformers
try:
    from transformers import AutoConfig
    _has_transformers = True
except ImportError:
    _has_transformers = False


class PositionalEncoding(nn.Module):
    """
    Positional encoding para transformers.
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        """
        Inicializar positional encoding.
        
        Args:
            d_model: Dimensión del modelo
            max_len: Longitud máxima
            dropout: Dropout rate
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Aplicar positional encoding.
        
        Args:
            x: Input tensor [seq_len, batch_size, d_model]
            
        Returns:
            Tensor con positional encoding
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism.
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
            query: Query tensor
            key: Key tensor
            value: Value tensor
            mask: Attention mask (opcional)
            
        Returns:
            Tupla (output, attention_weights)
        """
        batch_size = query.size(0)
        
        # Linear transformations and split into heads
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        output = self.W_o(context)
        return output, attention_weights


class TransformerBlock(nn.Module):
    """
    Bloque transformer estándar.
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1
    ):
        """
        Inicializar bloque transformer.
        
        Args:
            d_model: Dimensión del modelo
            num_heads: Número de heads de atención
            d_ff: Dimensión de feed-forward
            dropout: Dropout rate
        """
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor
            mask: Attention mask (opcional)
            
        Returns:
            Output tensor
        """
        # Self-attention
        attn_output, _ = self.attention(x, x, x, mask)
        x = self.norm1(x + attn_output)
        
        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        
        return x


class TransformerEncoder(nn.Module):
    """
    Encoder transformer completo.
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        d_ff: int = 2048,
        max_len: int = 5000,
        dropout: float = 0.1
    ):
        """
        Inicializar encoder transformer.
        
        Args:
            vocab_size: Tamaño del vocabulario
            d_model: Dimensión del modelo
            num_heads: Número de heads
            num_layers: Número de capas
            d_ff: Dimensión de feed-forward
            max_len: Longitud máxima
            dropout: Dropout rate
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tokens [batch_size, seq_len]
            mask: Attention mask (opcional)
            
        Returns:
            Encoded tensor
        """
        x = self.embedding(x) * math.sqrt(self.embedding.embedding_dim)
        x = self.pos_encoding(x.transpose(0, 1)).transpose(0, 1)
        x = self.dropout(x)
        
        for layer in self.layers:
            x = layer(x, mask)
        
        return x


class ResNetBlock(nn.Module):
    """
    Bloque ResNet básico.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None
    ):
        """
        Inicializar bloque ResNet.
        
        Args:
            in_channels: Canales de entrada
            out_channels: Canales de salida
            stride: Stride de convolución
            downsample: Módulo de downsample (opcional)
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out


class LSTMEncoder(nn.Module):
    """
    Encoder LSTM para secuencias.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bidirectional: bool = True,
        dropout: float = 0.1
    ):
        """
        Inicializar encoder LSTM.
        
        Args:
            input_size: Tamaño de entrada
            hidden_size: Tamaño oculto
            num_layers: Número de capas
            bidirectional: Bidireccional
            dropout: Dropout rate
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0
        )
    
    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch_size, seq_len, input_size]
            lengths: Longitudes de secuencias (opcional)
            
        Returns:
            Tupla (output, (hidden, cell))
        """
        if lengths is not None:
            x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        
        output, (hidden, cell) = self.lstm(x)
        
        if lengths is not None:
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        
        return output, (hidden, cell)


class CNNEncoder(nn.Module):
    """
    Encoder CNN para imágenes.
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        num_layers: int = 4
    ):
        """
        Inicializar encoder CNN.
        
        Args:
            in_channels: Canales de entrada
            base_channels: Canales base
            num_layers: Número de capas
        """
        super().__init__()
        layers = []
        
        channels = in_channels
        for i in range(num_layers):
            layers.extend([
                nn.Conv2d(channels, base_channels * (2 ** i), 3, 1, 1),
                nn.BatchNorm2d(base_channels * (2 ** i)),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2)
            ])
            channels = base_channels * (2 ** i)
        
        self.encoder = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch_size, channels, height, width]
            
        Returns:
            Encoded tensor
        """
        return self.encoder(x)


def create_transformer(
    vocab_size: int,
    d_model: int = 512,
    num_heads: int = 8,
    num_layers: int = 6,
    **kwargs
) -> TransformerEncoder:
    """
    Crear transformer encoder.
    
    Args:
        vocab_size: Tamaño del vocabulario
        d_model: Dimensión del modelo
        num_heads: Número de heads
        num_layers: Número de capas
        **kwargs: Argumentos adicionales
        
    Returns:
        TransformerEncoder
    """
    return TransformerEncoder(vocab_size, d_model, num_heads, num_layers, **kwargs)


def create_resnet(
    in_channels: int = 3,
    num_classes: int = 1000,
    num_blocks: List[int] = [2, 2, 2, 2]
) -> nn.Module:
    """
    Crear ResNet.
    
    Args:
        in_channels: Canales de entrada
        num_classes: Número de clases
        num_blocks: Número de bloques por capa
        
    Returns:
        Modelo ResNet
    """
    class ResNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(in_channels, 64, 7, 2, 3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(3, 2, 1)
            
            self.layer1 = self._make_layer(64, 64, num_blocks[0])
            self.layer2 = self._make_layer(64, 128, num_blocks[1], stride=2)
            self.layer3 = self._make_layer(128, 256, num_blocks[2], stride=2)
            self.layer4 = self._make_layer(256, 512, num_blocks[3], stride=2)
            
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512, num_classes)
        
        def _make_layer(self, in_channels, out_channels, blocks, stride=1):
            downsample = None
            if stride != 1 or in_channels != out_channels:
                downsample = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                    nn.BatchNorm2d(out_channels)
                )
            
            layers = [ResNetBlock(in_channels, out_channels, stride, downsample)]
            for _ in range(1, blocks):
                layers.append(ResNetBlock(out_channels, out_channels))
            
            return nn.Sequential(*layers)
        
        def forward(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
            
            return x
    
    return ResNet()




