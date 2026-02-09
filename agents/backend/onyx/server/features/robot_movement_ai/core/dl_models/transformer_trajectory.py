"""
Transformer-based Trajectory Predictor
=======================================

Modelo Transformer avanzado para predicción de trayectorias de robot
usando arquitecturas de atención multi-head y codificación posicional.
"""

import logging
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .base_model import BaseRobotModel

logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    """
    Codificación posicional para secuencias temporales.
    
    Implementa la codificación posicional sinusoidal de "Attention is All You Need".
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        """
        Inicializar codificación posicional.
        
        Args:
            d_model: Dimensión del modelo
            max_len: Longitud máxima de secuencia
            dropout: Tasa de dropout
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Crear matriz de codificación posicional
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Aplicar codificación posicional.
        
        Args:
            x: Tensor de entrada [seq_len, batch_size, d_model]
            
        Returns:
            Tensor con codificación posicional
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """
    Atención multi-head para capturar diferentes tipos de dependencias.
    """
    
    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        """
        Inicializar atención multi-head.
        
        Args:
            d_model: Dimensión del modelo
            num_heads: Número de cabezas de atención
            dropout: Tasa de dropout
        """
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
    ) -> torch.Tensor:
        """
        Forward pass de atención multi-head.
        
        Args:
            query: Tensor de consulta [batch_size, seq_len, d_model]
            key: Tensor de clave [batch_size, seq_len, d_model]
            value: Tensor de valor [batch_size, seq_len, d_model]
            mask: Máscara opcional
            
        Returns:
            Tensor de salida [batch_size, seq_len, d_model]
        """
        batch_size = query.size(0)
        
        # Proyectar y dividir en heads
        Q = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Calcular atención
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Aplicar atención a valores
        context = torch.matmul(attention_weights, V)
        
        # Concatenar heads
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        # Proyección final
        output = self.w_o(context)
        
        return output


class TransformerBlock(nn.Module):
    """
    Bloque Transformer con atención y feed-forward.
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1
    ):
        """
        Inicializar bloque Transformer.
        
        Args:
            d_model: Dimensión del modelo
            num_heads: Número de cabezas de atención
            d_ff: Dimensión de la capa feed-forward
            dropout: Tasa de dropout
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
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass del bloque Transformer.
        
        Args:
            x: Tensor de entrada [batch_size, seq_len, d_model]
            mask: Máscara opcional
            
        Returns:
            Tensor de salida [batch_size, seq_len, d_model]
        """
        # Self-attention con residual connection
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward con residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        
        return x


class TransformerTrajectoryPredictor(BaseRobotModel):
    """
    Predictor de trayectorias basado en Transformer.
    
    Usa arquitectura Transformer para modelar dependencias temporales
    en secuencias de movimiento de robot.
    """
    
    def __init__(
        self,
        input_size: int,
        output_size: int,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        d_ff: int = 1024,
        max_seq_len: int = 100,
        dropout: float = 0.1,
        use_positional_encoding: bool = True
    ):
        """
        Inicializar predictor Transformer.
        
        Args:
            input_size: Tamaño de entrada (features por timestep)
            output_size: Tamaño de salida (posición/velocidad predicha)
            d_model: Dimensión del modelo
            num_heads: Número de cabezas de atención
            num_layers: Número de capas Transformer
            d_ff: Dimensión de capa feed-forward
            max_seq_len: Longitud máxima de secuencia
            dropout: Tasa de dropout
            use_positional_encoding: Usar codificación posicional
        """
        super().__init__(input_size, output_size, name="TransformerTrajectoryPredictor")
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        
        # Proyección de entrada
        self.input_projection = nn.Linear(input_size, d_model)
        
        # Codificación posicional
        self.use_positional_encoding = use_positional_encoding
        if use_positional_encoding:
            self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # Bloques Transformer
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Capa de salida
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_ff // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff // 2, output_size)
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # Inicializar pesos
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Inicializar pesos usando inicialización Xavier."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Tensor de entrada [batch_size, seq_len, input_size]
            mask: Máscara opcional [batch_size, seq_len, seq_len]
            
        Returns:
            Tensor de salida [batch_size, output_size]
        """
        batch_size, seq_len, _ = x.size()
        
        # Proyectar entrada a dimensión del modelo
        x = self.input_projection(x)  # [batch_size, seq_len, d_model]
        
        # Aplicar codificación posicional si está habilitada
        if self.use_positional_encoding:
            # Transponer para formato [seq_len, batch_size, d_model]
            x = x.transpose(0, 1)
            x = self.pos_encoder(x)
            x = x.transpose(0, 1)  # Volver a [batch_size, seq_len, d_model]
        
        x = self.dropout(x)
        
        # Pasar por bloques Transformer
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, mask)
        
        # Usar la última posición de la secuencia para predicción
        # Alternativamente, se podría usar pooling o todas las posiciones
        last_hidden = x[:, -1, :]  # [batch_size, d_model]
        
        # Proyección de salida
        output = self.output_projection(last_hidden)  # [batch_size, output_size]
        
        return output
    
    def predict_sequence(
        self,
        x: torch.Tensor,
        future_steps: int = 10,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Predecir secuencia futura autoregresivamente.
        
        Args:
            x: Tensor de entrada [batch_size, seq_len, input_size]
            future_steps: Número de pasos futuros a predecir
            mask: Máscara opcional
            
        Returns:
            Tensor de predicciones [batch_size, future_steps, output_size]
        """
        self.eval()
        predictions = []
        
        with torch.no_grad():
            current_input = x
            
            for _ in range(future_steps):
                # Predecir siguiente paso
                next_pred = self.forward(current_input, mask)
                predictions.append(next_pred)
                
                # Agregar predicción a la entrada para siguiente iteración
                # (asumiendo que input_size == output_size o similar)
                if current_input.size(-1) == next_pred.size(-1):
                    next_input = torch.cat([
                        current_input[:, 1:, :],  # Desplazar ventana
                        next_pred.unsqueeze(1)    # Agregar predicción
                    ], dim=1)
                    current_input = next_input
                else:
                    # Si tamaños no coinciden, solo usar predicción
                    break
        
        return torch.stack(predictions, dim=1)  # [batch_size, future_steps, output_size]

