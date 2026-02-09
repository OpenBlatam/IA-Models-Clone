"""
Motion Controller Model
========================

Controlador de movimiento basado en LSTM para secuencias temporales.
"""

import logging
from typing import Optional
import torch
import torch.nn as nn

from .base_model import BaseRobotModel

logger = logging.getLogger(__name__)


class MotionController(BaseRobotModel):
    """
    Controlador de movimiento basado en deep learning.
    
    Usa LSTM para modelar secuencias temporales.
    """
    
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = False,
        use_attention: bool = False
    ):
        """
        Inicializar controlador.
        
        Args:
            input_size: Tamaño de entrada
            output_size: Tamaño de salida
            hidden_size: Tamaño de capa oculta LSTM
            num_layers: Número de capas LSTM
            dropout: Tasa de dropout
            bidirectional: LSTM bidireccional
            use_attention: Usar mecanismo de atención
        """
        super().__init__(input_size, output_size, name="MotionController")
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        
        # LSTM para secuencias temporales
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Mecanismo de atención (opcional)
        if use_attention:
            lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
            self.attention = nn.MultiheadAttention(
                embed_dim=lstm_output_size,
                num_heads=8,
                batch_first=True
            )
            self.attention_norm = nn.LayerNorm(lstm_output_size)
        
        # Capa de salida
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.fc = nn.Sequential(
            nn.Linear(lstm_output_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size)
        )
        
        self._initialize_weights()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Tensor de entrada [batch_size, seq_len, input_size]
            
        Returns:
            Tensor de salida [batch_size, output_size]
        """
        # LSTM
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Atención (opcional)
        if self.use_attention:
            attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
            lstm_out = self.attention_norm(lstm_out + attn_out)
        
        # Usar la última salida de la secuencia
        last_output = lstm_out[:, -1, :]
        
        # Fully connected
        return self.fc(last_output)
    
    def _initialize_weights(self):
        """Inicializar pesos."""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.zeros_(param.data)
                # Set forget gate bias to 1
                n = param.size(0)
                start, end = n // 4, n // 2
                param.data[start:end].fill_(1.0)
        
        for module in self.fc.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)




