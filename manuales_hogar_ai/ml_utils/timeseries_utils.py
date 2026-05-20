"""
Time Series Utils - Utilidades de Series Temporales
====================================================

Utilidades para procesamiento y modelado de series temporales.
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class LSTMForecaster(nn.Module):
    """
    Forecastador LSTM para series temporales.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        output_size: int = 1,
        dropout: float = 0.1
    ):
        """
        Inicializar forecastador LSTM.
        
        Args:
            input_size: Tamaño de entrada
            hidden_size: Tamaño oculto
            num_layers: Número de capas
            output_size: Tamaño de salida
            dropout: Dropout rate
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input [batch, seq_len, features]
            
        Returns:
            Forecast [batch, output_size]
        """
        lstm_out, _ = self.lstm(x)
        # Tomar última salida
        last_output = lstm_out[:, -1, :]
        forecast = self.fc(last_output)
        return forecast


class GRUForecaster(nn.Module):
    """
    Forecastador GRU para series temporales.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        output_size: int = 1,
        dropout: float = 0.1
    ):
        """
        Inicializar forecastador GRU.
        
        Args:
            input_size: Tamaño de entrada
            hidden_size: Tamaño oculto
            num_layers: Número de capas
            output_size: Tamaño de salida
            dropout: Dropout rate
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.gru = nn.GRU(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input [batch, seq_len, features]
            
        Returns:
            Forecast [batch, output_size]
        """
        gru_out, _ = self.gru(x)
        last_output = gru_out[:, -1, :]
        forecast = self.fc(last_output)
        return forecast


class TransformerForecaster(nn.Module):
    """
    Forecastador Transformer para series temporales.
    """
    
    def __init__(
        self,
        input_size: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        output_size: int = 1,
        dropout: float = 0.1
    ):
        """
        Inicializar forecastador Transformer.
        
        Args:
            input_size: Tamaño de entrada
            d_model: Dimensión del modelo
            nhead: Número de heads
            num_layers: Número de capas
            output_size: Tamaño de salida
            dropout: Dropout rate
        """
        super().__init__()
        self.input_projection = nn.Linear(input_size, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.fc = nn.Linear(d_model, output_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input [batch, seq_len, features]
            
        Returns:
            Forecast [batch, output_size]
        """
        x = self.input_projection(x)
        x = self.transformer(x)
        # Tomar última salida
        last_output = x[:, -1, :]
        forecast = self.fc(last_output)
        return forecast


class TimeSeriesDataset(torch.utils.data.Dataset):
    """
    Dataset para series temporales.
    """
    
    def __init__(
        self,
        data: np.ndarray,
        sequence_length: int = 10,
        forecast_horizon: int = 1
    ):
        """
        Inicializar dataset.
        
        Args:
            data: Datos de serie temporal
            sequence_length: Longitud de secuencia
            forecast_horizon: Horizonte de forecast
        """
        self.data = data
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
    
    def __len__(self) -> int:
        """Longitud del dataset."""
        return len(self.data) - self.sequence_length - self.forecast_horizon + 1
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Obtener item.
        
        Args:
            idx: Índice
            
        Returns:
            Tupla (sequence, target)
        """
        sequence = self.data[idx:idx + self.sequence_length]
        target = self.data[idx + self.sequence_length:idx + self.sequence_length + self.forecast_horizon]
        
        return torch.tensor(sequence, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)


def create_sliding_windows(
    data: np.ndarray,
    window_size: int,
    step_size: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Crear ventanas deslizantes.
    
    Args:
        data: Datos
        window_size: Tamaño de ventana
        step_size: Tamaño de paso
        
    Returns:
        Tupla (X, y)
    """
    X = []
    y = []
    
    for i in range(0, len(data) - window_size, step_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])
    
    return np.array(X), np.array(y)




