"""
Trajectory Predictor Model
==========================

Modelo MLP para predecir trayectorias de robot.
"""

import logging
from typing import List
import torch
import torch.nn as nn

from .base_model import BaseRobotModel

logger = logging.getLogger(__name__)


class TrajectoryPredictor(BaseRobotModel):
    """
    Modelo de red neuronal para predecir trayectorias.
    
    Arquitectura: MLP con capas ocultas configurables.
    """
    
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_sizes: List[int] = [128, 64, 32],
        activation: str = "relu",
        dropout: float = 0.1,
        use_batch_norm: bool = False
    ):
        """
        Inicializar modelo.
        
        Args:
            input_size: Tamaño de entrada (posición actual, velocidad, etc.)
            output_size: Tamaño de salida (posición futura, velocidad, etc.)
            hidden_sizes: Tamaños de capas ocultas
            activation: Función de activación
            dropout: Tasa de dropout
            use_batch_norm: Usar batch normalization
        """
        super().__init__(input_size, output_size, name="TrajectoryPredictor")
        
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        
        # Construir capas
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            
            # Función de activación
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            elif activation == "sigmoid":
                layers.append(nn.Sigmoid())
            elif activation == "gelu":
                layers.append(nn.GELU())
            elif activation == "leaky_relu":
                layers.append(nn.LeakyReLU(0.2))
            else:
                layers.append(nn.ReLU())
            
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            
            prev_size = hidden_size
        
        # Capa de salida
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
        
        # Reinicializar pesos (sobrescribe el de la clase base)
        self._initialize_weights()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Tensor de entrada [batch_size, input_size]
            
        Returns:
            Tensor de salida [batch_size, output_size]
        """
        return self.network(x)
    
    def _initialize_weights(self):
        """Inicializar pesos usando Kaiming normal para ReLU."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if self.activation in ["relu", "leaky_relu"]:
                    nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                else:
                    nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)




