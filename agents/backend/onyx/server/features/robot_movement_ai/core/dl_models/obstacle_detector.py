"""
Obstacle Detector Model
=======================

Detector de obstáculos usando CNN para procesar datos de sensores.
"""

import logging
from typing import List
import torch
import torch.nn as nn

from .base_model import BaseRobotModel

logger = logging.getLogger(__name__)


class ObstacleDetector(BaseRobotModel):
    """
    Detector de obstáculos usando CNN.
    
    Procesa datos de sensores (LIDAR, cámaras, etc.).
    """
    
    def __init__(
        self,
        input_channels: int = 1,
        num_classes: int = 2,  # obstáculo / sin obstáculo
        conv_channels: List[int] = [32, 64, 128],
        use_residual: bool = False
    ):
        """
        Inicializar detector.
        
        Args:
            input_channels: Canales de entrada
            num_classes: Número de clases
            conv_channels: Canales de convolución
            use_residual: Usar conexiones residuales
        """
        # Para CNN, input_size y output_size se interpretan diferente
        super().__init__(
            input_size=input_channels,
            output_size=num_classes,
            name="ObstacleDetector"
        )
        
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.conv_channels = conv_channels
        self.use_residual = use_residual
        
        # Capas convolucionales
        conv_layers = []
        prev_channels = input_channels
        
        for channels in conv_channels:
            conv_layers.extend([
                nn.Conv2d(prev_channels, channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(channels),
                nn.ReLU(),
                nn.MaxPool2d(2)
            ])
            prev_channels = channels
        
        self.conv_layers = nn.Sequential(*conv_layers)
        
        # Capas fully connected
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(conv_channels[-1], 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
        
        self._initialize_weights()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Tensor de entrada [batch_size, channels, height, width]
            
        Returns:
            Logits [batch_size, num_classes]
        """
        x = self.conv_layers(x)
        return self.fc(x)
    
    def _initialize_weights(self):
        """Inicializar pesos."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)




