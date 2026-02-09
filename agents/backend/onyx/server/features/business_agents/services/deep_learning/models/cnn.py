"""
CNN Model - Convolutional Neural Network
========================================

Simple CNN for image classification with BatchNorm and Dropout.
"""

import torch
import torch.nn as nn
from typing import Optional
from .base_model import BaseModel


class SimpleCNN(BaseModel):
    """
    Simple CNN for image classification.
    
    Architecture:
    - 3 convolutional blocks with BatchNorm and MaxPool
    - Adaptive average pooling
    - Dropout for regularization
    - Fully connected classifier
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 10,
        device: Optional[torch.device] = None
    ):
        """
        Initialize CNN model.
        
        Args:
            in_channels: Number of input channels (3 for RGB)
            num_classes: Number of output classes
            device: Target device
        """
        super().__init__(device)
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )
        
        self._initialize_weights("kaiming_uniform")
        self.to(self.device)
        self._initialized = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, channels, height, width)
        
        Returns:
            Output logits of shape (batch, num_classes)
        """
        x = self.conv_layers(x)
        x = self.classifier(x)
        return x



