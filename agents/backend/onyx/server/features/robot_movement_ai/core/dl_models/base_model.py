"""
Base Model
==========

Clase base para todos los modelos de robot.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None

logger = logging.getLogger(__name__)


class BaseRobotModel(nn.Module, ABC):
    """
    Clase base para modelos de robot.
    
    Proporciona funcionalidad común para todos los modelos.
    """
    
    def __init__(
        self,
        input_size: int,
        output_size: int,
        name: str = "BaseRobotModel"
    ):
        """
        Inicializar modelo base.
        
        Args:
            input_size: Tamaño de entrada
            output_size: Tamaño de salida
            name: Nombre del modelo
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for models")
        
        super().__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.name = name
        
        self._initialize_weights()
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass (debe ser implementado por subclases).
        
        Args:
            x: Tensor de entrada
            
        Returns:
            Tensor de salida
        """
        pass
    
    def _initialize_weights(self):
        """Inicializar pesos usando Xavier uniform."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def get_config(self) -> Dict[str, Any]:
        """
        Obtener configuración del modelo.
        
        Returns:
            Diccionario con configuración
        """
        return {
            "name": self.name,
            "input_size": self.input_size,
            "output_size": self.output_size,
            "num_parameters": sum(p.numel() for p in self.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad)
        }
    
    def freeze(self):
        """Congelar todos los parámetros."""
        for param in self.parameters():
            param.requires_grad = False
        logger.info(f"Frozen all parameters in {self.name}")
    
    def unfreeze(self):
        """Descongelar todos los parámetros."""
        for param in self.parameters():
            param.requires_grad = True
        logger.info(f"Unfrozen all parameters in {self.name}")
    
    def freeze_layers(self, layer_indices: list):
        """
        Congelar capas específicas.
        
        Args:
            layer_indices: Índices de capas a congelar
        """
        layers = list(self.modules())
        for idx in layer_indices:
            if 0 <= idx < len(layers):
                for param in layers[idx].parameters():
                    param.requires_grad = False
        logger.info(f"Frozen layers {layer_indices} in {self.name}")




