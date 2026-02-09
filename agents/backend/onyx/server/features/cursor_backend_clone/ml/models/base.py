"""
Base Model - Modelo base para todos los modelos
================================================

Clase base abstracta para todos los modelos de deep learning.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class BaseModel(nn.Module, ABC):
    """Modelo base abstracto para todos los modelos"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._initialized = False
    
    @abstractmethod
    def forward(self, *args, **kwargs):
        """Forward pass del modelo"""
        pass
    
    @abstractmethod
    def generate(self, *args, **kwargs):
        """Generar salida del modelo"""
        pass
    
    def to_device(self, device: Optional[torch.device] = None):
        """Mover modelo a dispositivo"""
        if device is None:
            device = self.device
        self.device = device
        self.to(device)
        logger.info(f"Model moved to {device}")
    
    def save(self, path: str):
        """Guardar modelo"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.config
        }, path)
        logger.info(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: str, device: Optional[torch.device] = None):
        """Cargar modelo"""
        checkpoint = torch.load(path, map_location=device)
        model = cls(checkpoint['config'])
        model.load_state_dict(checkpoint['model_state_dict'])
        if device:
            model.to_device(device)
        logger.info(f"Model loaded from {path}")
        return model
    
    def get_num_parameters(self) -> Dict[str, int]:
        """Obtener número de parámetros"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total": total,
            "trainable": trainable,
            "non_trainable": total - trainable
        }
    
    def freeze(self):
        """Congelar todos los parámetros"""
        for param in self.parameters():
            param.requires_grad = False
        logger.info("Model frozen")
    
    def unfreeze(self):
        """Descongelar todos los parámetros"""
        for param in self.parameters():
            param.requires_grad = True
        logger.info("Model unfrozen")


