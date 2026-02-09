"""
Base Model for Routing
======================

Clase base e interfaces para modelos de enrutamiento.
"""

import logging
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuración base para modelos."""
    input_dim: int = 20
    hidden_dims: list = field(default_factory=lambda: [128, 256, 128])
    output_dim: int = 4
    dropout: float = 0.2
    activation: str = "relu"
    use_batch_norm: bool = True
    use_layer_norm: bool = False
    device: Optional[str] = None
    
    def __post_init__(self):
        """Validar configuración."""
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"


class BaseRouteModel(nn.Module, ABC):
    """
    Clase base abstracta para modelos de enrutamiento.
    
    Todas las implementaciones de modelos deben heredar de esta clase.
    """
    
    def __init__(self, config: ModelConfig):
        """
        Inicializar modelo base.
        
        Args:
            config: Configuración del modelo
        """
        super(BaseRouteModel, self).__init__()
        self.config = config
        self.device = torch.device(config.device)
        
        # Mover modelo al dispositivo
        self.to(self.device)
        
        # Inicializar pesos
        self._initialize_weights()
    
    @abstractmethod
    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Forward pass (debe ser implementado por subclases).
        
        Args:
            x: Tensor de entrada
            
        Returns:
            Tensor de salida
        """
        pass
    
    def _initialize_weights(self):
        """
        Inicializar pesos del modelo.
        
        Puede ser sobrescrito por subclases para inicialización personalizada.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def get_activation(self, name: str) -> nn.Module:
        """
        Obtener función de activación.
        
        Args:
            name: Nombre de la activación
            
        Returns:
            Módulo de activación
        """
        activations = {
            "relu": nn.ReLU(),
            "elu": nn.ELU(),
            "gelu": nn.GELU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
            "leaky_relu": nn.LeakyReLU(0.2)
        }
        return activations.get(name.lower(), nn.ReLU())
    
    def save_checkpoint(self, path: str, optimizer: Optional[torch.optim.Optimizer] = None, 
                       scheduler: Optional[Any] = None, metadata: Optional[Dict[str, Any]] = None):
        """
        Guardar checkpoint del modelo.
        
        Args:
            path: Ruta donde guardar
            optimizer: Optimizador (opcional)
            scheduler: Scheduler (opcional)
            metadata: Metadata adicional (opcional)
        """
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "config": self.config.__dict__,
            "model_class": self.__class__.__name__
        }
        
        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        
        if scheduler is not None:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()
        
        if metadata is not None:
            checkpoint["metadata"] = metadata
        
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint guardado en: {path}")
    
    @classmethod
    def load_checkpoint(cls, path: str, device: Optional[str] = None) -> Tuple['BaseRouteModel', Dict[str, Any]]:
        """
        Cargar checkpoint del modelo.
        
        Args:
            path: Ruta del checkpoint
            device: Dispositivo (opcional)
            
        Returns:
            (modelo, metadata)
        """
        checkpoint = torch.load(path, map_location=device)
        
        # Reconstruir configuración
        config = ModelConfig(**checkpoint["config"])
        if device is not None:
            config.device = device
        
        # Crear instancia del modelo
        # Nota: Esto requiere que la subclase implemente __init__ correctamente
        model = cls(config)
        model.load_state_dict(checkpoint["model_state_dict"])
        
        metadata = checkpoint.get("metadata", {})
        
        logger.info(f"Checkpoint cargado desde: {path}")
        return model, metadata
    
    def count_parameters(self) -> int:
        """
        Contar número de parámetros entrenables.
        
        Returns:
            Número de parámetros
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Obtener información del modelo.
        
        Returns:
            Diccionario con información del modelo
        """
        return {
            "model_class": self.__class__.__name__,
            "num_parameters": self.count_parameters(),
            "config": self.config.__dict__,
            "device": str(self.device)
        }




