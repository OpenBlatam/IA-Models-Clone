"""
Base Model - Modelo base para todos los modelos
================================================

Clase base abstracta para todos los modelos de deep learning.
Proporciona funcionalidad común para guardar, cargar, y gestionar modelos.
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional, TypeVar, Type
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

T = TypeVar('T', bound='BaseModel')


class BaseModel(nn.Module, ABC):
    """
    Modelo base abstracto para todos los modelos de deep learning.
    
    Esta clase proporciona funcionalidad común para:
    - Gestión de dispositivos (CPU/GPU)
    - Guardado y carga de modelos
    - Congelado/descongelado de parámetros
    - Conteo de parámetros
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Inicializar modelo base.
        
        Args:
            config: Diccionario de configuración del modelo.
        """
        super().__init__()
        self.config: Dict[str, Any] = config
        self.device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._initialized: bool = False
    
    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """
        Forward pass del modelo.
        
        Args:
            *args: Argumentos posicionales.
            **kwargs: Argumentos con nombre.
        
        Returns:
            Tensor con la salida del modelo.
        """
        pass
    
    @abstractmethod
    def generate(self, *args: Any, **kwargs: Any) -> str:
        """
        Generar salida del modelo.
        
        Args:
            *args: Argumentos posicionales.
            **kwargs: Argumentos con nombre.
        
        Returns:
            Salida generada como string.
        """
        pass
    
    def to_device(self, device: Optional[torch.device] = None) -> None:
        """
        Mover modelo a dispositivo específico.
        
        Args:
            device: Dispositivo destino. Si es None, usa self.device.
        
        Raises:
            RuntimeError: Si hay error al mover el modelo.
        """
        if device is None:
            device = self.device
        
        try:
            self.device = device
            self.to(device)
            logger.info(f"Model moved to {device}")
        except Exception as e:
            logger.error(f"Error moving model to device: {e}", exc_info=True)
            raise RuntimeError(f"Failed to move model to {device}: {e}") from e
    
    def save(self, path: str) -> None:
        """
        Guardar modelo en disco.
        
        Args:
            path: Ruta donde guardar el modelo.
        
        Raises:
            RuntimeError: Si hay error al guardar.
            ValueError: Si la ruta es inválida.
        """
        if not path or not path.strip():
            raise ValueError("Path cannot be empty")
        
        try:
            path_obj = Path(path)
            path_obj.parent.mkdir(parents=True, exist_ok=True)
            
            checkpoint = {
                'model_state_dict': self.state_dict(),
                'config': self.config,
                'device': str(self.device)
            }
            
            torch.save(checkpoint, path)
            logger.info(f"Model saved to {path}")
        except Exception as e:
            logger.error(f"Error saving model: {e}", exc_info=True)
            raise RuntimeError(f"Failed to save model to {path}: {e}") from e
    
    @classmethod
    def load(
        cls: Type[T],
        path: str,
        device: Optional[torch.device] = None
    ) -> T:
        """
        Cargar modelo desde disco.
        
        Args:
            path: Ruta del modelo guardado.
            device: Dispositivo donde cargar el modelo (opcional).
        
        Returns:
            Instancia del modelo cargado.
        
        Raises:
            FileNotFoundError: Si el archivo no existe.
            RuntimeError: Si hay error al cargar el modelo.
            KeyError: Si faltan claves requeridas en el checkpoint.
        """
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        
        try:
            checkpoint = torch.load(path, map_location=device)
            
            if 'config' not in checkpoint:
                raise KeyError("Checkpoint missing 'config' key")
            if 'model_state_dict' not in checkpoint:
                raise KeyError("Checkpoint missing 'model_state_dict' key")
            
            model = cls(checkpoint['config'])
            model.load_state_dict(checkpoint['model_state_dict'])
            
            if device is not None:
                model.to_device(device)
            elif 'device' in checkpoint:
                model.device = torch.device(checkpoint['device'])
            
            logger.info(f"Model loaded from {path}")
            return model
            
        except FileNotFoundError:
            raise
        except KeyError as e:
            logger.error(f"Invalid checkpoint format: {e}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Error loading model: {e}", exc_info=True)
            raise RuntimeError(f"Failed to load model from {path}: {e}") from e
    
    def get_num_parameters(self) -> Dict[str, int]:
        """
        Obtener número de parámetros del modelo.
        
        Returns:
            Diccionario con:
                - total: Total de parámetros
                - trainable: Parámetros entrenables
                - non_trainable: Parámetros no entrenables
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        non_trainable = total - trainable
        
        return {
            "total": total,
            "trainable": trainable,
            "non_trainable": non_trainable
        }
    
    def freeze(self) -> None:
        """
        Congelar todos los parámetros del modelo.
        
        Los parámetros congelados no se actualizarán durante el entrenamiento.
        """
        for param in self.parameters():
            param.requires_grad = False
        logger.info("Model frozen")
    
    def unfreeze(self) -> None:
        """
        Descongelar todos los parámetros del modelo.
        
        Los parámetros descongelados se actualizarán durante el entrenamiento.
        """
        for param in self.parameters():
            param.requires_grad = True
        logger.info("Model unfrozen")



