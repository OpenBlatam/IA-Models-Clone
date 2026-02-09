"""
Clase base para servicios con funcionalidad común
"""

import logging
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import torch

from ..config import get_settings
from .exceptions import SocialMediaIdentityCloneError

logger = logging.getLogger(__name__)


class BaseService(ABC):
    """Clase base para todos los servicios"""
    
    def __init__(self):
        self.settings = get_settings()
        self.logger = logging.getLogger(self.__class__.__name__)
        self._device: Optional[torch.device] = None
    
    @property
    def device(self) -> torch.device:
        """Obtiene el dispositivo (GPU/CPU)"""
        if self._device is None:
            self._device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            self.logger.info(f"Using device: {self._device}")
        return self._device
    
    def _handle_error(
        self,
        error: Exception,
        operation: str,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Maneja errores de forma consistente"""
        context_str = f" | Context: {context}" if context else ""
        self.logger.error(
            f"Error in {operation}{context_str}: {error}",
            exc_info=True
        )
        raise SocialMediaIdentityCloneError(
            message=f"Error in {operation}: {str(error)}",
            details=context or {}
        ) from error
    
    def _validate_input(self, data: Any, field_name: str) -> None:
        """Valida entrada básica"""
        if data is None:
            raise ValueError(f"{field_name} cannot be None")
    
    def _log_operation(self, operation: str, **kwargs) -> None:
        """Log de operaciones"""
        kwargs_str = ", ".join(f"{k}={v}" for k, v in kwargs.items())
        self.logger.info(f"{operation} | {kwargs_str}")


class BaseMLService(BaseService):
    """Clase base para servicios de ML"""
    
    def __init__(self):
        super().__init__()
        self._model: Optional[torch.nn.Module] = None
        self._model_loaded = False
    
    @property
    def model(self) -> torch.nn.Module:
        """Obtiene el modelo cargado"""
        if not self._model_loaded:
            self._load_model()
        return self._model
    
    @abstractmethod
    def _load_model(self) -> None:
        """Carga el modelo"""
        pass
    
    def _move_to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """Mueve tensor al dispositivo"""
        return tensor.to(self.device)
    
    def _enable_mixed_precision(self) -> bool:
        """Habilita mixed precision si está disponible"""
        return torch.cuda.is_available() and hasattr(torch.cuda, "amp")




