"""
Base Model Service - Servicio base para modelos
================================================

Clase base con funcionalidades comunes para todos los servicios de modelos.
Sigue mejores prácticas de PyTorch y deep learning.
"""

import logging
import torch
import torch.nn as nn
from typing import Dict, Optional, Any, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass
import warnings

logger = logging.getLogger(__name__)


@dataclass
class DeviceConfig:
    """Configuración de dispositivo"""
    device: torch.device
    use_mixed_precision: bool = True
    torch_dtype: torch.dtype = torch.float16
    enable_cudnn_benchmark: bool = True
    enable_deterministic: bool = False


class BaseModelService(ABC):
    """Clase base para servicios de modelos"""
    
    def __init__(
        self,
        device_config: Optional[DeviceConfig] = None,
        enable_anomaly_detection: bool = False
    ):
        """
        Inicializar servicio base.
        
        Args:
            device_config: Configuración de dispositivo
            enable_anomaly_detection: Habilitar detección de anomalías en autograd
        """
        self.device_config = device_config or self._get_default_device_config()
        self.models: Dict[str, nn.Module] = {}
        self.enable_anomaly_detection = enable_anomaly_detection
        
        # Setup device optimizations
        self._setup_device_optimizations()
        
        logger.info(
            f"{self.__class__.__name__} initialized on device: {self.device_config.device}"
        )
    
    def _get_default_device_config(self) -> DeviceConfig:
        """Obtener configuración de dispositivo por defecto"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        use_mixed_precision = device.type == "cuda"
        
        return DeviceConfig(
            device=device,
            use_mixed_precision=use_mixed_precision,
            torch_dtype=torch.float16 if use_mixed_precision else torch.float32,
        )
    
    def _setup_device_optimizations(self) -> None:
        """Configurar optimizaciones de dispositivo"""
        if self.device_config.device.type == "cuda":
            # Enable cuDNN benchmark for faster convolutions
            if self.device_config.enable_cudnn_benchmark:
                torch.backends.cudnn.benchmark = True
            
            # Set deterministic mode if requested
            if self.device_config.enable_deterministic:
                torch.backends.cudnn.deterministic = True
                torch.use_deterministic_algorithms(True)
                warnings.warn(
                    "Deterministic mode enabled. This may reduce performance.",
                    UserWarning
                )
    
    def move_to_device(self, tensor_or_model: Union[torch.Tensor, nn.Module]) -> Any:
        """Mover tensor o modelo al dispositivo configurado"""
        return tensor_or_model.to(self.device_config.device)
    
    def get_mixed_precision_context(self):
        """Obtener contexto de mixed precision si está habilitado"""
        if self.device_config.use_mixed_precision and self.device_config.device.type == "cuda":
            return torch.cuda.amp.autocast(dtype=self.device_config.torch_dtype)
        return torch.cuda.amp.autocast(enabled=False)
    
    def enable_anomaly_detection_mode(self) -> None:
        """Habilitar modo de detección de anomalías"""
        if self.enable_anomaly_detection:
            torch.autograd.set_detect_anomaly(True)
            logger.warning("Anomaly detection enabled. This will slow down training.")
    
    def disable_anomaly_detection_mode(self) -> None:
        """Deshabilitar modo de detección de anomalías"""
        torch.autograd.set_detect_anomaly(False)
    
    def validate_model(self, model: nn.Module) -> bool:
        """Validar que el modelo sea válido"""
        try:
            # Check if model has parameters
            if not list(model.parameters()):
                logger.warning("Model has no parameters")
                return False
            
            # Check if model is on correct device
            first_param = next(model.parameters())
            if first_param.device != self.device_config.device:
                logger.warning(
                    f"Model device ({first_param.device}) doesn't match "
                    f"config device ({self.device_config.device})"
                )
            
            return True
        except Exception as e:
            logger.error(f"Error validating model: {e}")
            return False
    
    def get_model_info(self, model: nn.Module) -> Dict[str, Any]:
        """Obtener información del modelo"""
        try:
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            # Get model size in MB
            param_size = sum(p.numel() * p.element_size() for p in model.parameters())
            buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
            model_size_mb = (param_size + buffer_size) / (1024 ** 2)
            
            return {
                "total_parameters": total_params,
                "trainable_parameters": trainable_params,
                "frozen_parameters": total_params - trainable_params,
                "model_size_mb": round(model_size_mb, 2),
                "device": str(next(model.parameters()).device),
            }
        except Exception as e:
            logger.error(f"Error getting model info: {e}")
            return {"error": str(e)}
    
    def save_model_checkpoint(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        epoch: Optional[int] = None,
        loss: Optional[float] = None,
        filepath: str = "checkpoint.pt"
    ) -> bool:
        """Guardar checkpoint del modelo"""
        try:
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "loss": loss,
            }
            
            if optimizer is not None:
                checkpoint["optimizer_state_dict"] = optimizer.state_dict()
            
            torch.save(checkpoint, filepath)
            logger.info(f"Model checkpoint saved to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error saving checkpoint: {e}")
            return False
    
    def load_model_checkpoint(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        filepath: str = "checkpoint.pt",
        map_location: Optional[str] = None
    ) -> Dict[str, Any]:
        """Cargar checkpoint del modelo"""
        try:
            map_loc = map_location or str(self.device_config.device)
            checkpoint = torch.load(filepath, map_location=map_loc)
            
            model.load_state_dict(checkpoint["model_state_dict"])
            
            if optimizer is not None and "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            
            result = {
                "epoch": checkpoint.get("epoch"),
                "loss": checkpoint.get("loss"),
                "loaded": True,
            }
            
            logger.info(f"Model checkpoint loaded from {filepath}")
            return result
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            return {"loaded": False, "error": str(e)}
    
    @abstractmethod
    def create_model(self, *args, **kwargs) -> nn.Module:
        """Crear modelo (debe ser implementado por subclases)"""
        pass




