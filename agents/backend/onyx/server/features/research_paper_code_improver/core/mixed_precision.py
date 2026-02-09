"""
Mixed Precision Training Manager - Gestor de entrenamiento en precisión mixta
==============================================================================
"""

import logging
import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class PrecisionMode(Enum):
    """Modos de precisión"""
    FP32 = "fp32"  # Full precision
    FP16 = "fp16"  # Half precision
    BF16 = "bf16"  # Brain float 16
    MIXED = "mixed"  # Mixed precision (automatic)


@dataclass
class MixedPrecisionConfig:
    """Configuración de precisión mixta"""
    mode: PrecisionMode = PrecisionMode.MIXED
    enabled: bool = True
    init_scale: float = 2.**16
    growth_factor: float = 2.0
    backoff_factor: float = 0.5
    growth_interval: int = 2000
    enabled_for_forward: bool = True
    enabled_for_backward: bool = True


class MixedPrecisionManager:
    """Gestor de precisión mixta"""
    
    def __init__(self, config: MixedPrecisionConfig):
        self.config = config
        self.scaler: Optional[torch.cuda.amp.GradScaler] = None
        
        if config.enabled and config.mode == PrecisionMode.MIXED:
            if torch.cuda.is_available():
                self.scaler = torch.cuda.amp.GradScaler(
                    init_scale=config.init_scale,
                    growth_factor=config.growth_factor,
                    backoff_factor=config.backoff_factor,
                    growth_interval=config.growth_interval
                )
                logger.info("GradScaler inicializado para mixed precision")
            else:
                logger.warning("CUDA no disponible, mixed precision deshabilitado")
                self.config.enabled = False
    
    def autocast_context(self):
        """Crea contexto de autocast"""
        if not self.config.enabled:
            return torch.cuda.amp.autocast(enabled=False)
        
        if self.config.mode == PrecisionMode.FP16:
            return torch.cuda.amp.autocast(enabled=True, dtype=torch.float16)
        elif self.config.mode == PrecisionMode.BF16:
            return torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16)
        else:  # MIXED
            return torch.cuda.amp.autocast(enabled=True)
    
    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Escala la pérdida para mixed precision"""
        if self.scaler is not None:
            return self.scaler.scale(loss)
        return loss
    
    def step_optimizer(
        self,
        optimizer: torch.optim.Optimizer,
        loss: torch.Tensor
    ):
        """Ejecuta step del optimizador con mixed precision"""
        if self.scaler is not None:
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            optimizer.step()
    
    def backward(self, loss: torch.Tensor):
        """Backward pass con mixed precision"""
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
    
    def get_scale(self) -> float:
        """Obtiene el scale actual"""
        if self.scaler is not None:
            return self.scaler.get_scale()
        return 1.0
    
    def update_scale(self):
        """Actualiza el scale"""
        if self.scaler is not None:
            self.scaler.update()
    
    def state_dict(self) -> Dict[str, Any]:
        """Obtiene el estado del scaler"""
        if self.scaler is not None:
            return self.scaler.state_dict()
        return {}
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Carga el estado del scaler"""
        if self.scaler is not None:
            self.scaler.load_state_dict(state_dict)
    
    def convert_model_to_half(self, model: nn.Module) -> nn.Module:
        """Convierte modelo a FP16"""
        if self.config.mode == PrecisionMode.FP16:
            return model.half()
        return model
    
    def convert_model_to_bfloat16(self, model: nn.Module) -> nn.Module:
        """Convierte modelo a BF16"""
        if self.config.mode == PrecisionMode.BF16:
            return model.to(torch.bfloat16)
        return model




