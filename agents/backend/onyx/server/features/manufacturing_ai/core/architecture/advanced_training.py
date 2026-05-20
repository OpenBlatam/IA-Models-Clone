"""
Advanced Training Utilities
===========================

Utilidades avanzadas para entrenamiento: mixed precision, gradient accumulation, etc.
"""

import logging
from typing import Optional, Dict, Any, Callable
from contextlib import contextmanager

try:
    import torch
    import torch.nn as nn
    from torch.cuda.amp import autocast, GradScaler
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    autocast = None
    GradScaler = None

logger = logging.getLogger(__name__)


class MixedPrecisionTrainer:
    """
    Entrenador con mixed precision.
    
    Usa FP16 para acelerar entrenamiento.
    """
    
    def __init__(self, enabled: bool = True, device: Optional[str] = None):
        """
        Inicializar trainer.
        
        Args:
            enabled: Habilitar mixed precision
            device: Dispositivo (cuda/cpu)
        """
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available")
            self.enabled = False
            self.scaler = None
            return
        
        self.enabled = enabled and torch.cuda.is_available()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        if self.enabled:
            self.scaler = GradScaler()
            logger.info("Mixed precision training enabled")
        else:
            self.scaler = None
            logger.info("Mixed precision training disabled")
    
    @contextmanager
    def autocast_context(self):
        """Context manager para autocast."""
        if self.enabled:
            with autocast():
                yield
        else:
            yield
    
    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """
        Escalar pérdida para mixed precision.
        
        Args:
            loss: Pérdida
            
        Returns:
            Pérdida escalada
        """
        if self.enabled and self.scaler:
            return self.scaler.scale(loss)
        return loss
    
    def step_optimizer(self, optimizer: torch.optim.Optimizer):
        """
        Step del optimizador con scaling.
        
        Args:
            optimizer: Optimizador
        """
        if self.enabled and self.scaler:
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            optimizer.step()
    
    def get_state_dict(self) -> Dict[str, Any]:
        """Obtener estado."""
        if self.scaler:
            return {"scaler": self.scaler.state_dict()}
        return {}
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Cargar estado."""
        if self.scaler and "scaler" in state_dict:
            self.scaler.load_state_dict(state_dict["scaler"])


class GradientAccumulator:
    """
    Acumulador de gradientes.
    
    Permite entrenar con batch sizes grandes usando batches pequeños.
    """
    
    def __init__(self, accumulation_steps: int = 1):
        """
        Inicializar acumulador.
        
        Args:
            accumulation_steps: Número de pasos a acumular
        """
        self.accumulation_steps = accumulation_steps
        self.current_step = 0
    
    def should_update(self) -> bool:
        """Verificar si debe actualizar."""
        return (self.current_step + 1) % self.accumulation_steps == 0
    
    def step(self):
        """Incrementar paso."""
        self.current_step += 1
    
    def reset(self):
        """Resetear contador."""
        self.current_step = 0
    
    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """
        Escalar pérdida por accumulation steps.
        
        Args:
            loss: Pérdida
            
        Returns:
            Pérdida escalada
        """
        return loss / self.accumulation_steps


class AdvancedTrainer:
    """
    Entrenador avanzado con todas las optimizaciones.
    
    Combina mixed precision, gradient accumulation, etc.
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: Callable,
        mixed_precision: bool = True,
        accumulation_steps: int = 1,
        gradient_clip: Optional[float] = None,
        device: Optional[str] = None
    ):
        """
        Inicializar entrenador.
        
        Args:
            model: Modelo
            optimizer: Optimizador
            criterion: Función de pérdida
            mixed_precision: Usar mixed precision
            accumulation_steps: Pasos de acumulación
            gradient_clip: Valor para gradient clipping
            device: Dispositivo
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.gradient_clip = gradient_clip
        
        self.mixed_precision = MixedPrecisionTrainer(mixed_precision, device)
        self.gradient_accumulator = GradientAccumulator(accumulation_steps)
        
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
    
    def train_step(
        self,
        inputs: Any,
        targets: Any,
        zero_grad: bool = True
    ) -> Dict[str, float]:
        """
        Paso de entrenamiento.
        
        Args:
            inputs: Entradas
            targets: Targets
            zero_grad: Hacer zero_grad
            
        Returns:
            Métricas del paso
        """
        if zero_grad and self.gradient_accumulator.should_update():
            self.optimizer.zero_grad()
        
        # Forward pass con mixed precision
        with self.mixed_precision.autocast_context():
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
        
        # Escalar pérdida
        loss = self.mixed_precision.scale_loss(loss)
        loss = self.gradient_accumulator.scale_loss(loss)
        
        # Backward
        loss.backward()
        
        # Actualizar si es necesario
        if self.gradient_accumulator.should_update():
            # Gradient clipping
            if self.gradient_clip:
                if self.mixed_precision.enabled:
                    self.mixed_precision.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
            
            # Step
            self.mixed_precision.step_optimizer(self.optimizer)
        
        self.gradient_accumulator.step()
        
        return {
            "loss": loss.item() * self.gradient_accumulator.accumulation_steps,
            "step": self.gradient_accumulator.current_step
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Obtener estadísticas."""
        return {
            "mixed_precision": self.mixed_precision.enabled,
            "accumulation_steps": self.gradient_accumulator.accumulation_steps,
            "current_step": self.gradient_accumulator.current_step,
            "gradient_clip": self.gradient_clip
        }

