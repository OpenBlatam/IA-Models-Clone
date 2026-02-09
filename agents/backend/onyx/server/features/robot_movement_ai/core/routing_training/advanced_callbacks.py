"""
Advanced Callbacks
==================

Callbacks avanzados para entrenamiento.
"""

import logging
import torch
import numpy as np
from typing import Dict, Any, Optional
import os
from pathlib import Path

from .callbacks import Callback

logger = logging.getLogger(__name__)


class LearningRateFinder(Callback):
    """
    Callback para encontrar learning rate óptimo.
    """
    
    def __init__(self, min_lr: float = 1e-8, max_lr: float = 1.0, num_iterations: int = 100):
        """
        Inicializar LR finder.
        
        Args:
            min_lr: Learning rate mínimo
            max_lr: Learning rate máximo
            num_iterations: Número de iteraciones
        """
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.num_iterations = num_iterations
        self.lrs = []
        self.losses = []
        self.best_lr = None
    
    def on_train_begin(self, trainer):
        """Inicializar LR finder."""
        self.lrs = []
        self.losses = []
        original_lr = trainer.optimizer.param_groups[0]['lr']
        
        # Guardar estado original
        self.original_state = {
            'lr': original_lr,
            'model_state': trainer.model.state_dict(),
            'optimizer_state': trainer.optimizer.state_dict()
        }
    
    def on_train_end(self, trainer):
        """Analizar resultados y sugerir LR."""
        if len(self.losses) > 0:
            # Encontrar LR con mayor pendiente negativa
            losses_smooth = self._smooth(self.losses)
            gradients = np.gradient(losses_smooth)
            
            # LR óptimo: punto con mayor pendiente negativa
            min_grad_idx = np.argmin(gradients)
            self.best_lr = self.lrs[min_grad_idx]
            
            logger.info(f"Learning rate sugerido: {self.best_lr:.6f}")
            
            # Restaurar estado original
            trainer.model.load_state_dict(self.original_state['model_state'])
            trainer.optimizer.load_state_dict(self.original_state['optimizer_state'])
            trainer.optimizer.param_groups[0]['lr'] = self.original_state['lr']
    
    def on_epoch_begin(self, trainer, epoch: int):
        """Actualizar LR exponencialmente."""
        if epoch < self.num_iterations:
            lr = self.min_lr * (self.max_lr / self.min_lr) ** (epoch / self.num_iterations)
            for param_group in trainer.optimizer.param_groups:
                param_group['lr'] = lr
            self.lrs.append(lr)
    
    def on_epoch_end(self, trainer, epoch: int, metrics: Dict[str, float]) -> bool:
        """Registrar pérdida."""
        if epoch < self.num_iterations:
            self.losses.append(metrics.get('train_loss', float('inf')))
        return False
    
    def _smooth(self, values: list, beta: float = 0.98) -> np.ndarray:
        """Suavizar valores."""
        smoothed = []
        last = values[0]
        for value in values:
            last = beta * last + (1 - beta) * value
            smoothed.append(last)
        return np.array(smoothed)


class GradientMonitor(Callback):
    """
    Monitor de gradientes para debugging.
    """
    
    def __init__(self, log_frequency: int = 10, max_grad_norm: float = 10.0):
        """
        Inicializar monitor.
        
        Args:
            log_frequency: Frecuencia de logging
            max_grad_norm: Norma máxima esperada
        """
        self.log_frequency = log_frequency
        self.max_grad_norm = max_grad_norm
        self.grad_norms = []
    
    def on_train_begin(self, trainer):
        """Inicializar."""
        self.grad_norms = []
    
    def on_train_end(self, trainer):
        """Analizar gradientes."""
        if self.grad_norms:
            avg_norm = np.mean(self.grad_norms)
            max_norm = np.max(self.grad_norms)
            logger.info(f"Gradiente promedio: {avg_norm:.4f}, Máximo: {max_norm:.4f}")
    
    def on_epoch_begin(self, trainer, epoch: int):
        """No hacer nada."""
        pass
    
    def on_epoch_end(self, trainer, epoch: int, metrics: Dict[str, float]) -> bool:
        """Monitorear gradientes."""
        if (epoch + 1) % self.log_frequency == 0:
            total_norm = 0.0
            for p in trainer.model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
            
            self.grad_norms.append(total_norm)
            
            if total_norm > self.max_grad_norm:
                logger.warning(f"Gradiente grande detectado: {total_norm:.4f} (época {epoch + 1})")
        
        return False


class ModelEMA(Callback):
    """
    Exponential Moving Average para pesos del modelo.
    """
    
    def __init__(self, decay: float = 0.9999):
        """
        Inicializar EMA.
        
        Args:
            decay: Factor de decaimiento
        """
        self.decay = decay
        self.ema_model = None
    
    def on_train_begin(self, trainer):
        """Inicializar modelo EMA."""
        # Crear copia del modelo
        self.ema_model = type(trainer.model)(trainer.model.config)
        self.ema_model.load_state_dict(trainer.model.state_dict())
        self.ema_model.eval()
    
    def on_train_end(self, trainer):
        """Aplicar pesos EMA al modelo."""
        trainer.model.load_state_dict(self.ema_model.state_dict())
        logger.info("Pesos EMA aplicados al modelo final")
    
    def on_epoch_begin(self, trainer, epoch: int):
        """No hacer nada."""
        pass
    
    def on_epoch_end(self, trainer, epoch: int, metrics: Dict[str, float]) -> bool:
        """Actualizar EMA."""
        with torch.no_grad():
            for ema_param, param in zip(self.ema_model.parameters(), trainer.model.parameters()):
                ema_param.data.mul_(self.decay).add_(param.data, alpha=1 - self.decay)
        return False


class ProfilerCallback(Callback):
    """
    Callback para profiling de rendimiento.
    """
    
    def __init__(self, profile_steps: int = 10, output_dir: str = "./profiles"):
        """
        Inicializar profiler.
        
        Args:
            profile_steps: Número de pasos a perfilar
            output_dir: Directorio de salida
        """
        self.profile_steps = profile_steps
        self.output_dir = output_dir
        self.profiler = None
        os.makedirs(output_dir, exist_ok=True)
    
    def on_train_begin(self, trainer):
        """Inicializar profiler."""
        self.profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA
            ],
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(self.output_dir),
            record_shapes=True,
            profile_memory=True
        )
        self.profiler.start()
    
    def on_train_end(self, trainer):
        """Detener profiler."""
        if self.profiler:
            self.profiler.stop()
            logger.info(f"Profile guardado en: {self.output_dir}")
    
    def on_epoch_begin(self, trainer, epoch: int):
        """Step del profiler."""
        if self.profiler and epoch < self.profile_steps:
            self.profiler.step()
    
    def on_epoch_end(self, trainer, epoch: int, metrics: Dict[str, float]) -> bool:
        """No hacer nada."""
        return False


