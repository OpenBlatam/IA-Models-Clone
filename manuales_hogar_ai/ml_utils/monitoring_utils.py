"""
Monitoring Utils - Utilidades de Monitoreo
===========================================

Utilidades para monitorear entrenamiento y modelos.
"""

import logging
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Callable, Any
import time
from collections import defaultdict, deque
import numpy as np
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class TrainingMetrics:
    """Métricas de entrenamiento."""
    epoch: int = 0
    step: int = 0
    loss: float = 0.0
    accuracy: float = 0.0
    learning_rate: float = 0.0
    gradient_norm: float = 0.0
    timestamp: float = field(default_factory=time.time)


class TrainingMonitor:
    """
    Monitor de entrenamiento con métricas en tiempo real.
    """
    
    def __init__(
        self,
        window_size: int = 100,
        log_interval: int = 10
    ):
        """
        Inicializar monitor.
        
        Args:
            window_size: Tamaño de ventana para promedios
            log_interval: Intervalo de logging
        """
        self.window_size = window_size
        self.log_interval = log_interval
        self.metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self.current_metrics = TrainingMetrics()
        self.start_time = time.time()
    
    def update(
        self,
        loss: float,
        accuracy: Optional[float] = None,
        learning_rate: Optional[float] = None,
        gradient_norm: Optional[float] = None,
        step: Optional[int] = None
    ):
        """
        Actualizar métricas.
        
        Args:
            loss: Loss actual
            accuracy: Accuracy (opcional)
            learning_rate: Learning rate (opcional)
            gradient_norm: Norma de gradiente (opcional)
            step: Paso actual (opcional)
        """
        if step is not None:
            self.current_metrics.step = step
        
        self.current_metrics.loss = loss
        if accuracy is not None:
            self.current_metrics.accuracy = accuracy
        if learning_rate is not None:
            self.current_metrics.learning_rate = learning_rate
        if gradient_norm is not None:
            self.current_metrics.gradient_norm = gradient_norm
        
        # Guardar en historial
        self.metrics_history['loss'].append(loss)
        if accuracy is not None:
            self.metrics_history['accuracy'].append(accuracy)
        if learning_rate is not None:
            self.metrics_history['learning_rate'].append(learning_rate)
        if gradient_norm is not None:
            self.metrics_history['gradient_norm'].append(gradient_norm)
        
        # Logging periódico
        if self.current_metrics.step % self.log_interval == 0:
            self._log_metrics()
    
    def _log_metrics(self):
        """Loggear métricas actuales."""
        avg_loss = np.mean(self.metrics_history['loss'])
        metrics_str = f"Step {self.current_metrics.step}, Loss: {avg_loss:.4f}"
        
        if 'accuracy' in self.metrics_history:
            avg_acc = np.mean(self.metrics_history['accuracy'])
            metrics_str += f", Acc: {avg_acc:.4f}"
        
        if 'learning_rate' in self.metrics_history:
            lr = self.current_metrics.learning_rate
            metrics_str += f", LR: {lr:.6f}"
        
        logger.info(metrics_str)
    
    def get_averages(self) -> Dict[str, float]:
        """
        Obtener promedios de métricas.
        
        Returns:
            Diccionario con promedios
        """
        averages = {}
        for metric_name, values in self.metrics_history.items():
            if len(values) > 0:
                averages[metric_name] = np.mean(values)
        return averages
    
    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Obtener estadísticas de métricas.
        
        Returns:
            Diccionario con estadísticas
        """
        stats = {}
        for metric_name, values in self.metrics_history.items():
            if len(values) > 0:
                stats[metric_name] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values))
                }
        return stats


class GradientMonitor:
    """
    Monitor de gradientes.
    """
    
    def __init__(self, model: nn.Module):
        """
        Inicializar monitor de gradientes.
        
        Args:
            model: Modelo a monitorear
        """
        self.model = model
        self.gradient_norms: Dict[str, List[float]] = {}
        self.gradient_means: Dict[str, List[float]] = {}
    
    def compute_gradient_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Calcular estadísticas de gradientes.
        
        Returns:
            Estadísticas por parámetro
        """
        stats = {}
        
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_mean = param.grad.mean().item()
                
                if name not in self.gradient_norms:
                    self.gradient_norms[name] = []
                    self.gradient_means[name] = []
                
                self.gradient_norms[name].append(grad_norm)
                self.gradient_means[name].append(grad_mean)
                
                stats[name] = {
                    'norm': grad_norm,
                    'mean': grad_mean,
                    'std': param.grad.std().item()
                }
        
        return stats
    
    def detect_vanishing_gradients(self, threshold: float = 1e-6) -> List[str]:
        """
        Detectar gradientes que desaparecen.
        
        Args:
            threshold: Umbral
            
        Returns:
            Lista de parámetros con gradientes pequeños
        """
        vanishing = []
        for name, norms in self.gradient_norms.items():
            if len(norms) > 0 and np.mean(norms) < threshold:
                vanishing.append(name)
        return vanishing
    
    def detect_exploding_gradients(self, threshold: float = 100.0) -> List[str]:
        """
        Detectar gradientes que explotan.
        
        Args:
            threshold: Umbral
            
        Returns:
            Lista de parámetros con gradientes grandes
        """
        exploding = []
        for name, norms in self.gradient_norms.items():
            if len(norms) > 0 and np.mean(norms) > threshold:
                exploding.append(name)
        return exploding


class ModelHealthMonitor:
    """
    Monitor de salud del modelo.
    """
    
    def __init__(self, model: nn.Module):
        """
        Inicializar monitor de salud.
        
        Args:
            model: Modelo a monitorear
        """
        self.model = model
        self.weight_stats: Dict[str, List[Dict]] = {}
    
    def check_weights(self) -> Dict[str, Dict[str, float]]:
        """
        Verificar estadísticas de pesos.
        
        Returns:
            Estadísticas por capa
        """
        stats = {}
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                weight_data = param.data.cpu().numpy()
                
                stats[name] = {
                    'mean': float(np.mean(weight_data)),
                    'std': float(np.std(weight_data)),
                    'min': float(np.min(weight_data)),
                    'max': float(np.max(weight_data)),
                    'nan_count': int(np.isnan(weight_data).sum()),
                    'inf_count': int(np.isinf(weight_data).sum())
                }
        
        return stats
    
    def detect_dead_neurons(self, threshold: float = 1e-6) -> List[str]:
        """
        Detectar neuronas muertas.
        
        Args:
            threshold: Umbral
            
        Returns:
            Lista de capas con neuronas muertas
        """
        dead_neurons = []
        
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                weight = module.weight.data
                # Verificar si hay filas/columnas con norma muy pequeña
                if len(weight.shape) == 2:
                    norms = weight.norm(dim=1)
                    if (norms < threshold).any():
                        dead_neurons.append(name)
        
        return dead_neurons
    
    def check_for_nan_inf(self) -> Dict[str, Dict[str, int]]:
        """
        Verificar NaN e Inf en pesos.
        
        Returns:
            Diccionario con conteos de NaN/Inf
        """
        issues = {}
        
        for name, param in self.model.named_parameters():
            weight_data = param.data.cpu().numpy()
            nan_count = np.isnan(weight_data).sum()
            inf_count = np.isinf(weight_data).sum()
            
            if nan_count > 0 or inf_count > 0:
                issues[name] = {
                    'nan_count': int(nan_count),
                    'inf_count': int(inf_count)
                }
        
        return issues


class PerformanceMonitor:
    """
    Monitor de rendimiento.
    """
    
    def __init__(self):
        """Inicializar monitor de rendimiento."""
        self.timings: Dict[str, List[float]] = defaultdict(list)
        self.memory_usage: List[float] = []
    
    def time_operation(self, operation_name: str):
        """
        Decorador para medir tiempo de operación.
        
        Args:
            operation_name: Nombre de la operación
        """
        def decorator(func):
            def wrapper(*args, **kwargs):
                start = time.time()
                result = func(*args, **kwargs)
                elapsed = time.time() - start
                self.timings[operation_name].append(elapsed)
                return result
            return wrapper
        return decorator
    
    def get_timing_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Obtener estadísticas de tiempos.
        
        Returns:
            Estadísticas por operación
        """
        stats = {}
        for op_name, times in self.timings.items():
            if len(times) > 0:
                stats[op_name] = {
                    'mean': float(np.mean(times)),
                    'std': float(np.std(times)),
                    'min': float(np.min(times)),
                    'max': float(np.max(times)),
                    'total': float(np.sum(times))
                }
        return stats
    
    def get_memory_usage(self) -> Optional[float]:
        """
        Obtener uso de memoria GPU.
        
        Returns:
            Uso de memoria en MB
        """
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024**2
        return None




