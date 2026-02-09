"""
Training Monitor - Modular Training Monitoring
==============================================

Monitoreo modular del proceso de entrenamiento.
"""

import logging
from typing import Dict, Any, Optional, List
import time
from collections import defaultdict
import torch

logger = logging.getLogger(__name__)


class TrainingMonitor:
    """
    Monitor de entrenamiento modular.
    
    Rastrea métricas, tiempos y estadísticas
    durante el entrenamiento.
    """
    
    def __init__(self):
        """Inicializar monitor."""
        self.metrics_history: Dict[str, List[float]] = defaultdict(list)
        self.timings: Dict[str, List[float]] = defaultdict(list)
        self.current_epoch = 0
        self.current_step = 0
        self.start_time: Optional[float] = None
        self.epoch_start_time: Optional[float] = None
        
        logger.info("Training Monitor initialized")
    
    def start_training(self):
        """Iniciar monitoreo de entrenamiento."""
        self.start_time = time.time()
        logger.info("Training monitoring started")
    
    def start_epoch(self, epoch: int):
        """Iniciar época."""
        self.current_epoch = epoch
        self.epoch_start_time = time.time()
    
    def end_epoch(self):
        """Finalizar época."""
        if self.epoch_start_time:
            epoch_time = time.time() - self.epoch_start_time
            self.timings['epoch'].append(epoch_time)
            logger.debug(f"Epoch {self.current_epoch} took {epoch_time:.2f}s")
    
    def log_metric(self, name: str, value: float, step: Optional[int] = None):
        """
        Registrar métrica.
        
        Args:
            name: Nombre de la métrica
            value: Valor
            step: Paso (opcional)
        """
        self.metrics_history[name].append(value)
        if step is not None:
            self.current_step = step
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Registrar múltiples métricas.
        
        Args:
            metrics: Diccionario de métricas
            step: Paso (opcional)
        """
        for name, value in metrics.items():
            self.log_metric(name, value, step)
    
    def log_timing(self, name: str, duration: float):
        """
        Registrar tiempo.
        
        Args:
            name: Nombre de la operación
            duration: Duración en segundos
        """
        self.timings[name].append(duration)
    
    def get_metric_history(self, name: str) -> List[float]:
        """Obtener historial de métrica."""
        return self.metrics_history.get(name, [])
    
    def get_latest_metric(self, name: str) -> Optional[float]:
        """Obtener último valor de métrica."""
        history = self.metrics_history.get(name, [])
        return history[-1] if history else None
    
    def get_average_metric(self, name: str) -> float:
        """Obtener promedio de métrica."""
        history = self.metrics_history.get(name, [])
        return sum(history) / len(history) if history else 0.0
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Obtener resumen de entrenamiento.
        
        Returns:
            Diccionario con resumen
        """
        summary = {
            'epochs': self.current_epoch,
            'steps': self.current_step,
            'metrics': {
                name: {
                    'latest': self.get_latest_metric(name),
                    'average': self.get_average_metric(name),
                    'min': min(values) if values else None,
                    'max': max(values) if values else None
                }
                for name, values in self.metrics_history.items()
            },
            'timings': {
                name: {
                    'total': sum(times),
                    'average': sum(times) / len(times) if times else 0.0,
                    'count': len(times)
                }
                for name, times in self.timings.items()
            }
        }
        
        if self.start_time:
            summary['total_time'] = time.time() - self.start_time
        
        return summary
    
    def reset(self):
        """Resetear monitor."""
        self.metrics_history.clear()
        self.timings.clear()
        self.current_epoch = 0
        self.current_step = 0
        self.start_time = None
        self.epoch_start_time = None


class GPUMonitor:
    """Monitor de GPU."""
    
    def __init__(self):
        """Inicializar monitor de GPU."""
        self.is_available = torch.cuda.is_available()
        self.device_count = torch.cuda.device_count() if self.is_available else 0
    
    def get_memory_usage(self, device: int = 0) -> Dict[str, float]:
        """
        Obtener uso de memoria GPU.
        
        Args:
            device: ID del dispositivo
            
        Returns:
            Diccionario con información de memoria
        """
        if not self.is_available:
            return {}
        
        return {
            'allocated_gb': torch.cuda.memory_allocated(device) / 1024**3,
            'reserved_gb': torch.cuda.memory_reserved(device) / 1024**3,
            'max_allocated_gb': torch.cuda.max_memory_allocated(device) / 1024**3
        }
    
    def get_device_info(self, device: int = 0) -> Dict[str, Any]:
        """
        Obtener información del dispositivo.
        
        Args:
            device: ID del dispositivo
            
        Returns:
            Diccionario con información
        """
        if not self.is_available:
            return {}
        
        return {
            'name': torch.cuda.get_device_name(device),
            'capability': torch.cuda.get_device_capability(device),
            'memory': self.get_memory_usage(device)
        }








