"""
Sistema de Auto-Scaling
========================

Sistema para escalar recursos automáticamente según la demanda.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import time

logger = logging.getLogger(__name__)


class ScalingAction(Enum):
    """Acciones de escalado"""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    NO_ACTION = "no_action"


@dataclass
class ScalingMetrics:
    """Métricas para escalado"""
    timestamp: str
    cpu_usage: float
    memory_usage: float
    request_rate: float
    queue_size: int
    active_workers: int


class AutoScaler:
    """
    Auto-scaler inteligente
    
    Proporciona:
    - Escalado basado en métricas
    - Predicción de carga
    - Escalado proactivo
    - Límites configurables
    - Historial de escalado
    """
    
    def __init__(
        self,
        min_workers: int = 1,
        max_workers: int = 10,
        scale_up_threshold: float = 0.8,
        scale_down_threshold: float = 0.3,
        cooldown_period: int = 60
    ):
        """Inicializar auto-scaler"""
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        self.cooldown_period = cooldown_period
        
        self.current_workers = min_workers
        self.last_scale_time = 0
        self.scaling_history: List[Dict[str, Any]] = []
        self.metrics_history: List[ScalingMetrics] = []
        
        logger.info(f"AutoScaler inicializado: {min_workers}-{max_workers} workers")
    
    def record_metrics(
        self,
        cpu_usage: float,
        memory_usage: float,
        request_rate: float,
        queue_size: int
    ):
        """Registrar métricas actuales"""
        metrics = ScalingMetrics(
            timestamp=datetime.now().isoformat(),
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            request_rate=request_rate,
            queue_size=queue_size,
            active_workers=self.current_workers
        )
        
        self.metrics_history.append(metrics)
        
        # Mantener solo últimos 1000 registros
        if len(self.metrics_history) > 1000:
            self.metrics_history = self.metrics_history[-1000:]
    
    def should_scale(self) -> ScalingAction:
        """
        Determinar si se debe escalar
        
        Returns:
            Acción de escalado recomendada
        """
        # Verificar cooldown
        current_time = time.time()
        if current_time - self.last_scale_time < self.cooldown_period:
            return ScalingAction.NO_ACTION
        
        if not self.metrics_history:
            return ScalingAction.NO_ACTION
        
        # Obtener métricas más recientes (últimos 5 minutos)
        recent_metrics = [
            m for m in self.metrics_history[-10:]
            if (datetime.now() - datetime.fromisoformat(m.timestamp)).seconds < 300
        ]
        
        if not recent_metrics:
            return ScalingAction.NO_ACTION
        
        # Calcular promedios
        avg_cpu = sum(m.cpu_usage for m in recent_metrics) / len(recent_metrics)
        avg_memory = sum(m.memory_usage for m in recent_metrics) / len(recent_metrics)
        avg_queue = sum(m.queue_size for m in recent_metrics) / len(recent_metrics)
        
        # Lógica de escalado
        max_metric = max(avg_cpu, avg_memory)
        
        if max_metric > self.scale_up_threshold and self.current_workers < self.max_workers:
            return ScalingAction.SCALE_UP
        elif max_metric < self.scale_down_threshold and avg_queue < 10 and self.current_workers > self.min_workers:
            return ScalingAction.SCALE_DOWN
        
        return ScalingAction.NO_ACTION
    
    def scale_up(self, increment: int = 1) -> bool:
        """
        Escalar hacia arriba
        
        Args:
            increment: Número de workers a agregar
        
        Returns:
            True si el escalado fue exitoso
        """
        new_workers = min(self.current_workers + increment, self.max_workers)
        
        if new_workers == self.current_workers:
            return False
        
        old_workers = self.current_workers
        self.current_workers = new_workers
        self.last_scale_time = time.time()
        
        self.scaling_history.append({
            "action": "scale_up",
            "from": old_workers,
            "to": new_workers,
            "timestamp": datetime.now().isoformat(),
            "reason": "High utilization"
        })
        
        logger.info(f"Escalado hacia arriba: {old_workers} -> {new_workers} workers")
        return True
    
    def scale_down(self, decrement: int = 1) -> bool:
        """
        Escalar hacia abajo
        
        Args:
            decrement: Número de workers a quitar
        
        Returns:
            True si el escalado fue exitoso
        """
        new_workers = max(self.current_workers - decrement, self.min_workers)
        
        if new_workers == self.current_workers:
            return False
        
        old_workers = self.current_workers
        self.current_workers = new_workers
        self.last_scale_time = time.time()
        
        self.scaling_history.append({
            "action": "scale_down",
            "from": old_workers,
            "to": new_workers,
            "timestamp": datetime.now().isoformat(),
            "reason": "Low utilization"
        })
        
        logger.info(f"Escalado hacia abajo: {old_workers} -> {new_workers} workers")
        return True
    
    def get_scaling_recommendation(self) -> Dict[str, Any]:
        """Obtener recomendación de escalado"""
        action = self.should_scale()
        
        recommendation = {
            "action": action.value,
            "current_workers": self.current_workers,
            "min_workers": self.min_workers,
            "max_workers": self.max_workers
        }
        
        if self.metrics_history:
            latest = self.metrics_history[-1]
            recommendation["current_metrics"] = {
                "cpu_usage": latest.cpu_usage,
                "memory_usage": latest.memory_usage,
                "request_rate": latest.request_rate,
                "queue_size": latest.queue_size
            }
        
        return recommendation
    
    def get_scaling_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Obtener historial de escalado"""
        return self.scaling_history[-limit:]


# Instancia global
_auto_scaler: Optional[AutoScaler] = None


def get_auto_scaler(
    min_workers: int = 1,
    max_workers: int = 10
) -> AutoScaler:
    """Obtener instancia global del auto-scaler"""
    global _auto_scaler
    if _auto_scaler is None:
        _auto_scaler = AutoScaler(min_workers, max_workers)
    return _auto_scaler
















