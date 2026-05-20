"""
Auto Scaler
===========

Sistema de auto-escalado inteligente para el generador de proyectos.
"""

import asyncio
import time
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class ScaleAction(Enum):
    """Acciones de escalado."""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    NO_ACTION = "no_action"


@dataclass
class ScalingDecision:
    """Decisión de escalado."""
    action: ScaleAction
    reason: str
    current_workers: int
    recommended_workers: int
    confidence: float
    timestamp: datetime


class AutoScaler:
    """Sistema de auto-escalado inteligente."""
    
    def __init__(
        self,
        min_workers: int = 1,
        max_workers: int = 10,
        target_cpu: float = 70.0,
        target_memory: float = 80.0,
        scale_up_threshold: float = 85.0,
        scale_down_threshold: float = 50.0
    ):
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.target_cpu = target_cpu
        self.target_memory = target_memory
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        
        self.current_workers = min_workers
        self.scaling_history: List[ScalingDecision] = []
        self.metrics_history: List[Dict[str, Any]] = []
        self.scale_callbacks: List[Callable] = []
    
    def register_scale_callback(self, callback: Callable) -> None:
        """Registra callback para cuando se escala."""
        self.scale_callbacks.append(callback)
    
    def record_metrics(
        self,
        cpu_percent: float,
        memory_percent: float,
        queue_size: int,
        active_tasks: int
    ) -> None:
        """Registra métricas del sistema."""
        self.metrics_history.append({
            "cpu_percent": cpu_percent,
            "memory_percent": memory_percent,
            "queue_size": queue_size,
            "active_tasks": active_tasks,
            "timestamp": datetime.now()
        })
        
        # Mantener solo últimos 100
        if len(self.metrics_history) > 100:
            self.metrics_history = self.metrics_history[-100:]
    
    def should_scale_up(self, metrics: Dict[str, float]) -> bool:
        """Determina si se debe escalar hacia arriba."""
        cpu = metrics.get("cpu_percent", 0)
        memory = metrics.get("memory_percent", 0)
        queue_size = metrics.get("queue_size", 0)
        active_tasks = metrics.get("active_tasks", 0)
        
        # Escalar si CPU o memoria están altos
        if cpu > self.scale_up_threshold or memory > self.scale_up_threshold:
            return True
        
        # Escalar si hay mucha cola y tareas activas
        if queue_size > 10 and active_tasks >= self.current_workers:
            return True
        
        return False
    
    def should_scale_down(self, metrics: Dict[str, float]) -> bool:
        """Determina si se debe escalar hacia abajo."""
        if self.current_workers <= self.min_workers:
            return False
        
        cpu = metrics.get("cpu_percent", 0)
        memory = metrics.get("memory_percent", 0)
        queue_size = metrics.get("queue_size", 0)
        active_tasks = metrics.get("active_tasks", 0)
        
        # Escalar hacia abajo si recursos están bajos
        if (cpu < self.scale_down_threshold and 
            memory < self.scale_down_threshold and
            queue_size < 2 and
            active_tasks < self.current_workers * 0.5):
            return True
        
        return False
    
    def calculate_optimal_workers(self, metrics: Dict[str, float]) -> int:
        """Calcula número óptimo de workers."""
        cpu = metrics.get("cpu_percent", 0)
        memory = metrics.get("memory_percent", 0)
        queue_size = metrics.get("queue_size", 0)
        
        # Calcular basado en CPU
        cpu_workers = int((cpu / self.target_cpu) * self.current_workers) if self.target_cpu > 0 else self.current_workers
        
        # Calcular basado en memoria
        memory_workers = int((memory / self.target_memory) * self.current_workers) if self.target_memory > 0 else self.current_workers
        
        # Calcular basado en cola (1 worker por cada 5 items en cola)
        queue_workers = max(1, queue_size // 5)
        
        # Tomar el máximo pero limitado
        optimal = max(cpu_workers, memory_workers, queue_workers)
        optimal = min(optimal, self.max_workers)
        optimal = max(optimal, self.min_workers)
        
        return optimal
    
    def make_decision(self) -> ScalingDecision:
        """Toma decisión de escalado."""
        if not self.metrics_history:
            return ScalingDecision(
                action=ScaleAction.NO_ACTION,
                reason="No metrics available",
                current_workers=self.current_workers,
                recommended_workers=self.current_workers,
                confidence=0.0,
                timestamp=datetime.now()
            )
        
        # Usar métricas más recientes
        recent_metrics = self.metrics_history[-5:]
        avg_metrics = {
            "cpu_percent": sum(m["cpu_percent"] for m in recent_metrics) / len(recent_metrics),
            "memory_percent": sum(m["memory_percent"] for m in recent_metrics) / len(recent_metrics),
            "queue_size": sum(m["queue_size"] for m in recent_metrics) / len(recent_metrics),
            "active_tasks": sum(m["active_tasks"] for m in recent_metrics) / len(recent_metrics)
        }
        
        optimal_workers = self.calculate_optimal_workers(avg_metrics)
        
        # Determinar acción
        if self.should_scale_up(avg_metrics):
            action = ScaleAction.SCALE_UP
            recommended = min(optimal_workers, self.max_workers)
            reason = f"High resource usage: CPU={avg_metrics['cpu_percent']:.1f}%, Memory={avg_metrics['memory_percent']:.1f}%"
            confidence = 0.8
        elif self.should_scale_down(avg_metrics):
            action = ScaleAction.SCALE_DOWN
            recommended = max(optimal_workers, self.min_workers)
            reason = f"Low resource usage: CPU={avg_metrics['cpu_percent']:.1f}%, Memory={avg_metrics['memory_percent']:.1f}%"
            confidence = 0.7
        else:
            action = ScaleAction.NO_ACTION
            recommended = self.current_workers
            reason = "Resources within target range"
            confidence = 0.9
        
        decision = ScalingDecision(
            action=action,
            reason=reason,
            current_workers=self.current_workers,
            recommended_workers=recommended,
            confidence=confidence,
            timestamp=datetime.now()
        )
        
        self.scaling_history.append(decision)
        if len(self.scaling_history) > 100:
            self.scaling_history = self.scaling_history[-100:]
        
        return decision
    
    async def scale(self, decision: ScalingDecision) -> bool:
        """Ejecuta escalado."""
        if decision.action == ScaleAction.NO_ACTION:
            return False
        
        old_workers = self.current_workers
        self.current_workers = decision.recommended_workers
        
        # Notificar callbacks
        for callback in self.scale_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(old_workers, self.current_workers, decision)
                else:
                    callback(old_workers, self.current_workers, decision)
            except Exception as e:
                logger.error(f"Error in scale callback: {e}")
        
        logger.info(f"Scaled from {old_workers} to {self.current_workers} workers: {decision.reason}")
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del scaler."""
        recent_decisions = self.scaling_history[-10:] if self.scaling_history else []
        
        return {
            "current_workers": self.current_workers,
            "min_workers": self.min_workers,
            "max_workers": self.max_workers,
            "recent_decisions": [
                {
                    "action": d.action.value,
                    "reason": d.reason,
                    "workers": d.recommended_workers,
                    "confidence": d.confidence,
                    "timestamp": d.timestamp.isoformat()
                }
                for d in recent_decisions
            ],
            "total_scales": len([d for d in self.scaling_history if d.action != ScaleAction.NO_ACTION])
        }


# Factory function
_auto_scaler = None

def get_auto_scaler() -> AutoScaler:
    """Obtiene instancia global del auto scaler."""
    global _auto_scaler
    if _auto_scaler is None:
        _auto_scaler = AutoScaler()
    return _auto_scaler


