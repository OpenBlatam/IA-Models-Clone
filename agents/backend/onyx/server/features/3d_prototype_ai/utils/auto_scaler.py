"""
Auto Scaler - Sistema de auto-scaling
=======================================
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from collections import deque

logger = logging.getLogger(__name__)


class AutoScaler:
    """Sistema de auto-scaling"""
    
    def __init__(self, min_instances: int = 1, max_instances: int = 10):
        self.min_instances = min_instances
        self.max_instances = max_instances
        self.current_instances = min_instances
        
        self.metrics_history: deque = deque(maxlen=100)
        self.scaling_history: List[Dict[str, Any]] = []
        
        # Thresholds
        self.scale_up_threshold = 0.8  # 80% de uso
        self.scale_down_threshold = 0.3  # 30% de uso
        self.cooldown_period = timedelta(minutes=5)
        self.last_scale_time: Optional[datetime] = None
    
    def record_metric(self, cpu_usage: float, memory_usage: float, 
                     request_rate: float, queue_size: int = 0):
        """Registra métricas para decisión de scaling"""
        metric = {
            "timestamp": datetime.now(),
            "cpu_usage": cpu_usage,
            "memory_usage": memory_usage,
            "request_rate": request_rate,
            "queue_size": queue_size
        }
        
        self.metrics_history.append(metric)
        
        # Evaluar si necesitamos scaling
        self._evaluate_scaling()
    
    def _evaluate_scaling(self):
        """Evalúa si se necesita scaling"""
        if len(self.metrics_history) < 5:
            return
        
        # Verificar cooldown
        if self.last_scale_time:
            if datetime.now() - self.last_scale_time < self.cooldown_period:
                return
        
        # Calcular métricas promedio
        recent_metrics = list(self.metrics_history)[-5:]
        avg_cpu = sum(m["cpu_usage"] for m in recent_metrics) / len(recent_metrics)
        avg_memory = sum(m["memory_usage"] for m in recent_metrics) / len(recent_metrics)
        avg_queue = sum(m["queue_size"] for m in recent_metrics) / len(recent_metrics)
        
        # Decisión de scale up
        if (avg_cpu > self.scale_up_threshold or 
            avg_memory > self.scale_up_threshold or
            avg_queue > 100) and self.current_instances < self.max_instances:
            self._scale_up()
        
        # Decisión de scale down
        elif (avg_cpu < self.scale_down_threshold and 
              avg_memory < self.scale_down_threshold and
              avg_queue < 10) and self.current_instances > self.min_instances:
            self._scale_down()
    
    def _scale_up(self):
        """Escala hacia arriba"""
        old_instances = self.current_instances
        self.current_instances = min(self.current_instances + 1, self.max_instances)
        
        if self.current_instances > old_instances:
            self.scaling_history.append({
                "action": "scale_up",
                "from": old_instances,
                "to": self.current_instances,
                "timestamp": datetime.now().isoformat()
            })
            self.last_scale_time = datetime.now()
            logger.info(f"Auto-scaling UP: {old_instances} -> {self.current_instances}")
    
    def _scale_down(self):
        """Escala hacia abajo"""
        old_instances = self.current_instances
        self.current_instances = max(self.current_instances - 1, self.min_instances)
        
        if self.current_instances < old_instances:
            self.scaling_history.append({
                "action": "scale_down",
                "from": old_instances,
                "to": self.current_instances,
                "timestamp": datetime.now().isoformat()
            })
            self.last_scale_time = datetime.now()
            logger.info(f"Auto-scaling DOWN: {old_instances} -> {self.current_instances}")
    
    def get_scaling_status(self) -> Dict[str, Any]:
        """Obtiene estado de scaling"""
        recent_metrics = list(self.metrics_history)[-5:] if self.metrics_history else []
        
        return {
            "current_instances": self.current_instances,
            "min_instances": self.min_instances,
            "max_instances": self.max_instances,
            "recent_metrics": recent_metrics,
            "scaling_history": self.scaling_history[-10:],
            "last_scale": self.scaling_history[-1] if self.scaling_history else None
        }
    
    def manual_scale(self, target_instances: int):
        """Escalado manual"""
        if target_instances < self.min_instances or target_instances > self.max_instances:
            raise ValueError(f"Instancias deben estar entre {self.min_instances} y {self.max_instances}")
        
        old_instances = self.current_instances
        self.current_instances = target_instances
        
        self.scaling_history.append({
            "action": "manual_scale",
            "from": old_instances,
            "to": target_instances,
            "timestamp": datetime.now().isoformat()
        })
        
        logger.info(f"Escalado manual: {old_instances} -> {target_instances}")




