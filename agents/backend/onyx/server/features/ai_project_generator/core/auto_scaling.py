"""
Auto Scaling - Auto-escalado para Serverless
===========================================

Auto-escalado automático para funciones serverless:
- Métricas de carga
- Scaling policies
- Predictive scaling
- Cost optimization
"""

import logging
from typing import Optional, Dict, Any, List
from enum import Enum
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class ScalingPolicy(str, Enum):
    """Políticas de escalado"""
    TARGET_TRACKING = "target_tracking"
    STEP_SCALING = "step_scaling"
    PREDICTIVE = "predictive"


class AutoScaler:
    """
    Auto-escalador para funciones serverless.
    """
    
    def __init__(
        self,
        min_capacity: int = 1,
        max_capacity: int = 10,
        target_utilization: float = 0.7,
        policy: ScalingPolicy = ScalingPolicy.TARGET_TRACKING
    ) -> None:
        self.min_capacity = min_capacity
        self.max_capacity = max_capacity
        self.target_utilization = target_utilization
        self.policy = policy
        self.metrics: List[Dict[str, Any]] = []
        self.current_capacity: int = min_capacity
    
    def record_metric(
        self,
        metric_name: str,
        value: float,
        timestamp: Optional[datetime] = None
    ) -> None:
        """Registra métrica"""
        self.metrics.append({
            "name": metric_name,
            "value": value,
            "timestamp": timestamp or datetime.now()
        })
        
        # Mantener solo últimos 1000 métricas
        if len(self.metrics) > 1000:
            self.metrics = self.metrics[-1000:]
    
    def calculate_desired_capacity(self) -> int:
        """Calcula capacidad deseada según política"""
        if not self.metrics:
            return self.min_capacity
        
        if self.policy == ScalingPolicy.TARGET_TRACKING:
            return self._target_tracking_scaling()
        elif self.policy == ScalingPolicy.STEP_SCALING:
            return self._step_scaling()
        elif self.policy == ScalingPolicy.PREDICTIVE:
            return self._predictive_scaling()
        else:
            return self.current_capacity
    
    def _target_tracking_scaling(self) -> int:
        """Target tracking scaling"""
        # Obtener métricas recientes (últimos 5 minutos)
        recent_metrics = [
            m for m in self.metrics
            if (datetime.now() - m["timestamp"]).total_seconds() < 300
        ]
        
        if not recent_metrics:
            return self.current_capacity
        
        avg_utilization = sum(m["value"] for m in recent_metrics) / len(recent_metrics)
        
        if avg_utilization > self.target_utilization * 1.1:
            # Escalar arriba
            new_capacity = min(
                self.max_capacity,
                int(self.current_capacity * 1.5)
            )
        elif avg_utilization < self.target_utilization * 0.9:
            # Escalar abajo
            new_capacity = max(
                self.min_capacity,
                int(self.current_capacity * 0.8)
            )
        else:
            new_capacity = self.current_capacity
        
        return new_capacity
    
    def _step_scaling(self) -> int:
        """Step scaling"""
        recent_metrics = [
            m for m in self.metrics
            if (datetime.now() - m["timestamp"]).total_seconds() < 300
        ]
        
        if not recent_metrics:
            return self.current_capacity
        
        avg_utilization = sum(m["value"] for m in recent_metrics) / len(recent_metrics)
        
        # Step scaling basado en thresholds
        if avg_utilization > 0.9:
            return min(self.max_capacity, self.current_capacity + 3)
        elif avg_utilization > 0.7:
            return min(self.max_capacity, self.current_capacity + 1)
        elif avg_utilization < 0.3:
            return max(self.min_capacity, self.current_capacity - 1)
        else:
            return self.current_capacity
    
    def _predictive_scaling(self) -> int:
        """Predictive scaling usando tendencias"""
        # Implementación simplificada
        # En producción, usaría machine learning para predecir
        recent_metrics = [
            m for m in self.metrics
            if (datetime.now() - m["timestamp"]).total_seconds() < 600
        ]
        
        if len(recent_metrics) < 10:
            return self.current_capacity
        
        # Calcular tendencia
        values = [m["value"] for m in recent_metrics]
        trend = (values[-1] - values[0]) / len(values)
        
        if trend > 0.1:
            # Tendencia creciente, escalar proactivamente
            return min(self.max_capacity, self.current_capacity + 2)
        elif trend < -0.1:
            # Tendencia decreciente, reducir capacidad
            return max(self.min_capacity, self.current_capacity - 1)
        else:
            return self.current_capacity
    
    def update_capacity(self, new_capacity: int) -> None:
        """Actualiza capacidad actual"""
        self.current_capacity = max(
            self.min_capacity,
            min(self.max_capacity, new_capacity)
        )
        logger.info(f"Capacity updated to {self.current_capacity}")
    
    def get_scaling_recommendation(self) -> Dict[str, Any]:
        """Obtiene recomendación de escalado"""
        desired = self.calculate_desired_capacity()
        
        return {
            "current_capacity": self.current_capacity,
            "desired_capacity": desired,
            "recommendation": "scale_up" if desired > self.current_capacity else "scale_down" if desired < self.current_capacity else "no_change",
            "policy": self.policy.value,
            "metrics_count": len(self.metrics)
        }


def get_auto_scaler(
    min_capacity: int = 1,
    max_capacity: int = 10,
    target_utilization: float = 0.7,
    policy: ScalingPolicy = ScalingPolicy.TARGET_TRACKING
) -> AutoScaler:
    """Obtiene auto-escalador"""
    return AutoScaler(
        min_capacity=min_capacity,
        max_capacity=max_capacity,
        target_utilization=target_utilization,
        policy=policy
    )















