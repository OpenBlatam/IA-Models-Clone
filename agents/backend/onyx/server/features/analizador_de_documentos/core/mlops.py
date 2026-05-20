"""
Sistema de MLOps (Machine Learning Operations)
================================================

Sistema para monitoreo y gestión de modelos en producción.
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from collections import defaultdict
from enum import Enum

logger = logging.getLogger(__name__)


class ModelHealth(Enum):
    """Salud del modelo"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class ModelPerformance:
    """Rendimiento del modelo"""
    model_id: str
    timestamp: str
    accuracy: float
    latency: float
    throughput: float
    error_rate: float
    health: ModelHealth


class MLOpsManager:
    """
    Gestor de MLOps
    
    Proporciona:
    - Monitoreo de modelos en producción
    - Detección de drift
    - Alertas de degradación
    - Métricas de rendimiento
    - Health checks
    """
    
    def __init__(self):
        """Inicializar gestor"""
        self.model_performance: Dict[str, List[ModelPerformance]] = defaultdict(list)
        self.model_health: Dict[str, ModelHealth] = {}
        self.thresholds: Dict[str, Dict[str, float]] = {
            "default": {
                "min_accuracy": 0.7,
                "max_latency": 1.0,
                "max_error_rate": 0.1
            }
        }
        logger.info("MLOpsManager inicializado")
    
    def record_performance(
        self,
        model_id: str,
        accuracy: float,
        latency: float,
        throughput: float,
        error_rate: float
    ):
        """Registrar rendimiento del modelo"""
        health = self._calculate_health(model_id, accuracy, latency, error_rate)
        
        performance = ModelPerformance(
            model_id=model_id,
            timestamp=datetime.now().isoformat(),
            accuracy=accuracy,
            latency=latency,
            throughput=throughput,
            error_rate=error_rate,
            health=health
        )
        
        self.model_performance[model_id].append(performance)
        self.model_health[model_id] = health
        
        # Mantener solo últimos 1000 registros
        if len(self.model_performance[model_id]) > 1000:
            self.model_performance[model_id] = self.model_performance[model_id][-1000:]
        
        logger.debug(f"Rendimiento registrado para {model_id}: {health.value}")
    
    def _calculate_health(
        self,
        model_id: str,
        accuracy: float,
        latency: float,
        error_rate: float
    ) -> ModelHealth:
        """Calcular salud del modelo"""
        thresholds = self.thresholds.get(model_id, self.thresholds["default"])
        
        if (accuracy < thresholds["min_accuracy"] or
            latency > thresholds["max_latency"] or
            error_rate > thresholds["max_error_rate"]):
            return ModelHealth.UNHEALTHY
        elif (accuracy < thresholds["min_accuracy"] * 1.1 or
              latency > thresholds["max_latency"] * 0.9 or
              error_rate > thresholds["max_error_rate"] * 0.9):
            return ModelHealth.DEGRADED
        else:
            return ModelHealth.HEALTHY
    
    def get_model_health(self, model_id: str) -> Optional[ModelHealth]:
        """Obtener salud del modelo"""
        return self.model_health.get(model_id)
    
    def get_performance_stats(self, model_id: str, hours: int = 24) -> Dict[str, Any]:
        """Obtener estadísticas de rendimiento"""
        if model_id not in self.model_performance:
            return {}
        
        cutoff_time = datetime.now().timestamp() - (hours * 3600)
        recent_perfs = [
            p for p in self.model_performance[model_id]
            if datetime.fromisoformat(p.timestamp).timestamp() > cutoff_time
        ]
        
        if not recent_perfs:
            return {}
        
        accuracies = [p.accuracy for p in recent_perfs]
        latencies = [p.latency for p in recent_perfs]
        error_rates = [p.error_rate for p in recent_perfs]
        
        return {
            "model_id": model_id,
            "period_hours": hours,
            "samples": len(recent_perfs),
            "accuracy": {
                "mean": sum(accuracies) / len(accuracies),
                "min": min(accuracies),
                "max": max(accuracies)
            },
            "latency": {
                "mean": sum(latencies) / len(latencies),
                "min": min(latencies),
                "max": max(latencies)
            },
            "error_rate": {
                "mean": sum(error_rates) / len(error_rates),
                "min": min(error_rates),
                "max": max(error_rates)
            },
            "health": self.model_health.get(model_id, ModelHealth.HEALTHY).value
        }
    
    def detect_drift(self, model_id: str, window_size: int = 100) -> Dict[str, Any]:
        """Detectar drift en el modelo"""
        if model_id not in self.model_performance or len(self.model_performance[model_id]) < window_size:
            return {"drift_detected": False, "reason": "insufficient_data"}
        
        recent = self.model_performance[model_id][-window_size:]
        older = self.model_performance[model_id][-window_size*2:-window_size]
        
        if len(older) < window_size:
            return {"drift_detected": False, "reason": "insufficient_data"}
        
        recent_acc = sum(p.accuracy for p in recent) / len(recent)
        older_acc = sum(p.accuracy for p in older) / len(older)
        
        drift_magnitude = abs(recent_acc - older_acc)
        drift_threshold = 0.1
        
        return {
            "drift_detected": drift_magnitude > drift_threshold,
            "drift_magnitude": drift_magnitude,
            "recent_accuracy": recent_acc,
            "older_accuracy": older_acc,
            "threshold": drift_threshold
        }
    
    def set_thresholds(self, model_id: str, thresholds: Dict[str, float]):
        """Configurar umbrales para modelo"""
        self.thresholds[model_id] = thresholds
        logger.info(f"Umbrales configurados para {model_id}")


# Instancia global
_mlops_manager: Optional[MLOpsManager] = None


def get_mlops_manager() -> MLOpsManager:
    """Obtener instancia global del gestor"""
    global _mlops_manager
    if _mlops_manager is None:
        _mlops_manager = MLOpsManager()
    return _mlops_manager
















