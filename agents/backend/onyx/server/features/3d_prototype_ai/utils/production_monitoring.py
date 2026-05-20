"""
Production Monitoring - Monitoreo de modelos en producción
===========================================================
Monitoreo de drift, performance, y calidad de predicciones
"""

import logging
import torch
import numpy as np
from typing import Dict, List, Any, Optional
from collections import deque
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class ProductionMonitor:
    """Monitor de modelos en producción"""
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.prediction_history: deque = deque(maxlen=window_size)
        self.latency_history: deque = deque(maxlen=window_size)
        self.error_history: deque = deque(maxlen=window_size)
        self.input_distributions: Dict[str, deque] = {}
        self.output_distributions: Dict[str, deque] = {}
    
    def log_prediction(
        self,
        input_data: Any,
        prediction: Any,
        latency: float,
        error: Optional[str] = None
    ):
        """Registra predicción"""
        self.prediction_history.append({
            "timestamp": datetime.now().isoformat(),
            "input": str(input_data)[:100],  # Truncar
            "prediction": str(prediction)[:100],
            "error": error
        })
        
        self.latency_history.append(latency)
        
        if error:
            self.error_history.append({
                "timestamp": datetime.now().isoformat(),
                "error": error
            })
    
    def detect_data_drift(
        self,
        current_input: np.ndarray,
        reference_input: np.ndarray,
        threshold: float = 0.1
    ) -> Dict[str, Any]:
        """Detecta drift en datos de entrada"""
        # Calcular estadísticas
        current_mean = np.mean(current_input)
        current_std = np.std(current_input)
        reference_mean = np.mean(reference_input)
        reference_std = np.std(reference_input)
        
        # Calcular distancia
        mean_diff = abs(current_mean - reference_mean) / (reference_std + 1e-8)
        std_diff = abs(current_std - reference_std) / (reference_std + 1e-8)
        
        drift_detected = mean_diff > threshold or std_diff > threshold
        
        return {
            "drift_detected": drift_detected,
            "mean_difference": mean_diff,
            "std_difference": std_diff,
            "threshold": threshold
        }
    
    def detect_prediction_drift(
        self,
        current_predictions: np.ndarray,
        reference_predictions: np.ndarray,
        threshold: float = 0.1
    ) -> Dict[str, Any]:
        """Detecta drift en predicciones"""
        # Calcular distribución de predicciones
        current_dist = np.histogram(current_predictions, bins=10)[0]
        reference_dist = np.histogram(reference_predictions, bins=10)[0]
        
        # Normalizar
        current_dist = current_dist / (current_dist.sum() + 1e-8)
        reference_dist = reference_dist / (reference_dist.sum() + 1e-8)
        
        # Calcular distancia (KL divergence simplificada)
        kl_div = np.sum(reference_dist * np.log(reference_dist / (current_dist + 1e-8) + 1e-8))
        
        drift_detected = kl_div > threshold
        
        return {
            "drift_detected": drift_detected,
            "kl_divergence": kl_div,
            "threshold": threshold
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Obtiene métricas de performance"""
        if not self.latency_history:
            return {}
        
        latencies = list(self.latency_history)
        
        return {
            "avg_latency": np.mean(latencies),
            "p50_latency": np.percentile(latencies, 50),
            "p95_latency": np.percentile(latencies, 95),
            "p99_latency": np.percentile(latencies, 99),
            "max_latency": np.max(latencies),
            "min_latency": np.min(latencies),
            "total_predictions": len(self.prediction_history),
            "error_rate": len(self.error_history) / len(self.prediction_history) if self.prediction_history else 0
        }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Obtiene estado de salud del modelo"""
        metrics = self.get_performance_metrics()
        
        # Determinar salud
        health = "healthy"
        if metrics.get("error_rate", 0) > 0.05:
            health = "degraded"
        if metrics.get("error_rate", 0) > 0.1:
            health = "unhealthy"
        if metrics.get("p95_latency", 0) > 1.0:  # 1 segundo
            health = "degraded"
        
        return {
            "status": health,
            "metrics": metrics,
            "timestamp": datetime.now().isoformat()
        }
    
    def generate_report(self) -> Dict[str, Any]:
        """Genera reporte completo"""
        return {
            "performance": self.get_performance_metrics(),
            "health": self.get_health_status(),
            "recent_errors": list(self.error_history)[-10:],
            "timestamp": datetime.now().isoformat()
        }




