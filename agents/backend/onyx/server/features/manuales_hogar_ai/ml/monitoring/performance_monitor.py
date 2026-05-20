"""
Performance Monitor
==================

Monitoreo de rendimiento en tiempo real.
"""

import logging
import time
from typing import Dict, Any, List, Optional
from collections import deque
from datetime import datetime
import psutil
import torch

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """Monitor de rendimiento."""
    
    def __init__(self, history_size: int = 100):
        """
        Inicializar monitor.
        
        Args:
            history_size: Tamaño del historial
        """
        self.history_size = history_size
        self.metrics_history: Dict[str, deque] = {
            "latency": deque(maxlen=history_size),
            "throughput": deque(maxlen=history_size),
            "gpu_utilization": deque(maxlen=history_size),
            "memory_usage": deque(maxlen=history_size),
            "error_rate": deque(maxlen=history_size)
        }
        self._logger = logger
    
    def record_latency(self, latency: float):
        """Registrar latencia."""
        self.metrics_history["latency"].append({
            "value": latency,
            "timestamp": datetime.now().isoformat()
        })
    
    def record_throughput(self, throughput: float):
        """Registrar throughput."""
        self.metrics_history["throughput"].append({
            "value": throughput,
            "timestamp": datetime.now().isoformat()
        })
    
    def record_gpu_utilization(self):
        """Registrar utilización de GPU."""
        if torch.cuda.is_available():
            utilization = torch.cuda.utilization()
            self.metrics_history["gpu_utilization"].append({
                "value": utilization,
                "timestamp": datetime.now().isoformat()
            })
    
    def record_memory_usage(self):
        """Registrar uso de memoria."""
        memory = psutil.virtual_memory()
        self.metrics_history["memory_usage"].append({
            "value": memory.percent,
            "timestamp": datetime.now().isoformat()
        })
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Obtener métricas actuales."""
        metrics = {}
        
        for key, history in self.metrics_history.items():
            if history:
                values = [item["value"] for item in history]
                metrics[key] = {
                    "current": values[-1] if values else None,
                    "average": sum(values) / len(values) if values else 0,
                    "min": min(values) if values else None,
                    "max": max(values) if values else None
                }
            else:
                metrics[key] = {
                    "current": None,
                    "average": 0,
                    "min": None,
                    "max": None
                }
        
        # GPU info
        if torch.cuda.is_available():
            metrics["gpu"] = {
                "available": True,
                "device_count": torch.cuda.device_count(),
                "current_device": torch.cuda.current_device(),
                "memory_allocated": torch.cuda.memory_allocated() / 1e9,  # GB
                "memory_reserved": torch.cuda.memory_reserved() / 1e9,  # GB
                "utilization": torch.cuda.utilization() if hasattr(torch.cuda, 'utilization') else None
            }
        else:
            metrics["gpu"] = {"available": False}
        
        # CPU info
        metrics["cpu"] = {
            "usage": psutil.cpu_percent(),
            "count": psutil.cpu_count()
        }
        
        return metrics
    
    def check_alerts(self, thresholds: Dict[str, float]) -> List[str]:
        """
        Verificar alertas según umbrales.
        
        Args:
            thresholds: Umbrales de alerta
        
        Returns:
            Lista de alertas
        """
        alerts = []
        metrics = self.get_current_metrics()
        
        for metric_name, threshold in thresholds.items():
            if metric_name in metrics:
                current = metrics[metric_name].get("current")
                if current is not None:
                    if metric_name == "latency" and current > threshold:
                        alerts.append(f"Alta latencia: {current:.2f}s (umbral: {threshold}s)")
                    elif metric_name == "error_rate" and current > threshold:
                        alerts.append(f"Alta tasa de errores: {current:.2%} (umbral: {threshold:.2%})")
                    elif metric_name == "memory_usage" and current > threshold:
                        alerts.append(f"Alto uso de memoria: {current:.1f}% (umbral: {threshold}%)")
        
        return alerts




