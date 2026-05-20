"""
Sistema de monitoreo avanzado
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import deque, defaultdict
import time
import threading


@dataclass
class SystemMetric:
    """Métrica del sistema"""
    name: str
    value: float
    unit: str
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "name": self.name,
            "value": self.value,
            "unit": self.unit,
            "timestamp": self.timestamp
        }


class AdvancedMonitoring:
    """Sistema de monitoreo avanzado"""
    
    def __init__(self):
        """Inicializa el sistema"""
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.alerts: List[Dict] = []
        self.thresholds: Dict[str, Dict] = {}
        self.lock = threading.Lock()
    
    def record_metric(self, name: str, value: float, unit: str = ""):
        """
        Registra una métrica
        
        Args:
            name: Nombre de la métrica
            value: Valor
            unit: Unidad
        """
        metric = SystemMetric(name=name, value=value, unit=unit)
        
        with self.lock:
            self.metrics[name].append(metric)
            
            # Verificar umbrales
            self._check_thresholds(name, value)
    
    def set_threshold(self, metric_name: str, min_value: Optional[float] = None,
                     max_value: Optional[float] = None, alert_message: str = ""):
        """
        Establece umbral para una métrica
        
        Args:
            metric_name: Nombre de la métrica
            min_value: Valor mínimo
            max_value: Valor máximo
            alert_message: Mensaje de alerta
        """
        self.thresholds[metric_name] = {
            "min": min_value,
            "max": max_value,
            "alert_message": alert_message
        }
    
    def _check_thresholds(self, metric_name: str, value: float):
        """Verifica umbrales"""
        if metric_name not in self.thresholds:
            return
        
        threshold = self.thresholds[metric_name]
        
        alert_triggered = False
        if threshold["min"] is not None and value < threshold["min"]:
            alert_triggered = True
        if threshold["max"] is not None and value > threshold["max"]:
            alert_triggered = True
        
        if alert_triggered:
            alert = {
                "metric": metric_name,
                "value": value,
                "threshold": threshold,
                "message": threshold.get("alert_message", f"Umbral excedido para {metric_name}"),
                "timestamp": datetime.now().isoformat()
            }
            self.alerts.append(alert)
            
            # Mantener solo últimos 1000 alertas
            if len(self.alerts) > 1000:
                self.alerts = self.alerts[-1000:]
    
    def get_metric_history(self, metric_name: str, hours: int = 24) -> List[SystemMetric]:
        """Obtiene historial de una métrica"""
        cutoff = datetime.now() - timedelta(hours=hours)
        
        with self.lock:
            metrics = list(self.metrics.get(metric_name, deque()))
            return [
                m for m in metrics
                if datetime.fromisoformat(m.timestamp) >= cutoff
            ]
    
    def get_metric_statistics(self, metric_name: str, hours: int = 24) -> Dict:
        """Obtiene estadísticas de una métrica"""
        history = self.get_metric_history(metric_name, hours)
        
        if not history:
            return {
                "count": 0,
                "min": None,
                "max": None,
                "avg": None
            }
        
        values = [m.value for m in history]
        
        return {
            "count": len(values),
            "min": float(min(values)),
            "max": float(max(values)),
            "avg": float(sum(values) / len(values)),
            "latest": float(values[-1]) if values else None
        }
    
    def get_system_health(self) -> Dict:
        """Obtiene salud del sistema"""
        health_metrics = {}
        
        # CPU, Memory, etc. (placeholders)
        health_metrics["cpu_usage"] = self._get_cpu_usage()
        health_metrics["memory_usage"] = self._get_memory_usage()
        health_metrics["disk_usage"] = self._get_disk_usage()
        
        # Alertas recientes
        recent_alerts = [a for a in self.alerts[-10:]]
        
        return {
            "timestamp": datetime.now().isoformat(),
            "metrics": health_metrics,
            "recent_alerts": recent_alerts,
            "total_alerts": len(self.alerts),
            "status": "healthy" if len(recent_alerts) == 0 else "warning"
        }
    
    def _get_cpu_usage(self) -> float:
        """Obtiene uso de CPU (placeholder)"""
        try:
            import psutil
            return float(psutil.cpu_percent())
        except ImportError:
            return 0.0
    
    def _get_memory_usage(self) -> float:
        """Obtiene uso de memoria (placeholder)"""
        try:
            import psutil
            return float(psutil.virtual_memory().percent)
        except ImportError:
            return 0.0
    
    def _get_disk_usage(self) -> float:
        """Obtiene uso de disco (placeholder)"""
        try:
            import psutil
            return float(psutil.disk_usage('/').percent)
        except ImportError:
            return 0.0
    
    def get_alerts(self, limit: int = 100) -> List[Dict]:
        """Obtiene alertas"""
        with self.lock:
            return self.alerts[-limit:]






