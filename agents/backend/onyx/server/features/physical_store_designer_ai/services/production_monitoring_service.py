"""
Production Monitoring Service - Monitoreo de modelos en producción
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


class AlertLevel(str, Enum):
    """Niveles de alerta"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ProductionMonitoringService:
    """Servicio para monitoreo de modelos en producción"""
    
    def __init__(self):
        self.monitored_models: Dict[str, Dict[str, Any]] = {}
        self.metrics: Dict[str, List[Dict[str, Any]]] = {}
        self.alerts: Dict[str, List[Dict[str, Any]]] = {}
    
    def register_model_for_monitoring(
        self,
        model_id: str,
        deployment_id: str,
        thresholds: Dict[str, float]
    ) -> Dict[str, Any]:
        """Registrar modelo para monitoreo"""
        
        monitoring_id = f"monitor_{model_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        monitoring = {
            "monitoring_id": monitoring_id,
            "model_id": model_id,
            "deployment_id": deployment_id,
            "thresholds": thresholds,
            "status": "active",
            "registered_at": datetime.now().isoformat()
        }
        
        self.monitored_models[monitoring_id] = monitoring
        
        return monitoring
    
    def record_prediction(
        self,
        model_id: str,
        prediction: Any,
        latency_ms: float,
        timestamp: Optional[str] = None
    ) -> Dict[str, Any]:
        """Registrar predicción"""
        
        record_id = f"pred_{model_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        record = {
            "record_id": record_id,
            "model_id": model_id,
            "prediction": prediction,
            "latency_ms": latency_ms,
            "timestamp": timestamp or datetime.now().isoformat()
        }
        
        if model_id not in self.metrics:
            self.metrics[model_id] = []
        
        self.metrics[model_id].append(record)
        
        return record
    
    def check_model_health(
        self,
        model_id: str,
        time_window_minutes: int = 60
    ) -> Dict[str, Any]:
        """Verificar salud del modelo"""
        
        metrics = self.metrics.get(model_id, [])
        
        if not metrics:
            return {
                "model_id": model_id,
                "status": "unknown",
                "message": "No hay métricas disponibles"
            }
        
        # Filtrar métricas del último time_window
        cutoff = datetime.now() - timedelta(minutes=time_window_minutes)
        recent_metrics = [
            m for m in metrics
            if datetime.fromisoformat(m["timestamp"]) > cutoff
        ]
        
        if not recent_metrics:
            return {
                "model_id": model_id,
                "status": "unknown",
                "message": "No hay métricas recientes"
            }
        
        # Calcular estadísticas
        latencies = [m["latency_ms"] for m in recent_metrics]
        avg_latency = sum(latencies) / len(latencies)
        p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]
        
        health = {
            "model_id": model_id,
            "status": "healthy",
            "time_window_minutes": time_window_minutes,
            "metrics": {
                "total_predictions": len(recent_metrics),
                "avg_latency_ms": avg_latency,
                "p95_latency_ms": p95_latency,
                "min_latency_ms": min(latencies),
                "max_latency_ms": max(latencies)
            },
            "checked_at": datetime.now().isoformat()
        }
        
        # Verificar umbrales
        monitoring = next(
            (m for m in self.monitored_models.values() if m["model_id"] == model_id),
            None
        )
        
        if monitoring:
            thresholds = monitoring.get("thresholds", {})
            if thresholds.get("max_latency") and avg_latency > thresholds["max_latency"]:
                health["status"] = "degraded"
                health["alerts"] = ["Latency exceeds threshold"]
        
        return health
    
    def detect_drift(
        self,
        model_id: str,
        reference_data: List[Any],
        current_data: List[Any]
    ) -> Dict[str, Any]:
        """Detectar drift de datos"""
        
        drift_id = f"drift_{model_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        drift = {
            "drift_id": drift_id,
            "model_id": model_id,
            "detected_at": datetime.now().isoformat(),
            "note": "En producción, esto calcularía drift real usando KS test, PSI, etc."
        }
        
        # Simular detección
        drift["has_drift"] = False
        drift["drift_score"] = 0.15
        drift["threshold"] = 0.3
        
        if drift["drift_score"] > drift["threshold"]:
            drift["has_drift"] = True
            drift["severity"] = "moderate"
        
        return drift
    
    def generate_alert(
        self,
        model_id: str,
        alert_type: str,
        message: str,
        level: str = AlertLevel.WARNING.value
    ) -> Dict[str, Any]:
        """Generar alerta"""
        
        alert_id = f"alert_{model_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        alert = {
            "alert_id": alert_id,
            "model_id": model_id,
            "type": alert_type,
            "message": message,
            "level": level,
            "created_at": datetime.now().isoformat(),
            "acknowledged": False
        }
        
        if model_id not in self.alerts:
            self.alerts[model_id] = []
        
        self.alerts[model_id].append(alert)
        
        return alert
    
    def get_dashboard(
        self,
        model_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Obtener dashboard de monitoreo"""
        
        dashboard = {
            "generated_at": datetime.now().isoformat(),
            "models": []
        }
        
        if model_id:
            models_to_show = [m for m in self.monitored_models.values() if m["model_id"] == model_id]
        else:
            models_to_show = list(self.monitored_models.values())
        
        for monitoring in models_to_show:
            health = self.check_model_health(monitoring["model_id"])
            dashboard["models"].append({
                "model_id": monitoring["model_id"],
                "health": health,
                "alerts": self.alerts.get(monitoring["model_id"], [])
            })
        
        return dashboard




