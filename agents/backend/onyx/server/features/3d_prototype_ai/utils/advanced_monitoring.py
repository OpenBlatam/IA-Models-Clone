"""
Advanced Monitoring - Sistema de monitoring y logging avanzado
===============================================================
"""

import logging
import sys
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict, deque
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class LogLevel(str, Enum):
    """Niveles de log"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class AdvancedMonitoring:
    """Sistema de monitoring avanzado"""
    
    def __init__(self, log_dir: Optional[str] = None):
        self.log_dir = Path(log_dir) if log_dir else Path("logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.alerts: List[Dict[str, Any]] = []
        self.error_log: deque = deque(maxlen=500)
        self.performance_log: deque = deque(maxlen=500)
        
        self._setup_logging()
    
    def _setup_logging(self):
        """Configura logging avanzado"""
        # Formato detallado
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        
        # Handler para archivo
        file_handler = logging.FileHandler(
            self.log_dir / f"app_{datetime.now().strftime('%Y%m%d')}.log"
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)
        
        # Handler para consola
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.INFO)
        
        # Configurar root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
    
    def record_metric(self, metric_name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Registra una métrica"""
        timestamp = datetime.now()
        metric_data = {
            "name": metric_name,
            "value": value,
            "timestamp": timestamp.isoformat(),
            "tags": tags or {}
        }
        
        self.metrics[metric_name].append(metric_data)
        
        # Verificar alertas
        self._check_alerts(metric_name, value)
    
    def _check_alerts(self, metric_name: str, value: float):
        """Verifica si se debe disparar una alerta"""
        # Alertas predefinidas
        if metric_name == "error_rate" and value > 0.1:
            self.create_alert(
                "high_error_rate",
                f"Error rate alto: {value:.2%}",
                severity="warning"
            )
        elif metric_name == "response_time" and value > 5.0:
            self.create_alert(
                "slow_response",
                f"Tiempo de respuesta lento: {value:.2f}s",
                severity="warning"
            )
    
    def create_alert(self, alert_type: str, message: str, severity: str = "info",
                    metadata: Optional[Dict[str, Any]] = None):
        """Crea una alerta"""
        alert = {
            "id": f"alert_{len(self.alerts)}",
            "type": alert_type,
            "message": message,
            "severity": severity,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {},
            "acknowledged": False
        }
        
        self.alerts.append(alert)
        
        # Mantener solo últimas 100 alertas
        if len(self.alerts) > 100:
            self.alerts = self.alerts[-100:]
        
        logger.warning(f"ALERT [{severity.upper()}]: {message}")
    
    def log_error(self, error: Exception, context: Optional[Dict[str, Any]] = None):
        """Registra un error"""
        error_entry = {
            "timestamp": datetime.now().isoformat(),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context or {}
        }
        
        self.error_log.append(error_entry)
        
        logger.error(f"Error: {error}", exc_info=True, extra={"context": context})
    
    def log_performance(self, operation: str, duration: float, metadata: Optional[Dict[str, Any]] = None):
        """Registra métrica de rendimiento"""
        perf_entry = {
            "timestamp": datetime.now().isoformat(),
            "operation": operation,
            "duration": duration,
            "metadata": metadata or {}
        }
        
        self.performance_log.append(perf_entry)
        self.record_metric(f"performance.{operation}", duration)
    
    def get_metrics_summary(self, metric_name: Optional[str] = None,
                           time_range_minutes: int = 60) -> Dict[str, Any]:
        """Obtiene resumen de métricas"""
        cutoff = datetime.now() - timedelta(minutes=time_range_minutes)
        
        if metric_name:
            metric_data = self.metrics.get(metric_name, deque())
            recent_data = [
                m for m in metric_data
                if datetime.fromisoformat(m["timestamp"]) > cutoff
            ]
            
            if not recent_data:
                return {"metric": metric_name, "count": 0}
            
            values = [m["value"] for m in recent_data]
            return {
                "metric": metric_name,
                "count": len(recent_data),
                "avg": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
                "latest": values[-1]
            }
        
        # Resumen de todas las métricas
        summary = {}
        for name, data in self.metrics.items():
            recent_data = [
                m for m in data
                if datetime.fromisoformat(m["timestamp"]) > cutoff
            ]
            if recent_data:
                values = [m["value"] for m in recent_data]
                summary[name] = {
                    "count": len(recent_data),
                    "avg": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values)
                }
        
        return summary
    
    def get_alerts(self, severity: Optional[str] = None,
                  unacknowledged_only: bool = False) -> List[Dict[str, Any]]:
        """Obtiene alertas"""
        alerts = self.alerts
        
        if severity:
            alerts = [a for a in alerts if a["severity"] == severity]
        
        if unacknowledged_only:
            alerts = [a for a in alerts if not a["acknowledged"]]
        
        return alerts
    
    def acknowledge_alert(self, alert_id: str):
        """Reconoce una alerta"""
        for alert in self.alerts:
            if alert["id"] == alert_id:
                alert["acknowledged"] = True
                alert["acknowledged_at"] = datetime.now().isoformat()
                return True
        return False
    
    def get_error_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Obtiene resumen de errores"""
        cutoff = datetime.now() - timedelta(hours=hours)
        recent_errors = [
            e for e in self.error_log
            if datetime.fromisoformat(e["timestamp"]) > cutoff
        ]
        
        error_types = defaultdict(int)
        for error in recent_errors:
            error_types[error["error_type"]] += 1
        
        return {
            "total_errors": len(recent_errors),
            "error_types": dict(error_types),
            "errors": recent_errors[-10:]  # Últimos 10 errores
        }
    
    def get_performance_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Obtiene resumen de rendimiento"""
        cutoff = datetime.now() - timedelta(hours=hours)
        recent_perf = [
            p for p in self.performance_log
            if datetime.fromisoformat(p["timestamp"]) > cutoff
        ]
        
        operations = defaultdict(list)
        for perf in recent_perf:
            operations[perf["operation"]].append(perf["duration"])
        
        summary = {}
        for op, durations in operations.items():
            summary[op] = {
                "count": len(durations),
                "avg": sum(durations) / len(durations),
                "min": min(durations),
                "max": max(durations),
                "p95": sorted(durations)[int(len(durations) * 0.95)] if durations else 0
            }
        
        return summary
    
    def export_logs(self, log_type: str = "all", hours: int = 24) -> str:
        """Exporta logs a archivo"""
        cutoff = datetime.now() - timedelta(hours=hours)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        export_data = {}
        
        if log_type in ["all", "errors"]:
            export_data["errors"] = [
                e for e in self.error_log
                if datetime.fromisoformat(e["timestamp"]) > cutoff
            ]
        
        if log_type in ["all", "performance"]:
            export_data["performance"] = [
                p for p in self.performance_log
                if datetime.fromisoformat(p["timestamp"]) > cutoff
            ]
        
        if log_type in ["all", "alerts"]:
            export_data["alerts"] = self.alerts
        
        export_file = self.log_dir / f"export_{log_type}_{timestamp}.json"
        with open(export_file, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)
        
        return str(export_file)




