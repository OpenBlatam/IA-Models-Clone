"""
Sistema de monitoreo y alertas para el módulo de pipelines
"""

import time
import threading
from typing import Dict, Any, Optional, Callable, List
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """Niveles de alerta"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class Alert:
    """Representa una alerta"""
    level: AlertLevel
    message: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    details: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False


@dataclass
class HealthMetrics:
    """Métricas de salud del módulo"""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    health_score: float = 0.0
    import_status: str = "unknown"
    available_imports: int = 0
    missing_imports: int = 0
    coverage_percentage: float = 0.0
    validation_passed: bool = False
    last_check_duration: float = 0.0


class PipelineMonitor:
    """Monitor para el módulo de pipelines"""
    
    def __init__(
        self,
        check_interval: float = 60.0,
        alert_threshold: float = 0.8,
        enable_auto_check: bool = True
    ):
        """
        Inicializa el monitor.
        
        Args:
            check_interval: Intervalo en segundos entre checks automáticos
            alert_threshold: Umbral de health score para alertas
            enable_auto_check: Si activar checks automáticos
        """
        self.check_interval = check_interval
        self.alert_threshold = alert_threshold
        self.enable_auto_check = enable_auto_check
        
        self.metrics_history: List[HealthMetrics] = []
        self.alerts: List[Alert] = []
        self.alert_callbacks: List[Callable[[Alert], None]] = []
        
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
    
    def add_alert_callback(self, callback: Callable[[Alert], None]) -> None:
        """Agrega un callback para recibir alertas"""
        self.alert_callbacks.append(callback)
    
    def check_health(self) -> HealthMetrics:
        """
        Realiza un check de salud del módulo.
        
        Returns:
            HealthMetrics con los resultados del check
        """
        start_time = time.time()
        
        try:
            from .pipelines import (
                check_compatibility,
                get_import_statistics,
                validate_imports
            )
            
            compatibility = check_compatibility()
            statistics = get_import_statistics()
            imports_valid = validate_imports()
            
            duration = time.time() - start_time
            
            metrics = HealthMetrics(
                health_score=compatibility.get("health_score", 0.0),
                import_status=compatibility.get("details", {}).get("import_status", "unknown"),
                available_imports=statistics.get("available_imports", 0),
                missing_imports=statistics.get("missing_imports", 0),
                coverage_percentage=statistics.get("coverage_percentage", 0.0),
                validation_passed=imports_valid,
                last_check_duration=duration
            )
            
            # Guardar en historial
            with self._lock:
                self.metrics_history.append(metrics)
                # Mantener solo los últimos 100 registros
                if len(self.metrics_history) > 100:
                    self.metrics_history.pop(0)
            
            # Verificar si necesita alerta
            self._check_alerts(metrics, compatibility)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error checking health: {e}", exc_info=True)
            duration = time.time() - start_time
            
            metrics = HealthMetrics(
                health_score=0.0,
                import_status="error",
                last_check_duration=duration
            )
            
            # Crear alerta crítica
            alert = Alert(
                level=AlertLevel.CRITICAL,
                message=f"Error checking health: {str(e)}",
                details={"exception": str(e)}
            )
            self._trigger_alert(alert)
            
            return metrics
    
    def _check_alerts(self, metrics: HealthMetrics, compatibility: Dict) -> None:
        """Verifica si se deben generar alertas"""
        # Alerta por health score bajo
        if metrics.health_score < self.alert_threshold:
            level = AlertLevel.ERROR if metrics.health_score < 0.5 else AlertLevel.WARNING
            alert = Alert(
                level=level,
                message=f"Health score below threshold: {metrics.health_score:.2f} < {self.alert_threshold}",
                details={
                    "health_score": metrics.health_score,
                    "threshold": self.alert_threshold,
                    "compatibility": compatibility
                }
            )
            self._trigger_alert(alert)
        
        # Alerta por imports faltantes
        if metrics.missing_imports > 0:
            alert = Alert(
                level=AlertLevel.WARNING,
                message=f"Missing {metrics.missing_imports} imports",
                details={
                    "missing_imports": metrics.missing_imports,
                    "coverage": metrics.coverage_percentage
                }
            )
            self._trigger_alert(alert)
        
        # Alerta por validación fallida
        if not metrics.validation_passed:
            alert = Alert(
                level=AlertLevel.ERROR,
                message="Import validation failed",
                details={"metrics": metrics.__dict__}
            )
            self._trigger_alert(alert)
    
    def _trigger_alert(self, alert: Alert) -> None:
        """Dispara una alerta"""
        with self._lock:
            self.alerts.append(alert)
            # Mantener solo las últimas 50 alertas
            if len(self.alerts) > 50:
                self.alerts.pop(0)
        
        # Llamar callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}", exc_info=True)
    
    def get_current_health(self) -> Optional[HealthMetrics]:
        """Obtiene la métrica de salud más reciente"""
        with self._lock:
            return self.metrics_history[-1] if self.metrics_history else None
    
    def get_health_trend(self, minutes: int = 60) -> List[HealthMetrics]:
        """Obtiene el historial de salud de los últimos N minutos"""
        cutoff = datetime.utcnow() - timedelta(minutes=minutes)
        
        with self._lock:
            return [
                m for m in self.metrics_history
                if m.timestamp >= cutoff
            ]
    
    def get_active_alerts(self, level: Optional[AlertLevel] = None) -> List[Alert]:
        """Obtiene alertas activas (no resueltas)"""
        with self._lock:
            alerts = [a for a in self.alerts if not a.resolved]
            if level:
                alerts = [a for a in alerts if a.level == level]
            return alerts
    
    def resolve_alert(self, alert_index: int) -> bool:
        """Marca una alerta como resuelta"""
        with self._lock:
            if 0 <= alert_index < len(self.alerts):
                self.alerts[alert_index].resolved = True
                return True
            return False
    
    def start_monitoring(self) -> None:
        """Inicia el monitoreo automático"""
        if self._running:
            logger.warning("Monitor is already running")
            return
        
        if not self.enable_auto_check:
            logger.info("Auto-check is disabled")
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._thread.start()
        logger.info(f"Started pipeline monitoring (interval: {self.check_interval}s)")
    
    def stop_monitoring(self) -> None:
        """Detiene el monitoreo automático"""
        if not self._running:
            return
        
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
        logger.info("Stopped pipeline monitoring")
    
    def _monitoring_loop(self) -> None:
        """Loop de monitoreo automático"""
        while self._running:
            try:
                self.check_health()
                time.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}", exc_info=True)
                time.sleep(self.check_interval)
    
    def get_summary(self) -> Dict[str, Any]:
        """Obtiene un resumen del estado del monitor"""
        current = self.get_current_health()
        active_alerts = self.get_active_alerts()
        recent_metrics = self.get_health_trend(minutes=60)
        
        return {
            "monitoring_active": self._running,
            "check_interval": self.check_interval,
            "alert_threshold": self.alert_threshold,
            "current_health": current.__dict__ if current else None,
            "active_alerts_count": len(active_alerts),
            "total_alerts": len(self.alerts),
            "metrics_history_size": len(self.metrics_history),
            "recent_metrics_count": len(recent_metrics),
            "average_health_score": (
                sum(m.health_score for m in recent_metrics) / len(recent_metrics)
                if recent_metrics else 0.0
            )
        }


# Instancia global del monitor (singleton)
_global_monitor: Optional[PipelineMonitor] = None


def get_monitor(
    check_interval: float = 60.0,
    alert_threshold: float = 0.8,
    auto_start: bool = False
) -> PipelineMonitor:
    """
    Obtiene la instancia global del monitor.
    
    Args:
        check_interval: Intervalo de checks (solo en primera llamada)
        alert_threshold: Umbral de alertas (solo en primera llamada)
        auto_start: Si iniciar monitoreo automáticamente
        
    Returns:
        PipelineMonitor instance
    """
    global _global_monitor
    
    if _global_monitor is None:
        _global_monitor = PipelineMonitor(
            check_interval=check_interval,
            alert_threshold=alert_threshold
        )
        
        if auto_start:
            _global_monitor.start_monitoring()
    
    return _global_monitor


def quick_health_check() -> Dict[str, Any]:
    """
    Realiza un check rápido de salud.
    
    Returns:
        dict con resumen de salud
    """
    monitor = get_monitor()
    metrics = monitor.check_health()
    
    return {
        "health_score": metrics.health_score,
        "status": "healthy" if metrics.health_score >= 0.8 else "unhealthy",
        "coverage": metrics.coverage_percentage,
        "missing_imports": metrics.missing_imports,
        "check_duration": metrics.last_check_duration
    }

