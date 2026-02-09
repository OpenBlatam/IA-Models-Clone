"""
Monitoring - Sistema de Monitoring Avanzado
===========================================

Sistema de monitoreo avanzado con métricas y alertas.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import time

logger = logging.getLogger(__name__)


@dataclass
class Metric:
    """Métrica."""
    name: str
    value: float
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Alert:
    """Alerta de monitoreo."""
    alert_id: str
    name: str
    severity: str  # "info", "warning", "critical"
    message: str
    metric_name: str
    threshold: float
    current_value: float
    timestamp: datetime
    resolved: bool = False
    resolved_at: Optional[datetime] = None


class AdvancedMonitoring:
    """Sistema de monitoreo avanzado."""
    
    def __init__(self, retention_hours: int = 24):
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.alerts: List[Alert] = []
        self.retention_hours = retention_hours
        self._lock = asyncio.Lock()
    
    async def record_metric(
        self,
        name: str,
        value: float,
        tags: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Registrar métrica."""
        metric = Metric(
            name=name,
            value=value,
            timestamp=datetime.now(),
            tags=tags or {},
            metadata=metadata or {},
        )
        
        async with self._lock:
            self.metrics[name].append(metric)
        
        logger.debug(f"Recorded metric: {name} = {value}")
    
    async def get_metric_stats(
        self,
        name: str,
        window_minutes: int = 60,
    ) -> Dict[str, Any]:
        """Obtener estadísticas de métrica."""
        now = datetime.now()
        cutoff = now - timedelta(minutes=window_minutes)
        
        metrics = [
            m for m in self.metrics.get(name, [])
            if m.timestamp >= cutoff
        ]
        
        if not metrics:
            return {
                "name": name,
                "count": 0,
                "average": 0.0,
                "min": 0.0,
                "max": 0.0,
            }
        
        values = [m.value for m in metrics]
        
        return {
            "name": name,
            "count": len(metrics),
            "average": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
            "first_timestamp": metrics[0].timestamp.isoformat(),
            "last_timestamp": metrics[-1].timestamp.isoformat(),
        }
    
    async def create_alert(
        self,
        name: str,
        metric_name: str,
        threshold: float,
        severity: str = "warning",
        condition: str = "greater_than",  # "greater_than", "less_than", "equals"
    ):
        """Crear alerta."""
        alert_id = f"alert_{datetime.now().timestamp()}_{name}"
        
        # Verificar condición
        current_value = await self._get_current_metric_value(metric_name)
        
        triggered = False
        if condition == "greater_than" and current_value > threshold:
            triggered = True
        elif condition == "less_than" and current_value < threshold:
            triggered = True
        elif condition == "equals" and abs(current_value - threshold) < 0.001:
            triggered = True
        
        if triggered:
            alert = Alert(
                alert_id=alert_id,
                name=name,
                severity=severity,
                message=f"{metric_name} is {condition.replace('_', ' ')} {threshold} (current: {current_value})",
                metric_name=metric_name,
                threshold=threshold,
                current_value=current_value,
                timestamp=datetime.now(),
            )
            
            async with self._lock:
                self.alerts.append(alert)
            
            logger.warning(f"Alert triggered: {name} - {alert.message}")
            return alert
        
        return None
    
    async def _get_current_metric_value(self, metric_name: str) -> float:
        """Obtener valor actual de métrica."""
        metrics = self.metrics.get(metric_name, deque())
        if metrics:
            return metrics[-1].value
        return 0.0
    
    async def get_alerts(
        self,
        severity: Optional[str] = None,
        resolved: Optional[bool] = None,
        limit: int = 100,
    ) -> List[Alert]:
        """Obtener alertas."""
        alerts = self.alerts
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        
        if resolved is not None:
            alerts = [a for a in alerts if a.resolved == resolved]
        
        alerts.sort(key=lambda a: a.timestamp, reverse=True)
        
        return alerts[:limit]
    
    async def resolve_alert(self, alert_id: str):
        """Resolver alerta."""
        async with self._lock:
            for alert in self.alerts:
                if alert.alert_id == alert_id:
                    alert.resolved = True
                    alert.resolved_at = datetime.now()
                    logger.info(f"Resolved alert: {alert_id}")
                    return
        
        logger.warning(f"Alert not found: {alert_id}")
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Obtener resumen de métricas."""
        summary = {
            "total_metrics": len(self.metrics),
            "total_data_points": sum(len(metrics) for metrics in self.metrics.values()),
            "active_alerts": sum(1 for a in self.alerts if not a.resolved),
            "total_alerts": len(self.alerts),
            "metrics": {},
        }
        
        for name, metrics in self.metrics.items():
            if metrics:
                values = [m.value for m in metrics]
                summary["metrics"][name] = {
                    "count": len(metrics),
                    "average": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "latest": values[-1] if values else 0.0,
                }
        
        return summary
































