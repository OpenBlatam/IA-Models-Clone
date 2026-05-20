"""
Intelligent Alerts - Alertas Inteligentes
=========================================

Sistema de alertas inteligentes basado en ML para detectar problemas proactivamente.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import deque, defaultdict
import statistics

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Severidad de alerta."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class IntelligentAlert:
    """Alerta inteligente."""
    alert_id: str
    metric_name: str
    severity: AlertSeverity
    message: str
    value: float
    threshold: float
    anomaly_score: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


class IntelligentAlertSystem:
    """Sistema de alertas inteligente."""
    
    def __init__(
        self,
        window_size: int = 100,
        anomaly_threshold: float = 2.0,
    ):
        self.metric_history: Dict[str, deque] = {}
        self.alerts: List[IntelligentAlert] = []
        self.alert_handlers: Dict[AlertSeverity, List[Callable]] = defaultdict(list)
        self.window_size = window_size
        self.anomaly_threshold = anomaly_threshold
        self._lock = asyncio.Lock()
    
    def register_alert_handler(
        self,
        severity: AlertSeverity,
        handler: Callable,
    ):
        """Registrar handler para alertas de severidad específica."""
        self.alert_handlers[severity].append(handler)
        logger.info(f"Registered alert handler for {severity.value}")
    
    async def record_metric(
        self,
        metric_name: str,
        value: float,
        timestamp: Optional[datetime] = None,
    ):
        """
        Registrar métrica.
        
        Args:
            metric_name: Nombre de la métrica
            value: Valor de la métrica
            timestamp: Timestamp (opcional)
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        async with self._lock:
            if metric_name not in self.metric_history:
                self.metric_history[metric_name] = deque(maxlen=self.window_size)
            
            self.metric_history[metric_name].append({
                "value": value,
                "timestamp": timestamp,
            })
        
        # Verificar si hay anomalía
        await self._check_anomaly(metric_name, value, timestamp)
    
    async def _check_anomaly(
        self,
        metric_name: str,
        value: float,
        timestamp: datetime,
    ):
        """Verificar si el valor es una anomalía."""
        history = self.metric_history.get(metric_name, deque())
        
        if len(history) < 10:
            return  # No hay suficientes datos
        
        values = [h["value"] for h in history]
        mean = statistics.mean(values)
        std_dev = statistics.stdev(values) if len(values) > 1 else 0.0
        
        if std_dev == 0:
            return
        
        # Calcular z-score
        z_score = abs((value - mean) / std_dev) if std_dev > 0 else 0.0
        
        if z_score >= self.anomaly_threshold:
            # Determinar severidad
            if z_score >= 4.0:
                severity = AlertSeverity.CRITICAL
            elif z_score >= 3.0:
                severity = AlertSeverity.HIGH
            elif z_score >= 2.5:
                severity = AlertSeverity.MEDIUM
            else:
                severity = AlertSeverity.LOW
            
            alert = IntelligentAlert(
                alert_id=f"alert_{metric_name}_{timestamp.timestamp()}",
                metric_name=metric_name,
                severity=severity,
                message=f"Anomaly detected in {metric_name}: {value:.2f} (mean: {mean:.2f}, z-score: {z_score:.2f})",
                value=value,
                threshold=mean + (self.anomaly_threshold * std_dev),
                anomaly_score=z_score,
                timestamp=timestamp,
                metadata={
                    "mean": mean,
                    "std_dev": std_dev,
                    "z_score": z_score,
                },
            )
            
            async with self._lock:
                self.alerts.append(alert)
                if len(self.alerts) > 1000:
                    self.alerts.pop(0)
            
            # Ejecutar handlers
            handlers = self.alert_handlers.get(severity, [])
            for handler in handlers:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(alert)
                    else:
                        handler(alert)
                except Exception as e:
                    logger.error(f"Error executing alert handler: {e}")
            
            logger.warning(f"Alert triggered: {alert.message}")
    
    def get_recent_alerts(
        self,
        severity: Optional[AlertSeverity] = None,
        metric_name: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Obtener alertas recientes."""
        alerts = self.alerts
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        
        if metric_name:
            alerts = [a for a in alerts if a.metric_name == metric_name]
        
        return [
            {
                "alert_id": a.alert_id,
                "metric_name": a.metric_name,
                "severity": a.severity.value,
                "message": a.message,
                "value": a.value,
                "threshold": a.threshold,
                "anomaly_score": a.anomaly_score,
                "timestamp": a.timestamp.isoformat(),
            }
            for a in alerts[-limit:]
        ]
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Obtener resumen de alertas."""
        if not self.alerts:
            return {
                "total_alerts": 0,
                "by_severity": {},
                "by_metric": {},
            }
        
        by_severity: Dict[str, int] = defaultdict(int)
        by_metric: Dict[str, int] = defaultdict(int)
        
        for alert in self.alerts:
            by_severity[alert.severity.value] += 1
            by_metric[alert.metric_name] += 1
        
        return {
            "total_alerts": len(self.alerts),
            "by_severity": dict(by_severity),
            "by_metric": dict(by_metric),
            "recent_alerts_count": len([a for a in self.alerts if (datetime.now() - a.timestamp).total_seconds() < 3600]),
        }

