"""
Real-time Metrics - Métricas en Tiempo Real
===========================================

Sistema de métricas en tiempo real con agregación, alertas y visualización.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict, deque
import statistics

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Tipo de métrica."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class Metric:
    """Métrica."""
    metric_id: str
    metric_name: str
    metric_type: MetricType
    value: float
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricAlert:
    """Alerta de métrica."""
    alert_id: str
    metric_name: str
    condition: str  # "gt", "lt", "eq", "gte", "lte"
    threshold: float
    triggered: bool = False
    triggered_at: Optional[datetime] = None
    handler: Optional[Callable] = None


class RealTimeMetrics:
    """Sistema de métricas en tiempo real."""
    
    def __init__(self, history_size: int = 100000):
        self.history_size = history_size
        self.metrics: deque = deque(maxlen=history_size)
        self.metric_aggregates: Dict[str, Dict[str, Any]] = {}
        self.alerts: Dict[str, MetricAlert] = {}
        self._lock = asyncio.Lock()
    
    def record_metric(
        self,
        metric_name: str,
        value: float,
        metric_type: MetricType = MetricType.GAUGE,
        labels: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Registrar métrica."""
        metric_id = f"metric_{metric_name}_{datetime.now().timestamp()}"
        
        metric = Metric(
            metric_id=metric_id,
            metric_name=metric_name,
            metric_type=metric_type,
            value=value,
            timestamp=datetime.now(),
            labels=labels or {},
            metadata=metadata or {},
        )
        
        self.metrics.append(metric)
        
        # Actualizar agregados
        asyncio.create_task(self._update_aggregates(metric))
        
        # Verificar alertas
        asyncio.create_task(self._check_alerts(metric))
        
        return metric_id
    
    async def _update_aggregates(self, metric: Metric):
        """Actualizar agregados de métricas."""
        async with self._lock:
            key = f"{metric.metric_name}_{str(metric.labels)}"
            
            if key not in self.metric_aggregates:
                self.metric_aggregates[key] = {
                    "count": 0,
                    "sum": 0.0,
                    "min": float('inf'),
                    "max": float('-inf'),
                    "values": deque(maxlen=1000),
                }
            
            agg = self.metric_aggregates[key]
            agg["count"] += 1
            agg["sum"] += metric.value
            agg["min"] = min(agg["min"], metric.value)
            agg["max"] = max(agg["max"], metric.value)
            agg["values"].append(metric.value)
    
    async def _check_alerts(self, metric: Metric):
        """Verificar alertas."""
        for alert in self.alerts.values():
            if alert.metric_name != metric.metric_name:
                continue
            
            triggered = False
            
            if alert.condition == "gt" and metric.value > alert.threshold:
                triggered = True
            elif alert.condition == "lt" and metric.value < alert.threshold:
                triggered = True
            elif alert.condition == "eq" and metric.value == alert.threshold:
                triggered = True
            elif alert.condition == "gte" and metric.value >= alert.threshold:
                triggered = True
            elif alert.condition == "lte" and metric.value <= alert.threshold:
                triggered = True
            
            if triggered and not alert.triggered:
                alert.triggered = True
                alert.triggered_at = datetime.now()
                
                if alert.handler:
                    try:
                        if asyncio.iscoroutinefunction(alert.handler):
                            await alert.handler(metric, alert)
                        else:
                            alert.handler(metric, alert)
                    except Exception as e:
                        logger.error(f"Error in alert handler: {e}")
    
    def create_alert(
        self,
        alert_id: str,
        metric_name: str,
        condition: str,
        threshold: float,
        handler: Optional[Callable] = None,
    ) -> str:
        """Crear alerta."""
        alert = MetricAlert(
            alert_id=alert_id,
            metric_name=metric_name,
            condition=condition,
            threshold=threshold,
            handler=handler,
        )
        
        async def save_alert():
            async with self._lock:
                self.alerts[alert_id] = alert
        
        asyncio.create_task(save_alert())
        
        logger.info(f"Created alert: {alert_id} for {metric_name}")
        return alert_id
    
    def get_metric_aggregates(
        self,
        metric_name: str,
        labels: Optional[Dict[str, str]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Obtener agregados de métrica."""
        key = f"{metric_name}_{str(labels or {})}"
        agg = self.metric_aggregates.get(key)
        
        if not agg:
            return None
        
        values = list(agg["values"])
        
        return {
            "metric_name": metric_name,
            "count": agg["count"],
            "sum": agg["sum"],
            "min": agg["min"] if agg["min"] != float('inf') else 0.0,
            "max": agg["max"] if agg["max"] != float('-inf') else 0.0,
            "avg": agg["sum"] / agg["count"] if agg["count"] > 0 else 0.0,
            "median": statistics.median(values) if values else 0.0,
            "p95": sorted(values)[int(len(values) * 0.95)] if len(values) > 20 else (max(values) if values else 0.0),
            "p99": sorted(values)[int(len(values) * 0.99)] if len(values) > 100 else (max(values) if values else 0.0),
        }
    
    def get_metrics(
        self,
        metric_name: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000,
    ) -> List[Dict[str, Any]]:
        """Obtener métricas."""
        metrics = list(self.metrics)
        
        if metric_name:
            metrics = [m for m in metrics if m.metric_name == metric_name]
        
        if start_time:
            metrics = [m for m in metrics if m.timestamp >= start_time]
        
        if end_time:
            metrics = [m for m in metrics if m.timestamp <= end_time]
        
        metrics.sort(key=lambda m: m.timestamp, reverse=True)
        
        return [
            {
                "metric_id": m.metric_id,
                "metric_name": m.metric_name,
                "metric_type": m.metric_type.value,
                "value": m.value,
                "timestamp": m.timestamp.isoformat(),
                "labels": m.labels,
            }
            for m in metrics[:limit]
        ]
    
    def get_realtime_metrics_summary(self) -> Dict[str, Any]:
        """Obtener resumen de métricas."""
        by_name: Dict[str, int] = defaultdict(int)
        by_type: Dict[str, int] = defaultdict(int)
        
        for metric in self.metrics:
            by_name[metric.metric_name] += 1
            by_type[metric.metric_type.value] += 1
        
        return {
            "total_metrics": len(self.metrics),
            "metrics_by_name": dict(by_name),
            "metrics_by_type": dict(by_type),
            "total_aggregates": len(self.metric_aggregates),
            "total_alerts": len(self.alerts),
            "active_alerts": len([a for a in self.alerts.values() if a.triggered]),
        }














