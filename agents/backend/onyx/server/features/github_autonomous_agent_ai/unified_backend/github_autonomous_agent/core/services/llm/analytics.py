"""
Analytics Avanzado para LLM Service.

Sistema completo de analytics con:
- Tracking de métricas en tiempo real
- Análisis de tendencias
- Alertas automáticas
- Reportes personalizados
"""

import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from collections import defaultdict
from enum import Enum
import os

from config.logging_config import get_logger

logger = get_logger(__name__)


class MetricType(str, Enum):
    """Tipos de métricas."""
    REQUEST_COUNT = "request_count"
    LATENCY = "latency"
    TOKEN_USAGE = "token_usage"
    COST = "cost"
    ERROR_RATE = "error_rate"
    CACHE_HIT_RATE = "cache_hit_rate"
    SUCCESS_RATE = "success_rate"


@dataclass
class MetricPoint:
    """Punto de métrica."""
    timestamp: datetime
    value: float
    tags: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "value": self.value,
            "tags": self.tags
        }


@dataclass
class AlertRule:
    """Regla de alerta."""
    alert_id: str
    name: str
    metric_type: MetricType
    threshold: float
    comparison: str  # "gt", "lt", "eq"
    window_minutes: int = 5
    enabled: bool = True
    notification_channels: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario."""
        return {
            "alert_id": self.alert_id,
            "name": self.name,
            "metric_type": self.metric_type.value,
            "threshold": self.threshold,
            "comparison": self.comparison,
            "window_minutes": self.window_minutes,
            "enabled": self.enabled,
            "notification_channels": self.notification_channels
        }


@dataclass
class Alert:
    """Alerta activa."""
    alert_id: str
    rule: AlertRule
    triggered_at: datetime
    current_value: float
    threshold: float
    severity: str = "warning"  # "info", "warning", "critical"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario."""
        return {
            "alert_id": self.alert_id,
            "rule": self.rule.to_dict(),
            "triggered_at": self.triggered_at.isoformat(),
            "current_value": self.current_value,
            "threshold": self.threshold,
            "severity": self.severity
        }


class LLMAnalytics:
    """
    Sistema de analytics avanzado para LLM Service.
    
    Características:
    - Tracking de métricas en tiempo real
    - Análisis de tendencias
    - Alertas automáticas
    - Reportes personalizados
    - Agregación por múltiples dimensiones
    """
    
    def __init__(self, storage_path: Optional[str] = None, retention_days: int = 30):
        """
        Inicializar analytics.
        
        Args:
            storage_path: Ruta para almacenar datos (opcional)
            retention_days: Días de retención de datos
        """
        self.storage_path = storage_path or "data/llm_analytics"
        self.retention_days = retention_days
        
        # Métricas en memoria (últimas 24 horas)
        self.metrics: Dict[str, List[MetricPoint]] = defaultdict(list)
        
        # Alertas
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        
        # Agregaciones
        self.aggregations: Dict[str, Dict[str, Any]] = defaultdict(dict)
        
        # Crear directorio si no existe
        os.makedirs(self.storage_path, exist_ok=True)
        
        # Cargar reglas de alertas
        self._load_alert_rules()
    
    def record_metric(
        self,
        metric_type: MetricType,
        value: float,
        tags: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Registrar una métrica.
        
        Args:
            metric_type: Tipo de métrica
            value: Valor de la métrica
            tags: Tags adicionales (modelo, endpoint, etc.)
        """
        point = MetricPoint(
            timestamp=datetime.now(),
            value=value,
            tags=tags or {}
        )
        
        key = metric_type.value
        self.metrics[key].append(point)
        
        # Limpiar métricas antiguas (más de 24 horas)
        cutoff = datetime.now() - timedelta(hours=24)
        self.metrics[key] = [
            p for p in self.metrics[key]
            if p.timestamp > cutoff
        ]
        
        # Actualizar agregaciones
        self._update_aggregations(metric_type, value, tags)
        
        # Verificar alertas
        self._check_alerts(metric_type, value)
    
    def get_metrics(
        self,
        metric_type: MetricType,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        tags: Optional[Dict[str, str]] = None,
        aggregation: Optional[str] = None  # "avg", "sum", "min", "max", "count"
    ) -> List[MetricPoint]:
        """
        Obtener métricas con filtros.
        
        Args:
            metric_type: Tipo de métrica
            start_time: Tiempo de inicio (opcional)
            end_time: Tiempo de fin (opcional)
            tags: Filtrar por tags (opcional)
            aggregation: Tipo de agregación (opcional)
            
        Returns:
            Lista de puntos de métrica
        """
        key = metric_type.value
        points = self.metrics.get(key, [])
        
        # Filtrar por tiempo
        if start_time:
            points = [p for p in points if p.timestamp >= start_time]
        if end_time:
            points = [p for p in points if p.timestamp <= end_time]
        
        # Filtrar por tags
        if tags:
            points = [
                p for p in points
                if all(p.tags.get(k) == v for k, v in tags.items())
            ]
        
        # Aplicar agregación si se solicita
        if aggregation and points:
            return self._aggregate_points(points, aggregation)
        
        return points
    
    def get_statistics(
        self,
        metric_type: MetricType,
        window_minutes: int = 60,
        tags: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Obtener estadísticas de una métrica.
        
        Args:
            metric_type: Tipo de métrica
            window_minutes: Ventana de tiempo en minutos
            tags: Filtrar por tags (opcional)
            
        Returns:
            Estadísticas (min, max, avg, sum, count, p50, p95, p99)
        """
        end_time = datetime.now()
        start_time = end_time - timedelta(minutes=window_minutes)
        
        points = self.get_metrics(metric_type, start_time, end_time, tags)
        
        if not points:
            return {
                "count": 0,
                "min": 0,
                "max": 0,
                "avg": 0,
                "sum": 0,
                "p50": 0,
                "p95": 0,
                "p99": 0
            }
        
        values = [p.value for p in points]
        values_sorted = sorted(values)
        
        def percentile(data, p):
            if not data:
                return 0
            k = (len(data) - 1) * p
            f = int(k)
            c = k - f
            if f + 1 < len(data):
                return data[f] + c * (data[f + 1] - data[f])
            return data[f]
        
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "avg": sum(values) / len(values),
            "sum": sum(values),
            "p50": percentile(values_sorted, 0.50),
            "p95": percentile(values_sorted, 0.95),
            "p99": percentile(values_sorted, 0.99)
        }
    
    def create_alert_rule(
        self,
        name: str,
        metric_type: MetricType,
        threshold: float,
        comparison: str = "gt",
        window_minutes: int = 5,
        notification_channels: Optional[List[str]] = None
    ) -> str:
        """
        Crear regla de alerta.
        
        Args:
            name: Nombre de la alerta
            metric_type: Tipo de métrica a monitorear
            threshold: Umbral de alerta
            comparison: Comparación ("gt", "lt", "eq")
            window_minutes: Ventana de tiempo en minutos
            notification_channels: Canales de notificación (opcional)
            
        Returns:
            ID de la alerta
        """
        import hashlib
        
        alert_id = hashlib.md5(
            f"{name}{metric_type.value}{threshold}{datetime.now().isoformat()}".encode()
        ).hexdigest()[:12]
        
        rule = AlertRule(
            alert_id=alert_id,
            name=name,
            metric_type=metric_type,
            threshold=threshold,
            comparison=comparison,
            window_minutes=window_minutes,
            notification_channels=notification_channels or []
        )
        
        self.alert_rules[alert_id] = rule
        self._save_alert_rule(rule)
        
        logger.info(f"Regla de alerta creada: {alert_id} - {name}")
        return alert_id
    
    def get_active_alerts(self) -> List[Alert]:
        """Obtener alertas activas."""
        return list(self.active_alerts.values())
    
    def get_trends(
        self,
        metric_type: MetricType,
        periods: int = 7,
        period_hours: int = 24
    ) -> Dict[str, Any]:
        """
        Obtener tendencias de una métrica.
        
        Args:
            metric_type: Tipo de métrica
            periods: Número de períodos
            period_hours: Horas por período
            
        Returns:
            Tendencias con comparación período a período
        """
        trends = []
        
        for i in range(periods):
            end_time = datetime.now() - timedelta(hours=i * period_hours)
            start_time = end_time - timedelta(hours=period_hours)
            
            stats = self.get_statistics(
                metric_type,
                window_minutes=period_hours * 60
            )
            
            trends.append({
                "period": i,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "stats": stats
            })
        
        # Calcular cambios
        if len(trends) >= 2:
            current = trends[0]["stats"]["avg"]
            previous = trends[1]["stats"]["avg"]
            
            if previous > 0:
                change_percent = ((current - previous) / previous) * 100
            else:
                change_percent = 0
            
            trends[0]["change_percent"] = change_percent
        
        return {
            "metric_type": metric_type.value,
            "periods": trends,
            "current_period": trends[0] if trends else None
        }
    
    def _update_aggregations(
        self,
        metric_type: MetricType,
        value: float,
        tags: Optional[Dict[str, str]]
    ) -> None:
        """Actualizar agregaciones."""
        key = metric_type.value
        
        if key not in self.aggregations:
            self.aggregations[key] = {
                "count": 0,
                "sum": 0.0,
                "min": float('inf'),
                "max": float('-inf')
            }
        
        agg = self.aggregations[key]
        agg["count"] += 1
        agg["sum"] += value
        agg["min"] = min(agg["min"], value)
        agg["max"] = max(agg["max"], value)
        agg["avg"] = agg["sum"] / agg["count"] if agg["count"] > 0 else 0
    
    def _check_alerts(self, metric_type: MetricType, value: float) -> None:
        """Verificar reglas de alerta."""
        for rule in self.alert_rules.values():
            if not rule.enabled:
                continue
            
            if rule.metric_type != metric_type:
                continue
            
            # Obtener estadísticas de la ventana
            stats = self.get_statistics(
                metric_type,
                window_minutes=rule.window_minutes
            )
            
            avg_value = stats.get("avg", 0)
            
            # Verificar condición
            should_alert = False
            if rule.comparison == "gt" and avg_value > rule.threshold:
                should_alert = True
            elif rule.comparison == "lt" and avg_value < rule.threshold:
                should_alert = True
            elif rule.comparison == "eq" and abs(avg_value - rule.threshold) < 0.01:
                should_alert = True
            
            if should_alert:
                # Verificar si ya existe una alerta activa
                if rule.alert_id not in self.active_alerts:
                    alert = Alert(
                        alert_id=rule.alert_id,
                        rule=rule,
                        triggered_at=datetime.now(),
                        current_value=avg_value,
                        threshold=rule.threshold,
                        severity="critical" if abs(avg_value - rule.threshold) > rule.threshold * 0.5 else "warning"
                    )
                    
                    self.active_alerts[rule.alert_id] = alert
                    
                    # Enviar notificaciones
                    self._send_notifications(alert)
                    
                    logger.warning(
                        f"Alerta activada: {rule.name} - "
                        f"{metric_type.value} = {avg_value} "
                        f"({rule.comparison} {rule.threshold})"
                    )
            else:
                # Remover alerta si ya no se cumple la condición
                if rule.alert_id in self.active_alerts:
                    del self.active_alerts[rule.alert_id]
                    logger.info(f"Alerta resuelta: {rule.name}")
    
    def _send_notifications(self, alert: Alert) -> None:
        """Enviar notificaciones de alerta."""
        # Integrar con webhook service
        try:
            from core.services.llm import get_webhook_service, WebhookEvent
            
            webhook_service = get_webhook_service()
            
            # Determinar evento apropiado
            event_map = {
                MetricType.ERROR_RATE: WebhookEvent.GENERATION_FAILED,
                MetricType.LATENCY: WebhookEvent.GENERATION_FAILED,
                MetricType.COST: WebhookEvent.COST_THRESHOLD_REACHED,
                MetricType.RATE_LIMIT_EXCEEDED: WebhookEvent.RATE_LIMIT_EXCEEDED,
            }
            
            event = event_map.get(alert.rule.metric_type, WebhookEvent.GENERATION_FAILED)
            
            # Nota: trigger_webhook es async, pero estamos en método sync
            # En producción, esto debería manejarse de forma asíncrona
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Si hay un loop corriendo, crear tarea
                    asyncio.create_task(webhook_service.trigger_webhook(
                        event,
                        data={
                            "alert_id": alert.alert_id,
                            "rule_name": alert.rule.name,
                            "metric_type": alert.rule.metric_type.value,
                            "current_value": alert.current_value,
                            "threshold": alert.threshold,
                            "severity": alert.severity
                        }
                    ))
                else:
                    loop.run_until_complete(webhook_service.trigger_webhook(
                        event,
                        data={
                            "alert_id": alert.alert_id,
                            "rule_name": alert.rule.name,
                            "metric_type": alert.rule.metric_type.value,
                            "current_value": alert.current_value,
                            "threshold": alert.threshold,
                            "severity": alert.severity
                        }
                    ))
            except RuntimeError:
                # No hay event loop, crear uno nuevo
                asyncio.run(webhook_service.trigger_webhook(
                    event,
                    data={
                        "alert_id": alert.alert_id,
                        "rule_name": alert.rule.name,
                        "metric_type": alert.rule.metric_type.value,
                        "current_value": alert.current_value,
                        "threshold": alert.threshold,
                        "severity": alert.severity
                    }
                ))
        except Exception as e:
            logger.error(f"Error enviando notificaciones: {e}")
    
    def _aggregate_points(
        self,
        points: List[MetricPoint],
        aggregation: str
    ) -> List[MetricPoint]:
        """Agregar puntos de métrica."""
        if not points:
            return []
        
        if aggregation == "avg":
            avg_value = sum(p.value for p in points) / len(points)
            return [MetricPoint(
                timestamp=points[0].timestamp,
                value=avg_value,
                tags=points[0].tags
            )]
        elif aggregation == "sum":
            sum_value = sum(p.value for p in points)
            return [MetricPoint(
                timestamp=points[0].timestamp,
                value=sum_value,
                tags=points[0].tags
            )]
        elif aggregation == "min":
            min_value = min(p.value for p in points)
            return [MetricPoint(
                timestamp=points[0].timestamp,
                value=min_value,
                tags=points[0].tags
            )]
        elif aggregation == "max":
            max_value = max(p.value for p in points)
            return [MetricPoint(
                timestamp=points[0].timestamp,
                value=max_value,
                tags=points[0].tags
            )]
        elif aggregation == "count":
            return [MetricPoint(
                timestamp=points[0].timestamp,
                value=len(points),
                tags=points[0].tags
            )]
        
        return points
    
    def _save_alert_rule(self, rule: AlertRule) -> None:
        """Guardar regla de alerta."""
        file_path = os.path.join(self.storage_path, f"alert_{rule.alert_id}.json")
        with open(file_path, 'w') as f:
            json.dump(rule.to_dict(), f, indent=2, default=str)
    
    def _load_alert_rules(self) -> None:
        """Cargar reglas de alerta."""
        if not os.path.exists(self.storage_path):
            return
        
        for filename in os.listdir(self.storage_path):
            if filename.startswith("alert_") and filename.endswith(".json"):
                file_path = os.path.join(self.storage_path, filename)
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    
                    rule = AlertRule(
                        alert_id=data["alert_id"],
                        name=data["name"],
                        metric_type=MetricType(data["metric_type"]),
                        threshold=data["threshold"],
                        comparison=data["comparison"],
                        window_minutes=data.get("window_minutes", 5),
                        enabled=data.get("enabled", True),
                        notification_channels=data.get("notification_channels", [])
                    )
                    
                    self.alert_rules[rule.alert_id] = rule
                except Exception as e:
                    logger.error(f"Error cargando regla de alerta desde {filename}: {e}")


def get_llm_analytics(
    storage_path: Optional[str] = None,
    retention_days: int = 30
) -> LLMAnalytics:
    """Factory function para obtener instancia singleton de analytics."""
    if not hasattr(get_llm_analytics, "_instance"):
        get_llm_analytics._instance = LLMAnalytics(storage_path, retention_days)
    return get_llm_analytics._instance

