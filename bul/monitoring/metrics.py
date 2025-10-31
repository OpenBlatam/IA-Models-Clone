"""
Metrics and Monitoring
======================

Sistema de métricas y monitoreo para el sistema BUL.
"""

import asyncio
import time
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import uuid
from datetime import datetime, timedelta
from collections import defaultdict, deque
import json

logger = logging.getLogger(__name__)

class MetricType(Enum):
    """Tipos de métricas"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"

@dataclass
class Metric:
    """Definición de métrica"""
    name: str
    type: MetricType
    value: Union[int, float]
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Alert:
    """Definición de alerta"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    metric_name: str = ""
    condition: str = ""  # e.g., "> 100", "< 0.5"
    severity: str = "warning"  # info, warning, error, critical
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    last_triggered: Optional[datetime] = None
    trigger_count: int = 0

class MetricsCollector:
    """
    Recolector de métricas
    
    Recolecta y almacena métricas del sistema BUL.
    """
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history))
        self.alerts: Dict[str, Alert] = {}
        self.is_initialized = False
        
        logger.info("Metrics Collector initialized")
    
    async def initialize(self) -> bool:
        """Inicializar el recolector de métricas"""
        try:
            await self._setup_default_alerts()
            self.is_initialized = True
            logger.info("Metrics Collector fully initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Metrics Collector: {e}")
            return False
    
    async def _setup_default_alerts(self):
        """Configurar alertas por defecto"""
        default_alerts = [
            {
                "name": "High Processing Time",
                "description": "Tiempo de procesamiento de documentos muy alto",
                "metric_name": "document_processing_time",
                "condition": "> 30",
                "severity": "warning"
            },
            {
                "name": "Low Confidence Score",
                "description": "Puntuación de confianza muy baja",
                "metric_name": "document_confidence_score",
                "condition": "< 0.7",
                "severity": "error"
            },
            {
                "name": "High Error Rate",
                "description": "Tasa de errores muy alta",
                "metric_name": "api_error_rate",
                "condition": "> 0.1",
                "severity": "critical"
            },
            {
                "name": "Low Cache Hit Rate",
                "description": "Tasa de aciertos de cache muy baja",
                "metric_name": "cache_hit_rate",
                "condition": "< 0.5",
                "severity": "warning"
            }
        ]
        
        for alert_data in default_alerts:
            alert = Alert(**alert_data)
            self.alerts[alert.id] = alert
            logger.info(f"Setup alert: {alert.name}")
    
    def record_metric(self, name: str, value: Union[int, float], 
                     metric_type: MetricType = MetricType.GAUGE,
                     labels: Dict[str, str] = None):
        """Registrar una métrica"""
        metric = Metric(
            name=name,
            type=metric_type,
            value=value,
            labels=labels or {}
        )
        
        self.metrics[name].append(metric)
        
        # Verificar alertas
        asyncio.create_task(self._check_alerts(metric))
    
    def increment_counter(self, name: str, value: int = 1, labels: Dict[str, str] = None):
        """Incrementar contador"""
        self.record_metric(name, value, MetricType.COUNTER, labels)
    
    def set_gauge(self, name: str, value: Union[int, float], labels: Dict[str, str] = None):
        """Establecer gauge"""
        self.record_metric(name, value, MetricType.GAUGE, labels)
    
    def record_histogram(self, name: str, value: Union[int, float], labels: Dict[str, str] = None):
        """Registrar histograma"""
        self.record_metric(name, value, MetricType.HISTOGRAM, labels)
    
    def record_timer(self, name: str, duration: float, labels: Dict[str, str] = None):
        """Registrar tiempo"""
        self.record_metric(name, duration, MetricType.TIMER, labels)
    
    async def _check_alerts(self, metric: Metric):
        """Verificar alertas para una métrica"""
        for alert in self.alerts.values():
            if not alert.is_active or alert.metric_name != metric.name:
                continue
            
            if self._evaluate_condition(metric.value, alert.condition):
                await self._trigger_alert(alert, metric)
    
    def _evaluate_condition(self, value: Union[int, float], condition: str) -> bool:
        """Evaluar condición de alerta"""
        try:
            if condition.startswith(">"):
                threshold = float(condition[1:].strip())
                return value > threshold
            elif condition.startswith("<"):
                threshold = float(condition[1:].strip())
                return value < threshold
            elif condition.startswith(">="):
                threshold = float(condition[2:].strip())
                return value >= threshold
            elif condition.startswith("<="):
                threshold = float(condition[2:].strip())
                return value <= threshold
            elif condition.startswith("=="):
                threshold = float(condition[2:].strip())
                return value == threshold
            else:
                return False
        except (ValueError, TypeError):
            return False
    
    async def _trigger_alert(self, alert: Alert, metric: Metric):
        """Disparar alerta"""
        alert.last_triggered = datetime.now()
        alert.trigger_count += 1
        
        logger.warning(f"ALERT TRIGGERED: {alert.name} - {alert.description}")
        logger.warning(f"Metric: {metric.name} = {metric.value}, Condition: {alert.condition}")
        logger.warning(f"Severity: {alert.severity}")
        
        # En una implementación real, aquí enviarías notificaciones
        # (email, Slack, webhook, etc.)
    
    def get_metric_history(self, name: str, minutes: int = 60) -> List[Metric]:
        """Obtener historial de métrica"""
        if name not in self.metrics:
            return []
        
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        return [m for m in self.metrics[name] if m.timestamp >= cutoff_time]
    
    def get_metric_summary(self, name: str, minutes: int = 60) -> Dict[str, Any]:
        """Obtener resumen de métrica"""
        history = self.get_metric_history(name, minutes)
        
        if not history:
            return {"count": 0}
        
        values = [m.value for m in history]
        
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "avg": sum(values) / len(values),
            "latest": values[-1] if values else None,
            "period_minutes": minutes
        }
    
    def get_all_metrics_summary(self, minutes: int = 60) -> Dict[str, Any]:
        """Obtener resumen de todas las métricas"""
        summary = {}
        
        for metric_name in self.metrics.keys():
            summary[metric_name] = self.get_metric_summary(metric_name, minutes)
        
        return summary
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Obtener alertas activas"""
        return [
            {
                "id": alert.id,
                "name": alert.name,
                "description": alert.description,
                "metric_name": alert.metric_name,
                "condition": alert.condition,
                "severity": alert.severity,
                "last_triggered": alert.last_triggered.isoformat() if alert.last_triggered else None,
                "trigger_count": alert.trigger_count
            }
            for alert in self.alerts.values()
            if alert.is_active
        ]
    
    def add_alert(self, alert: Alert):
        """Agregar nueva alerta"""
        self.alerts[alert.id] = alert
        logger.info(f"Added alert: {alert.name}")
    
    def remove_alert(self, alert_id: str):
        """Eliminar alerta"""
        if alert_id in self.alerts:
            del self.alerts[alert_id]
            logger.info(f"Removed alert: {alert_id}")

class PerformanceMonitor:
    """
    Monitor de rendimiento
    
    Monitorea el rendimiento del sistema BUL.
    """
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
        self.start_times: Dict[str, float] = {}
        
    def start_timer(self, operation: str):
        """Iniciar temporizador"""
        self.start_times[operation] = time.time()
    
    def end_timer(self, operation: str, labels: Dict[str, str] = None):
        """Finalizar temporizador"""
        if operation in self.start_times:
            duration = time.time() - self.start_times[operation]
            self.metrics.record_timer(f"{operation}_duration", duration, labels)
            del self.start_times[operation]
            return duration
        return 0.0
    
    def record_document_generation(self, processing_time: float, word_count: int, 
                                 confidence_score: float, business_area: str, 
                                 document_type: str, agent_used: str):
        """Registrar generación de documento"""
        labels = {
            "business_area": business_area,
            "document_type": document_type,
            "agent": agent_used
        }
        
        self.metrics.record_timer("document_processing_time", processing_time, labels)
        self.metrics.set_gauge("document_word_count", word_count, labels)
        self.metrics.set_gauge("document_confidence_score", confidence_score, labels)
        self.metrics.increment_counter("documents_generated", 1, labels)
    
    def record_api_request(self, endpoint: str, method: str, status_code: int, 
                          response_time: float, user_id: str = None):
        """Registrar solicitud de API"""
        labels = {
            "endpoint": endpoint,
            "method": method,
            "status_code": str(status_code)
        }
        
        if user_id:
            labels["user_id"] = user_id
        
        self.metrics.record_timer("api_response_time", response_time, labels)
        self.metrics.increment_counter("api_requests", 1, labels)
        
        if status_code >= 400:
            self.metrics.increment_counter("api_errors", 1, labels)
    
    def record_cache_operation(self, operation: str, hit: bool, key: str = None):
        """Registrar operación de cache"""
        labels = {
            "operation": operation,
            "hit": str(hit)
        }
        
        if key:
            labels["key_prefix"] = key.split(":")[0] if ":" in key else "unknown"
        
        self.metrics.increment_counter("cache_operations", 1, labels)
        
        if hit:
            self.metrics.increment_counter("cache_hits", 1, labels)
        else:
            self.metrics.increment_counter("cache_misses", 1, labels)
    
    def record_agent_usage(self, agent_id: str, agent_type: str, success: bool, 
                          processing_time: float, confidence_score: float):
        """Registrar uso de agente"""
        labels = {
            "agent_id": agent_id,
            "agent_type": agent_type,
            "success": str(success)
        }
        
        self.metrics.record_timer("agent_processing_time", processing_time, labels)
        self.metrics.set_gauge("agent_confidence_score", confidence_score, labels)
        self.metrics.increment_counter("agent_usage", 1, labels)
        
        if not success:
            self.metrics.increment_counter("agent_errors", 1, labels)
    
    def calculate_cache_hit_rate(self) -> float:
        """Calcular tasa de aciertos de cache"""
        hits = sum(1 for m in self.metrics.get_metric_history("cache_hits", 60) 
                  if m.value > 0)
        misses = sum(1 for m in self.metrics.get_metric_history("cache_misses", 60) 
                    if m.value > 0)
        
        total = hits + misses
        return hits / total if total > 0 else 0.0
    
    def calculate_error_rate(self) -> float:
        """Calcular tasa de errores"""
        errors = sum(1 for m in self.metrics.get_metric_history("api_errors", 60) 
                    if m.value > 0)
        total = sum(1 for m in self.metrics.get_metric_history("api_requests", 60) 
                   if m.value > 0)
        
        return errors / total if total > 0 else 0.0

# Global instances
_metrics_collector: Optional[MetricsCollector] = None
_performance_monitor: Optional[PerformanceMonitor] = None

async def get_global_metrics_collector() -> MetricsCollector:
    """Obtener la instancia global del recolector de métricas"""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
        await _metrics_collector.initialize()
    return _metrics_collector

async def get_global_performance_monitor() -> PerformanceMonitor:
    """Obtener la instancia global del monitor de rendimiento"""
    global _performance_monitor
    if _performance_monitor is None:
        metrics_collector = await get_global_metrics_collector()
        _performance_monitor = PerformanceMonitor(metrics_collector)
    return _performance_monitor
























