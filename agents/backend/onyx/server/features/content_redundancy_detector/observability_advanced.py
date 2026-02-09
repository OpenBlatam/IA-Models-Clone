"""
Advanced Observability Support for Comprehensive Monitoring
Sistema de observabilidad avanzada para monitoreo comprehensivo ultra-optimizado
"""

import asyncio
import logging
import time
import json
import psutil
import threading
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, asdict
from enum import Enum
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.routing import APIRouter
from collections import defaultdict, deque
import numpy as np

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Tipos de métricas"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class AlertSeverity(Enum):
    """Severidad de alertas"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class LogLevel(Enum):
    """Niveles de log"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class MetricData:
    """Datos de métrica"""
    name: str
    type: MetricType
    value: Union[float, int]
    labels: Dict[str, str]
    timestamp: float
    help: str = ""


@dataclass
class AlertRule:
    """Regla de alerta"""
    id: str
    name: str
    expression: str
    severity: AlertSeverity
    threshold: float
    duration: float
    enabled: bool
    labels: Dict[str, str]
    annotations: Dict[str, str]


@dataclass
class Alert:
    """Alerta"""
    id: str
    rule_id: str
    name: str
    severity: AlertSeverity
    status: str  # firing, resolved
    value: float
    threshold: float
    labels: Dict[str, str]
    annotations: Dict[str, str]
    starts_at: float
    ends_at: Optional[float]
    generator_url: str


@dataclass
class TraceSpan:
    """Span de trace"""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    operation_name: str
    start_time: float
    end_time: float
    duration: float
    tags: Dict[str, Any]
    logs: List[Dict[str, Any]]
    status: str


@dataclass
class LogEntry:
    """Entrada de log"""
    timestamp: float
    level: LogLevel
    message: str
    source: str
    trace_id: Optional[str]
    span_id: Optional[str]
    labels: Dict[str, str]
    fields: Dict[str, Any]


class MetricsCollector:
    """Colector de métricas"""
    
    def __init__(self):
        self.metrics: Dict[str, List[MetricData]] = defaultdict(list)
        self.counters: Dict[str, float] = defaultdict(float)
        self.gauges: Dict[str, float] = defaultdict(float)
        self.histograms: Dict[str, List[float]] = defaultdict(list)
        self.summaries: Dict[str, Dict[str, float]] = defaultdict(dict)
        self._lock = threading.Lock()
    
    def increment_counter(self, name: str, value: float = 1.0, labels: Dict[str, str] = None):
        """Incrementar contador"""
        with self._lock:
            key = f"{name}:{json.dumps(labels or {}, sort_keys=True)}"
            self.counters[key] += value
            
            metric = MetricData(
                name=name,
                type=MetricType.COUNTER,
                value=self.counters[key],
                labels=labels or {},
                timestamp=time.time()
            )
            self.metrics[name].append(metric)
    
    def set_gauge(self, name: str, value: float, labels: Dict[str, str] = None):
        """Establecer gauge"""
        with self._lock:
            key = f"{name}:{json.dumps(labels or {}, sort_keys=True)}"
            self.gauges[key] = value
            
            metric = MetricData(
                name=name,
                type=MetricType.GAUGE,
                value=value,
                labels=labels or {},
                timestamp=time.time()
            )
            self.metrics[name].append(metric)
    
    def observe_histogram(self, name: str, value: float, labels: Dict[str, str] = None):
        """Observar histograma"""
        with self._lock:
            key = f"{name}:{json.dumps(labels or {}, sort_keys=True)}"
            self.histograms[key].append(value)
            
            # Mantener solo los últimos 1000 valores
            if len(self.histograms[key]) > 1000:
                self.histograms[key] = self.histograms[key][-1000:]
            
            metric = MetricData(
                name=name,
                type=MetricType.HISTOGRAM,
                value=value,
                labels=labels or {},
                timestamp=time.time()
            )
            self.metrics[name].append(metric)
    
    def observe_summary(self, name: str, value: float, labels: Dict[str, str] = None):
        """Observar summary"""
        with self._lock:
            key = f"{name}:{json.dumps(labels or {}, sort_keys=True)}"
            
            if key not in self.summaries:
                self.summaries[key] = {
                    "count": 0,
                    "sum": 0.0,
                    "min": float('inf'),
                    "max": float('-inf')
                }
            
            summary = self.summaries[key]
            summary["count"] += 1
            summary["sum"] += value
            summary["min"] = min(summary["min"], value)
            summary["max"] = max(summary["max"], value)
            
            metric = MetricData(
                name=name,
                type=MetricType.SUMMARY,
                value=value,
                labels=labels or {},
                timestamp=time.time()
            )
            self.metrics[name].append(metric)
    
    def get_metric(self, name: str, labels: Dict[str, str] = None) -> Optional[MetricData]:
        """Obtener métrica"""
        with self._lock:
            if name not in self.metrics:
                return None
            
            metrics = self.metrics[name]
            if not metrics:
                return None
            
            # Buscar métrica con labels específicos
            if labels:
                for metric in reversed(metrics):
                    if metric.labels == labels:
                        return metric
            else:
                return metrics[-1]  # Última métrica
            
            return None
    
    def get_all_metrics(self) -> Dict[str, List[MetricData]]:
        """Obtener todas las métricas"""
        with self._lock:
            return dict(self.metrics)
    
    def get_metric_stats(self, name: str) -> Dict[str, Any]:
        """Obtener estadísticas de métrica"""
        with self._lock:
            if name not in self.metrics:
                return {}
            
            metrics = self.metrics[name]
            if not metrics:
                return {}
            
            values = [m.value for m in metrics]
            
            return {
                "count": len(values),
                "min": min(values),
                "max": max(values),
                "mean": np.mean(values),
                "median": np.median(values),
                "std": np.std(values),
                "last_value": values[-1],
                "last_timestamp": metrics[-1].timestamp
            }


class AlertManager:
    """Manager de alertas"""
    
    def __init__(self):
        self.rules: Dict[str, AlertRule] = {}
        self.alerts: Dict[str, Alert] = {}
        self.metrics_collector: Optional[MetricsCollector] = None
        self._lock = threading.Lock()
    
    def set_metrics_collector(self, collector: MetricsCollector):
        """Establecer colector de métricas"""
        self.metrics_collector = collector
    
    def add_rule(self, rule: AlertRule):
        """Agregar regla de alerta"""
        with self._lock:
            self.rules[rule.id] = rule
    
    def remove_rule(self, rule_id: str):
        """Remover regla de alerta"""
        with self._lock:
            if rule_id in self.rules:
                del self.rules[rule_id]
    
    def evaluate_rules(self):
        """Evaluar reglas de alerta"""
        if not self.metrics_collector:
            return
        
        with self._lock:
            for rule_id, rule in self.rules.items():
                if not rule.enabled:
                    continue
                
                try:
                    # Evaluar expresión de la regla
                    value = self._evaluate_expression(rule.expression)
                    
                    if value is not None and value >= rule.threshold:
                        # Crear o actualizar alerta
                        alert_id = f"alert_{rule_id}_{int(time.time())}"
                        
                        if alert_id not in self.alerts:
                            alert = Alert(
                                id=alert_id,
                                rule_id=rule_id,
                                name=rule.name,
                                severity=rule.severity,
                                status="firing",
                                value=value,
                                threshold=rule.threshold,
                                labels=rule.labels.copy(),
                                annotations=rule.annotations.copy(),
                                starts_at=time.time(),
                                ends_at=None,
                                generator_url=f"/alerts/{alert_id}"
                            )
                            self.alerts[alert_id] = alert
                            logger.warning(f"Alert fired: {rule.name} (value: {value}, threshold: {rule.threshold})")
                    else:
                        # Resolver alertas existentes para esta regla
                        for alert_id, alert in list(self.alerts.items()):
                            if alert.rule_id == rule_id and alert.status == "firing":
                                alert.status = "resolved"
                                alert.ends_at = time.time()
                                logger.info(f"Alert resolved: {rule.name}")
                
                except Exception as e:
                    logger.error(f"Error evaluating rule {rule.name}: {e}")
    
    def _evaluate_expression(self, expression: str) -> Optional[float]:
        """Evaluar expresión de métrica"""
        try:
            # Expresiones simples como "metric_name" o "metric_name{label=value}"
            if "{" in expression:
                # Extraer nombre de métrica y labels
                metric_name = expression.split("{")[0]
                labels_str = expression.split("{")[1].rstrip("}")
                labels = {}
                
                if labels_str:
                    for pair in labels_str.split(","):
                        if "=" in pair:
                            key, value = pair.split("=", 1)
                            labels[key.strip()] = value.strip().strip('"')
            else:
                metric_name = expression
                labels = {}
            
            # Obtener métrica
            metric = self.metrics_collector.get_metric(metric_name, labels)
            return metric.value if metric else None
            
        except Exception as e:
            logger.error(f"Error evaluating expression {expression}: {e}")
            return None
    
    def get_alerts(self, status: Optional[str] = None) -> List[Alert]:
        """Obtener alertas"""
        with self._lock:
            alerts = list(self.alerts.values())
            if status:
                alerts = [alert for alert in alerts if alert.status == status]
            return alerts
    
    def get_rules(self) -> List[AlertRule]:
        """Obtener reglas"""
        with self._lock:
            return list(self.rules.values())


class TraceCollector:
    """Colector de traces"""
    
    def __init__(self):
        self.traces: Dict[str, List[TraceSpan]] = defaultdict(list)
        self.spans: Dict[str, TraceSpan] = {}
        self._lock = threading.Lock()
    
    def start_span(self, trace_id: str, span_id: str, operation_name: str,
                   parent_span_id: Optional[str] = None, tags: Dict[str, Any] = None) -> TraceSpan:
        """Iniciar span"""
        with self._lock:
            span = TraceSpan(
                trace_id=trace_id,
                span_id=span_id,
                parent_span_id=parent_span_id,
                operation_name=operation_name,
                start_time=time.time(),
                end_time=0,
                duration=0,
                tags=tags or {},
                logs=[],
                status="started"
            )
            
            self.spans[span_id] = span
            self.traces[trace_id].append(span)
            
            return span
    
    def finish_span(self, span_id: str, status: str = "ok", tags: Dict[str, Any] = None):
        """Finalizar span"""
        with self._lock:
            if span_id in self.spans:
                span = self.spans[span_id]
                span.end_time = time.time()
                span.duration = span.end_time - span.start_time
                span.status = status
                
                if tags:
                    span.tags.update(tags)
    
    def add_span_log(self, span_id: str, message: str, fields: Dict[str, Any] = None):
        """Agregar log a span"""
        with self._lock:
            if span_id in self.spans:
                span = self.spans[span_id]
                log_entry = {
                    "timestamp": time.time(),
                    "message": message,
                    "fields": fields or {}
                }
                span.logs.append(log_entry)
    
    def get_trace(self, trace_id: str) -> List[TraceSpan]:
        """Obtener trace"""
        with self._lock:
            return self.traces.get(trace_id, [])
    
    def get_span(self, span_id: str) -> Optional[TraceSpan]:
        """Obtener span"""
        with self._lock:
            return self.spans.get(span_id)
    
    def get_trace_stats(self, trace_id: str) -> Dict[str, Any]:
        """Obtener estadísticas de trace"""
        with self._lock:
            spans = self.traces.get(trace_id, [])
            if not spans:
                return {}
            
            durations = [span.duration for span in spans if span.duration > 0]
            
            return {
                "span_count": len(spans),
                "total_duration": sum(durations),
                "min_duration": min(durations) if durations else 0,
                "max_duration": max(durations) if durations else 0,
                "avg_duration": sum(durations) / len(durations) if durations else 0
            }


class LogCollector:
    """Colector de logs"""
    
    def __init__(self):
        self.logs: deque = deque(maxlen=10000)  # Mantener últimos 10000 logs
        self._lock = threading.Lock()
    
    def add_log(self, level: LogLevel, message: str, source: str = "system",
                trace_id: Optional[str] = None, span_id: Optional[str] = None,
                labels: Dict[str, str] = None, fields: Dict[str, Any] = None):
        """Agregar log"""
        with self._lock:
            log_entry = LogEntry(
                timestamp=time.time(),
                level=level,
                message=message,
                source=source,
                trace_id=trace_id,
                span_id=span_id,
                labels=labels or {},
                fields=fields or {}
            )
            self.logs.append(log_entry)
    
    def get_logs(self, level: Optional[LogLevel] = None, source: Optional[str] = None,
                 limit: int = 100) -> List[LogEntry]:
        """Obtener logs"""
        with self._lock:
            logs = list(self.logs)
            
            if level:
                logs = [log for log in logs if log.level == level]
            
            if source:
                logs = [log for log in logs if log.source == source]
            
            return logs[-limit:] if limit > 0 else logs
    
    def get_log_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de logs"""
        with self._lock:
            logs = list(self.logs)
            if not logs:
                return {}
            
            level_counts = defaultdict(int)
            source_counts = defaultdict(int)
            
            for log in logs:
                level_counts[log.level.value] += 1
                source_counts[log.source] += 1
            
            return {
                "total_logs": len(logs),
                "level_counts": dict(level_counts),
                "source_counts": dict(source_counts),
                "oldest_log": logs[0].timestamp if logs else None,
                "newest_log": logs[-1].timestamp if logs else None
            }


class SystemMonitor:
    """Monitor del sistema"""
    
    def __init__(self):
        self.metrics_collector: Optional[MetricsCollector] = None
        self.is_monitoring = False
        self._monitor_task: Optional[asyncio.Task] = None
    
    def set_metrics_collector(self, collector: MetricsCollector):
        """Establecer colector de métricas"""
        self.metrics_collector = collector
    
    async def start_monitoring(self, interval: float = 5.0):
        """Iniciar monitoreo del sistema"""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self._monitor_task = asyncio.create_task(self._monitor_loop(interval))
        logger.info("System monitoring started")
    
    async def stop_monitoring(self):
        """Detener monitoreo del sistema"""
        if not self.is_monitoring:
            return
        
        self.is_monitoring = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        
        logger.info("System monitoring stopped")
    
    async def _monitor_loop(self, interval: float):
        """Loop de monitoreo"""
        while self.is_monitoring:
            try:
                await self._collect_system_metrics()
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in system monitoring: {e}")
                await asyncio.sleep(interval)
    
    async def _collect_system_metrics(self):
        """Recolectar métricas del sistema"""
        if not self.metrics_collector:
            return
        
        try:
            # CPU
            cpu_percent = psutil.cpu_percent(interval=1)
            self.metrics_collector.set_gauge("system_cpu_percent", cpu_percent)
            
            # Memoria
            memory = psutil.virtual_memory()
            self.metrics_collector.set_gauge("system_memory_percent", memory.percent)
            self.metrics_collector.set_gauge("system_memory_used_bytes", memory.used)
            self.metrics_collector.set_gauge("system_memory_total_bytes", memory.total)
            
            # Disco
            disk = psutil.disk_usage('/')
            self.metrics_collector.set_gauge("system_disk_percent", disk.percent)
            self.metrics_collector.set_gauge("system_disk_used_bytes", disk.used)
            self.metrics_collector.set_gauge("system_disk_total_bytes", disk.total)
            
            # Red
            network = psutil.net_io_counters()
            self.metrics_collector.set_gauge("system_network_bytes_sent", network.bytes_sent)
            self.metrics_collector.set_gauge("system_network_bytes_recv", network.bytes_recv)
            
            # Procesos
            process_count = len(psutil.pids())
            self.metrics_collector.set_gauge("system_process_count", process_count)
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")


class ObservabilityManager:
    """Manager de observabilidad"""
    
    def __init__(self):
        self.metrics_collector = MetricsCollector()
        self.alert_manager = AlertManager()
        self.trace_collector = TraceCollector()
        self.log_collector = LogCollector()
        self.system_monitor = SystemMonitor()
        
        # Configurar dependencias
        self.alert_manager.set_metrics_collector(self.metrics_collector)
        self.system_monitor.set_metrics_collector(self.metrics_collector)
        
        # Configurar reglas de alerta por defecto
        self._setup_default_alert_rules()
        
        # Iniciar evaluación de alertas
        self._alert_evaluation_task = None
    
    def _setup_default_alert_rules(self):
        """Configurar reglas de alerta por defecto"""
        default_rules = [
            AlertRule(
                id="high_cpu_usage",
                name="High CPU Usage",
                expression="system_cpu_percent",
                severity=AlertSeverity.HIGH,
                threshold=80.0,
                duration=300.0,  # 5 minutos
                enabled=True,
                labels={"service": "system"},
                annotations={"description": "CPU usage is above 80%"}
            ),
            AlertRule(
                id="high_memory_usage",
                name="High Memory Usage",
                expression="system_memory_percent",
                severity=AlertSeverity.HIGH,
                threshold=85.0,
                duration=300.0,
                enabled=True,
                labels={"service": "system"},
                annotations={"description": "Memory usage is above 85%"}
            ),
            AlertRule(
                id="high_disk_usage",
                name="High Disk Usage",
                expression="system_disk_percent",
                severity=AlertSeverity.MEDIUM,
                threshold=90.0,
                duration=600.0,  # 10 minutos
                enabled=True,
                labels={"service": "system"},
                annotations={"description": "Disk usage is above 90%"}
            )
        ]
        
        for rule in default_rules:
            self.alert_manager.add_rule(rule)
    
    async def start(self):
        """Iniciar observabilidad"""
        try:
            # Iniciar monitoreo del sistema
            await self.system_monitor.start_monitoring()
            
            # Iniciar evaluación de alertas
            self._alert_evaluation_task = asyncio.create_task(self._alert_evaluation_loop())
            
            logger.info("Observability manager started")
            
        except Exception as e:
            logger.error(f"Error starting observability manager: {e}")
            raise
    
    async def stop(self):
        """Detener observabilidad"""
        try:
            # Detener monitoreo del sistema
            await self.system_monitor.stop_monitoring()
            
            # Detener evaluación de alertas
            if self._alert_evaluation_task:
                self._alert_evaluation_task.cancel()
                try:
                    await self._alert_evaluation_task
                except asyncio.CancelledError:
                    pass
            
            logger.info("Observability manager stopped")
            
        except Exception as e:
            logger.error(f"Error stopping observability manager: {e}")
    
    async def _alert_evaluation_loop(self):
        """Loop de evaluación de alertas"""
        while True:
            try:
                self.alert_manager.evaluate_rules()
                await asyncio.sleep(30)  # Evaluar cada 30 segundos
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in alert evaluation: {e}")
                await asyncio.sleep(30)
    
    def get_observability_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de observabilidad"""
        return {
            "metrics": {
                "total_metrics": len(self.metrics_collector.get_all_metrics()),
                "counters": len(self.metrics_collector.counters),
                "gauges": len(self.metrics_collector.gauges),
                "histograms": len(self.metrics_collector.histograms),
                "summaries": len(self.metrics_collector.summaries)
            },
            "alerts": {
                "total_rules": len(self.alert_manager.get_rules()),
                "firing_alerts": len(self.alert_manager.get_alerts("firing")),
                "resolved_alerts": len(self.alert_manager.get_alerts("resolved"))
            },
            "traces": {
                "total_traces": len(self.trace_collector.traces),
                "total_spans": len(self.trace_collector.spans)
            },
            "logs": self.log_collector.get_log_stats(),
            "system_monitoring": {
                "is_monitoring": self.system_monitor.is_monitoring
            }
        }


# Instancia global del manager de observabilidad
observability_manager = ObservabilityManager()


# Router para endpoints de observabilidad
observability_router = APIRouter()


@observability_router.get("/observability/metrics")
async def get_metrics_endpoint():
    """Obtener métricas"""
    try:
        metrics = observability_manager.metrics_collector.get_all_metrics()
        return {"metrics": metrics}
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")


@observability_router.get("/observability/metrics/{metric_name}")
async def get_metric_endpoint(metric_name: str, labels: Optional[str] = None):
    """Obtener métrica específica"""
    try:
        metric_labels = json.loads(labels) if labels else None
        metric = observability_manager.metrics_collector.get_metric(metric_name, metric_labels)
        
        if not metric:
            raise HTTPException(status_code=404, detail="Metric not found")
        
        return {
            "name": metric.name,
            "type": metric.type.value,
            "value": metric.value,
            "labels": metric.labels,
            "timestamp": metric.timestamp,
            "help": metric.help
        }
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid labels JSON")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting metric: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get metric: {str(e)}")


@observability_router.get("/observability/metrics/{metric_name}/stats")
async def get_metric_stats_endpoint(metric_name: str):
    """Obtener estadísticas de métrica"""
    try:
        stats = observability_manager.metrics_collector.get_metric_stats(metric_name)
        return {"metric_name": metric_name, "stats": stats}
    except Exception as e:
        logger.error(f"Error getting metric stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get metric stats: {str(e)}")


@observability_router.get("/observability/alerts")
async def get_alerts_endpoint(status: Optional[str] = None):
    """Obtener alertas"""
    try:
        alerts = observability_manager.alert_manager.get_alerts(status)
        return {
            "alerts": [
                {
                    "id": alert.id,
                    "rule_id": alert.rule_id,
                    "name": alert.name,
                    "severity": alert.severity.value,
                    "status": alert.status,
                    "value": alert.value,
                    "threshold": alert.threshold,
                    "labels": alert.labels,
                    "annotations": alert.annotations,
                    "starts_at": alert.starts_at,
                    "ends_at": alert.ends_at,
                    "generator_url": alert.generator_url
                }
                for alert in alerts
            ]
        }
    except Exception as e:
        logger.error(f"Error getting alerts: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get alerts: {str(e)}")


@observability_router.get("/observability/alerts/rules")
async def get_alert_rules_endpoint():
    """Obtener reglas de alerta"""
    try:
        rules = observability_manager.alert_manager.get_rules()
        return {
            "rules": [
                {
                    "id": rule.id,
                    "name": rule.name,
                    "expression": rule.expression,
                    "severity": rule.severity.value,
                    "threshold": rule.threshold,
                    "duration": rule.duration,
                    "enabled": rule.enabled,
                    "labels": rule.labels,
                    "annotations": rule.annotations
                }
                for rule in rules
            ]
        }
    except Exception as e:
        logger.error(f"Error getting alert rules: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get alert rules: {str(e)}")


@observability_router.post("/observability/alerts/rules")
async def create_alert_rule_endpoint(rule_data: dict):
    """Crear regla de alerta"""
    try:
        rule = AlertRule(
            id=rule_data["id"],
            name=rule_data["name"],
            expression=rule_data["expression"],
            severity=AlertSeverity(rule_data["severity"]),
            threshold=rule_data["threshold"],
            duration=rule_data.get("duration", 300.0),
            enabled=rule_data.get("enabled", True),
            labels=rule_data.get("labels", {}),
            annotations=rule_data.get("annotations", {})
        )
        
        observability_manager.alert_manager.add_rule(rule)
        
        return {
            "message": "Alert rule created successfully",
            "rule_id": rule.id,
            "name": rule.name
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid severity: {e}")
    except Exception as e:
        logger.error(f"Error creating alert rule: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create alert rule: {str(e)}")


@observability_router.delete("/observability/alerts/rules/{rule_id}")
async def delete_alert_rule_endpoint(rule_id: str):
    """Eliminar regla de alerta"""
    try:
        observability_manager.alert_manager.remove_rule(rule_id)
        return {"message": "Alert rule deleted successfully", "rule_id": rule_id}
    except Exception as e:
        logger.error(f"Error deleting alert rule: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete alert rule: {str(e)}")


@observability_router.get("/observability/traces/{trace_id}")
async def get_trace_endpoint(trace_id: str):
    """Obtener trace"""
    try:
        spans = observability_manager.trace_collector.get_trace(trace_id)
        if not spans:
            raise HTTPException(status_code=404, detail="Trace not found")
        
        return {
            "trace_id": trace_id,
            "spans": [
                {
                    "span_id": span.span_id,
                    "parent_span_id": span.parent_span_id,
                    "operation_name": span.operation_name,
                    "start_time": span.start_time,
                    "end_time": span.end_time,
                    "duration": span.duration,
                    "tags": span.tags,
                    "logs": span.logs,
                    "status": span.status
                }
                for span in spans
            ]
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting trace: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get trace: {str(e)}")


@observability_router.get("/observability/traces/{trace_id}/stats")
async def get_trace_stats_endpoint(trace_id: str):
    """Obtener estadísticas de trace"""
    try:
        stats = observability_manager.trace_collector.get_trace_stats(trace_id)
        return {"trace_id": trace_id, "stats": stats}
    except Exception as e:
        logger.error(f"Error getting trace stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get trace stats: {str(e)}")


@observability_router.get("/observability/logs")
async def get_logs_endpoint(level: Optional[str] = None, source: Optional[str] = None, limit: int = 100):
    """Obtener logs"""
    try:
        log_level = LogLevel(level) if level else None
        logs = observability_manager.log_collector.get_logs(log_level, source, limit)
        
        return {
            "logs": [
                {
                    "timestamp": log.timestamp,
                    "level": log.level.value,
                    "message": log.message,
                    "source": log.source,
                    "trace_id": log.trace_id,
                    "span_id": log.span_id,
                    "labels": log.labels,
                    "fields": log.fields
                }
                for log in logs
            ]
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid log level: {e}")
    except Exception as e:
        logger.error(f"Error getting logs: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get logs: {str(e)}")


@observability_router.get("/observability/logs/stats")
async def get_log_stats_endpoint():
    """Obtener estadísticas de logs"""
    try:
        stats = observability_manager.log_collector.get_log_stats()
        return {"stats": stats}
    except Exception as e:
        logger.error(f"Error getting log stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get log stats: {str(e)}")


@observability_router.get("/observability/stats")
async def get_observability_stats_endpoint():
    """Obtener estadísticas de observabilidad"""
    try:
        stats = observability_manager.get_observability_stats()
        return stats
    except Exception as e:
        logger.error(f"Error getting observability stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get observability stats: {str(e)}")


# Funciones de utilidad para integración
async def start_observability():
    """Iniciar observabilidad"""
    await observability_manager.start()


async def stop_observability():
    """Detener observabilidad"""
    await observability_manager.stop()


def increment_counter(name: str, value: float = 1.0, labels: Dict[str, str] = None):
    """Incrementar contador"""
    observability_manager.metrics_collector.increment_counter(name, value, labels)


def set_gauge(name: str, value: float, labels: Dict[str, str] = None):
    """Establecer gauge"""
    observability_manager.metrics_collector.set_gauge(name, value, labels)


def observe_histogram(name: str, value: float, labels: Dict[str, str] = None):
    """Observar histograma"""
    observability_manager.metrics_collector.observe_histogram(name, value, labels)


def observe_summary(name: str, value: float, labels: Dict[str, str] = None):
    """Observar summary"""
    observability_manager.metrics_collector.observe_summary(name, value, labels)


def add_log(level: LogLevel, message: str, source: str = "system", **kwargs):
    """Agregar log"""
    observability_manager.log_collector.add_log(level, message, source, **kwargs)


def start_span(trace_id: str, span_id: str, operation_name: str, **kwargs) -> TraceSpan:
    """Iniciar span"""
    return observability_manager.trace_collector.start_span(trace_id, span_id, operation_name, **kwargs)


def finish_span(span_id: str, status: str = "ok", **kwargs):
    """Finalizar span"""
    observability_manager.trace_collector.finish_span(span_id, status, **kwargs)


logger.info("Advanced observability module loaded successfully")

