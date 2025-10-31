"""
Motor de Monitoreo AI
====================

Motor para monitoreo en tiempo real, alertas inteligentes y observabilidad del sistema.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
import psutil
import time
from pathlib import Path
import hashlib
import threading
from collections import deque

logger = logging.getLogger(__name__)

class MetricType(str, Enum):
    """Tipos de métricas"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"
    CUSTOM = "custom"

class AlertSeverity(str, Enum):
    """Severidad de alertas"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class AlertStatus(str, Enum):
    """Estado de alertas"""
    ACTIVE = "active"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"
    ACKNOWLEDGED = "acknowledged"

@dataclass
class Metric:
    """Métrica del sistema"""
    name: str
    value: float
    metric_type: MetricType
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    description: str = ""

@dataclass
class Alert:
    """Alerta del sistema"""
    id: str
    name: str
    description: str
    severity: AlertSeverity
    status: AlertStatus
    metric_name: str
    threshold: float
    current_value: float
    triggered_at: datetime
    resolved_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    labels: Dict[str, str] = field(default_factory=dict)

@dataclass
class AlertRule:
    """Regla de alerta"""
    id: str
    name: str
    description: str
    metric_name: str
    condition: str  # >, <, >=, <=, ==, !=
    threshold: float
    severity: AlertSeverity
    duration: int = 0  # Duración en segundos antes de activar
    enabled: bool = True
    labels: Dict[str, str] = field(default_factory=dict)

@dataclass
class SystemHealth:
    """Salud del sistema"""
    overall_status: str
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: Dict[str, float]
    active_connections: int
    response_time_avg: float
    error_rate: float
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class PerformanceMetrics:
    """Métricas de rendimiento"""
    request_count: int
    success_count: int
    error_count: int
    avg_response_time: float
    p95_response_time: float
    p99_response_time: float
    throughput: float
    error_rate: float
    timestamp: datetime = field(default_factory=datetime.now)

class AIMonitoringEngine:
    """Motor de monitoreo AI"""
    
    def __init__(self):
        self.metrics: Dict[str, deque] = {}
        self.alerts: Dict[str, Alert] = {}
        self.alert_rules: Dict[str, AlertRule] = {}
        self.system_health_history: deque = deque(maxlen=1000)
        self.performance_metrics_history: deque = deque(maxlen=1000)
        
        # Configuración
        self.metrics_retention_hours = 24
        self.alert_cooldown_minutes = 5
        self.health_check_interval = 30  # segundos
        self.metrics_collection_interval = 10  # segundos
        
        # Workers
        self.monitoring_active = False
        self.metrics_worker: Optional[asyncio.Task] = None
        self.health_worker: Optional[asyncio.Task] = None
        self.alert_worker: Optional[asyncio.Task] = None
        
        # Callbacks
        self.alert_callbacks: List[Callable] = []
        
    async def initialize(self):
        """Inicializa el motor de monitoreo"""
        logger.info("Inicializando motor de monitoreo AI...")
        
        # Cargar reglas de alerta predefinidas
        await self._load_default_alert_rules()
        
        # Cargar configuración existente
        await self._load_monitoring_config()
        
        # Iniciar workers de monitoreo
        await self._start_monitoring_workers()
        
        logger.info("Motor de monitoreo AI inicializado")
    
    async def _load_default_alert_rules(self):
        """Carga reglas de alerta por defecto"""
        try:
            # Regla de CPU alta
            cpu_rule = AlertRule(
                id="high_cpu_usage",
                name="Alto Uso de CPU",
                description="CPU usage above 80%",
                metric_name="system.cpu.usage",
                condition=">",
                threshold=80.0,
                severity=AlertSeverity.WARNING,
                duration=60  # 1 minuto
            )
            
            # Regla de memoria alta
            memory_rule = AlertRule(
                id="high_memory_usage",
                name="Alto Uso de Memoria",
                description="Memory usage above 85%",
                metric_name="system.memory.usage",
                condition=">",
                threshold=85.0,
                severity=AlertSeverity.WARNING,
                duration=60
            )
            
            # Regla de disco lleno
            disk_rule = AlertRule(
                id="high_disk_usage",
                name="Alto Uso de Disco",
                description="Disk usage above 90%",
                metric_name="system.disk.usage",
                condition=">",
                threshold=90.0,
                severity=AlertSeverity.ERROR,
                duration=0
            )
            
            # Regla de tiempo de respuesta alto
            response_time_rule = AlertRule(
                id="high_response_time",
                name="Tiempo de Respuesta Alto",
                description="Average response time above 5 seconds",
                metric_name="api.response_time.avg",
                condition=">",
                threshold=5.0,
                severity=AlertSeverity.WARNING,
                duration=120  # 2 minutos
            )
            
            # Regla de tasa de error alta
            error_rate_rule = AlertRule(
                id="high_error_rate",
                name="Alta Tasa de Error",
                description="Error rate above 5%",
                metric_name="api.error_rate",
                condition=">",
                threshold=5.0,
                severity=AlertSeverity.ERROR,
                duration=60
            )
            
            # Regla de conexiones activas
            connections_rule = AlertRule(
                id="high_active_connections",
                name="Muchas Conexiones Activas",
                description="Active connections above 1000",
                metric_name="system.connections.active",
                condition=">",
                threshold=1000.0,
                severity=AlertSeverity.WARNING,
                duration=60
            )
            
            # Guardar reglas
            self.alert_rules["high_cpu_usage"] = cpu_rule
            self.alert_rules["high_memory_usage"] = memory_rule
            self.alert_rules["high_disk_usage"] = disk_rule
            self.alert_rules["high_response_time"] = response_time_rule
            self.alert_rules["high_error_rate"] = error_rate_rule
            self.alert_rules["high_active_connections"] = connections_rule
            
            logger.info(f"Cargadas {len(self.alert_rules)} reglas de alerta por defecto")
            
        except Exception as e:
            logger.error(f"Error cargando reglas de alerta por defecto: {e}")
    
    async def _load_monitoring_config(self):
        """Carga configuración de monitoreo"""
        try:
            config_file = Path("data/monitoring_config.json")
            if config_file.exists():
                with open(config_file, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                
                # Cargar reglas de alerta personalizadas
                for rule_data in config_data.get("alert_rules", []):
                    rule = AlertRule(
                        id=rule_data["id"],
                        name=rule_data["name"],
                        description=rule_data["description"],
                        metric_name=rule_data["metric_name"],
                        condition=rule_data["condition"],
                        threshold=rule_data["threshold"],
                        severity=AlertSeverity(rule_data["severity"]),
                        duration=rule_data.get("duration", 0),
                        enabled=rule_data.get("enabled", True),
                        labels=rule_data.get("labels", {})
                    )
                    self.alert_rules[rule.id] = rule
                
                logger.info("Configuración de monitoreo cargada")
            
        except Exception as e:
            logger.error(f"Error cargando configuración de monitoreo: {e}")
    
    async def _start_monitoring_workers(self):
        """Inicia workers de monitoreo"""
        try:
            self.monitoring_active = True
            
            # Worker de recolección de métricas
            self.metrics_worker = asyncio.create_task(self._metrics_collection_worker())
            
            # Worker de salud del sistema
            self.health_worker = asyncio.create_task(self._health_check_worker())
            
            # Worker de alertas
            self.alert_worker = asyncio.create_task(self._alert_evaluation_worker())
            
            logger.info("Workers de monitoreo iniciados")
            
        except Exception as e:
            logger.error(f"Error iniciando workers de monitoreo: {e}")
    
    async def _metrics_collection_worker(self):
        """Worker de recolección de métricas"""
        while self.monitoring_active:
            try:
                await self._collect_system_metrics()
                await self._collect_application_metrics()
                await asyncio.sleep(self.metrics_collection_interval)
                
            except Exception as e:
                logger.error(f"Error en worker de recolección de métricas: {e}")
                await asyncio.sleep(10)
    
    async def _health_check_worker(self):
        """Worker de verificación de salud"""
        while self.monitoring_active:
            try:
                health = await self._check_system_health()
                self.system_health_history.append(health)
                await asyncio.sleep(self.health_check_interval)
                
            except Exception as e:
                logger.error(f"Error en worker de verificación de salud: {e}")
                await asyncio.sleep(10)
    
    async def _alert_evaluation_worker(self):
        """Worker de evaluación de alertas"""
        while self.monitoring_active:
            try:
                await self._evaluate_alert_rules()
                await asyncio.sleep(30)  # Evaluar cada 30 segundos
                
            except Exception as e:
                logger.error(f"Error en worker de evaluación de alertas: {e}")
                await asyncio.sleep(10)
    
    async def _collect_system_metrics(self):
        """Recolecta métricas del sistema"""
        try:
            # CPU
            cpu_percent = psutil.cpu_percent(interval=1)
            await self._record_metric("system.cpu.usage", cpu_percent, MetricType.GAUGE)
            
            # Memoria
            memory = psutil.virtual_memory()
            await self._record_metric("system.memory.usage", memory.percent, MetricType.GAUGE)
            await self._record_metric("system.memory.available", memory.available, MetricType.GAUGE)
            await self._record_metric("system.memory.total", memory.total, MetricType.GAUGE)
            
            # Disco
            disk = psutil.disk_usage('/')
            await self._record_metric("system.disk.usage", (disk.used / disk.total) * 100, MetricType.GAUGE)
            await self._record_metric("system.disk.free", disk.free, MetricType.GAUGE)
            await self._record_metric("system.disk.total", disk.total, MetricType.GAUGE)
            
            # Red
            network = psutil.net_io_counters()
            await self._record_metric("system.network.bytes_sent", network.bytes_sent, MetricType.COUNTER)
            await self._record_metric("system.network.bytes_recv", network.bytes_recv, MetricType.COUNTER)
            await self._record_metric("system.network.packets_sent", network.packets_sent, MetricType.COUNTER)
            await self._record_metric("system.network.packets_recv", network.packets_recv, MetricType.COUNTER)
            
            # Procesos
            process_count = len(psutil.pids())
            await self._record_metric("system.processes.count", process_count, MetricType.GAUGE)
            
            # Carga del sistema
            load_avg = psutil.getloadavg()
            await self._record_metric("system.load.1m", load_avg[0], MetricType.GAUGE)
            await self._record_metric("system.load.5m", load_avg[1], MetricType.GAUGE)
            await self._record_metric("system.load.15m", load_avg[2], MetricType.GAUGE)
            
        except Exception as e:
            logger.error(f"Error recolectando métricas del sistema: {e}")
    
    async def _collect_application_metrics(self):
        """Recolecta métricas de la aplicación"""
        try:
            # Métricas de la aplicación (simuladas)
            await self._record_metric("app.requests.total", 100, MetricType.COUNTER)
            await self._record_metric("app.requests.success", 95, MetricType.COUNTER)
            await self._record_metric("app.requests.error", 5, MetricType.COUNTER)
            await self._record_metric("app.response_time.avg", 1.5, MetricType.GAUGE)
            await self._record_metric("app.response_time.p95", 3.2, MetricType.GAUGE)
            await self._record_metric("app.response_time.p99", 5.8, MetricType.GAUGE)
            await self._record_metric("app.active_connections", 25, MetricType.GAUGE)
            await self._record_metric("app.documents_processed", 150, MetricType.COUNTER)
            await self._record_metric("app.cache_hit_rate", 85.5, MetricType.GAUGE)
            
        except Exception as e:
            logger.error(f"Error recolectando métricas de la aplicación: {e}")
    
    async def _record_metric(self, name: str, value: float, metric_type: MetricType, labels: Dict[str, str] = None):
        """Registra una métrica"""
        try:
            metric = Metric(
                name=name,
                value=value,
                metric_type=metric_type,
                labels=labels or {},
                timestamp=datetime.now()
            )
            
            if name not in self.metrics:
                self.metrics[name] = deque(maxlen=1000)
            
            self.metrics[name].append(metric)
            
        except Exception as e:
            logger.error(f"Error registrando métrica {name}: {e}")
    
    async def _check_system_health(self) -> SystemHealth:
        """Verifica salud del sistema"""
        try:
            # Obtener métricas actuales
            cpu_usage = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            network = psutil.net_io_counters()
            
            # Calcular estado general
            overall_status = "healthy"
            if cpu_usage > 80 or memory.percent > 85 or (disk.used / disk.total) > 0.9:
                overall_status = "unhealthy"
            elif cpu_usage > 60 or memory.percent > 70 or (disk.used / disk.total) > 0.8:
                overall_status = "warning"
            
            # Calcular métricas de red
            network_io = {
                "bytes_sent": network.bytes_sent,
                "bytes_recv": network.bytes_recv,
                "packets_sent": network.packets_sent,
                "packets_recv": network.packets_recv
            }
            
            # Métricas de aplicación (simuladas)
            active_connections = 25
            response_time_avg = 1.5
            error_rate = 2.5
            
            health = SystemHealth(
                overall_status=overall_status,
                cpu_usage=cpu_usage,
                memory_usage=memory.percent,
                disk_usage=(disk.used / disk.total) * 100,
                network_io=network_io,
                active_connections=active_connections,
                response_time_avg=response_time_avg,
                error_rate=error_rate
            )
            
            return health
            
        except Exception as e:
            logger.error(f"Error verificando salud del sistema: {e}")
            return SystemHealth(
                overall_status="unknown",
                cpu_usage=0.0,
                memory_usage=0.0,
                disk_usage=0.0,
                network_io={},
                active_connections=0,
                response_time_avg=0.0,
                error_rate=0.0
            )
    
    async def _evaluate_alert_rules(self):
        """Evalúa reglas de alerta"""
        try:
            for rule in self.alert_rules.values():
                if not rule.enabled:
                    continue
                
                # Obtener valor actual de la métrica
                current_value = await self._get_metric_value(rule.metric_name)
                if current_value is None:
                    continue
                
                # Verificar condición
                condition_met = self._evaluate_condition(current_value, rule.condition, rule.threshold)
                
                if condition_met:
                    # Verificar si ya existe una alerta activa
                    existing_alert = self._get_active_alert(rule.id)
                    
                    if existing_alert is None:
                        # Crear nueva alerta
                        await self._create_alert(rule, current_value)
                    else:
                        # Actualizar alerta existente
                        existing_alert.current_value = current_value
                else:
                    # Resolver alerta si existe
                    existing_alert = self._get_active_alert(rule.id)
                    if existing_alert:
                        await self._resolve_alert(existing_alert)
            
        except Exception as e:
            logger.error(f"Error evaluando reglas de alerta: {e}")
    
    async def _get_metric_value(self, metric_name: str) -> Optional[float]:
        """Obtiene valor actual de una métrica"""
        try:
            if metric_name not in self.metrics or not self.metrics[metric_name]:
                return None
            
            # Obtener el valor más reciente
            latest_metric = self.metrics[metric_name][-1]
            return latest_metric.value
            
        except Exception as e:
            logger.error(f"Error obteniendo valor de métrica {metric_name}: {e}")
            return None
    
    def _evaluate_condition(self, value: float, condition: str, threshold: float) -> bool:
        """Evalúa condición de alerta"""
        try:
            if condition == ">":
                return value > threshold
            elif condition == "<":
                return value < threshold
            elif condition == ">=":
                return value >= threshold
            elif condition == "<=":
                return value <= threshold
            elif condition == "==":
                return value == threshold
            elif condition == "!=":
                return value != threshold
            else:
                logger.warning(f"Condición no soportada: {condition}")
                return False
                
        except Exception as e:
            logger.error(f"Error evaluando condición: {e}")
            return False
    
    def _get_active_alert(self, rule_id: str) -> Optional[Alert]:
        """Obtiene alerta activa para una regla"""
        try:
            for alert in self.alerts.values():
                if alert.metric_name == rule_id and alert.status == AlertStatus.ACTIVE:
                    return alert
            return None
            
        except Exception as e:
            logger.error(f"Error obteniendo alerta activa: {e}")
            return None
    
    async def _create_alert(self, rule: AlertRule, current_value: float):
        """Crea nueva alerta"""
        try:
            alert_id = f"alert_{rule.id}_{int(time.time())}"
            
            alert = Alert(
                id=alert_id,
                name=rule.name,
                description=rule.description,
                severity=rule.severity,
                status=AlertStatus.ACTIVE,
                metric_name=rule.metric_name,
                threshold=rule.threshold,
                current_value=current_value,
                triggered_at=datetime.now(),
                labels=rule.labels
            )
            
            self.alerts[alert_id] = alert
            
            # Notificar alerta
            await self._notify_alert(alert)
            
            logger.warning(f"Alerta creada: {alert.name} - {alert.description}")
            
        except Exception as e:
            logger.error(f"Error creando alerta: {e}")
    
    async def _resolve_alert(self, alert: Alert):
        """Resuelve alerta"""
        try:
            alert.status = AlertStatus.RESOLVED
            alert.resolved_at = datetime.now()
            
            # Notificar resolución
            await self._notify_alert_resolution(alert)
            
            logger.info(f"Alerta resuelta: {alert.name}")
            
        except Exception as e:
            logger.error(f"Error resolviendo alerta: {e}")
    
    async def _notify_alert(self, alert: Alert):
        """Notifica alerta"""
        try:
            # Ejecutar callbacks registrados
            for callback in self.alert_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(alert)
                    else:
                        callback(alert)
                except Exception as e:
                    logger.error(f"Error en callback de alerta: {e}")
            
            # En implementación real, enviar notificaciones (email, Slack, etc.)
            logger.warning(f"ALERTA [{alert.severity.value.upper()}]: {alert.name} - {alert.description}")
            
        except Exception as e:
            logger.error(f"Error notificando alerta: {e}")
    
    async def _notify_alert_resolution(self, alert: Alert):
        """Notifica resolución de alerta"""
        try:
            logger.info(f"ALERTA RESUELTA: {alert.name}")
            
        except Exception as e:
            logger.error(f"Error notificando resolución de alerta: {e}")
    
    async def create_alert_rule(
        self,
        name: str,
        description: str,
        metric_name: str,
        condition: str,
        threshold: float,
        severity: AlertSeverity,
        duration: int = 0,
        labels: Dict[str, str] = None
    ) -> str:
        """Crea nueva regla de alerta"""
        try:
            rule_id = f"rule_{int(time.time())}"
            
            rule = AlertRule(
                id=rule_id,
                name=name,
                description=description,
                metric_name=metric_name,
                condition=condition,
                threshold=threshold,
                severity=severity,
                duration=duration,
                enabled=True,
                labels=labels or {}
            )
            
            self.alert_rules[rule_id] = rule
            
            # Guardar configuración
            await self._save_monitoring_config()
            
            logger.info(f"Regla de alerta creada: {rule_id}")
            return rule_id
            
        except Exception as e:
            logger.error(f"Error creando regla de alerta: {e}")
            raise
    
    async def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """Reconoce alerta"""
        try:
            if alert_id not in self.alerts:
                return False
            
            alert = self.alerts[alert_id]
            alert.status = AlertStatus.ACKNOWLEDGED
            alert.acknowledged_by = acknowledged_by
            alert.acknowledged_at = datetime.now()
            
            logger.info(f"Alerta reconocida: {alert_id} por {acknowledged_by}")
            return True
            
        except Exception as e:
            logger.error(f"Error reconociendo alerta: {e}")
            return False
    
    async def suppress_alert(self, alert_id: str, suppressed_by: str) -> bool:
        """Suprime alerta"""
        try:
            if alert_id not in self.alerts:
                return False
            
            alert = self.alerts[alert_id]
            alert.status = AlertStatus.SUPPRESSED
            
            logger.info(f"Alerta suprimida: {alert_id} por {suppressed_by}")
            return True
            
        except Exception as e:
            logger.error(f"Error suprimiendo alerta: {e}")
            return False
    
    def register_alert_callback(self, callback: Callable):
        """Registra callback para alertas"""
        try:
            self.alert_callbacks.append(callback)
            logger.info("Callback de alerta registrado")
            
        except Exception as e:
            logger.error(f"Error registrando callback de alerta: {e}")
    
    async def get_metrics(self, metric_name: str, time_range: int = 3600) -> List[Dict[str, Any]]:
        """Obtiene métricas"""
        try:
            if metric_name not in self.metrics:
                return []
            
            cutoff_time = datetime.now() - timedelta(seconds=time_range)
            metrics = []
            
            for metric in self.metrics[metric_name]:
                if metric.timestamp >= cutoff_time:
                    metrics.append({
                        "name": metric.name,
                        "value": metric.value,
                        "type": metric.metric_type.value,
                        "labels": metric.labels,
                        "timestamp": metric.timestamp.isoformat()
                    })
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error obteniendo métricas: {e}")
            return []
    
    async def get_system_health(self) -> Dict[str, Any]:
        """Obtiene salud del sistema"""
        try:
            if not self.system_health_history:
                return {"status": "no_data"}
            
            latest_health = self.system_health_history[-1]
            
            return {
                "overall_status": latest_health.overall_status,
                "cpu_usage": latest_health.cpu_usage,
                "memory_usage": latest_health.memory_usage,
                "disk_usage": latest_health.disk_usage,
                "network_io": latest_health.network_io,
                "active_connections": latest_health.active_connections,
                "response_time_avg": latest_health.response_time_avg,
                "error_rate": latest_health.error_rate,
                "timestamp": latest_health.timestamp.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error obteniendo salud del sistema: {e}")
            return {"error": str(e)}
    
    async def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Obtiene alertas activas"""
        try:
            active_alerts = []
            
            for alert in self.alerts.values():
                if alert.status == AlertStatus.ACTIVE:
                    active_alerts.append({
                        "id": alert.id,
                        "name": alert.name,
                        "description": alert.description,
                        "severity": alert.severity.value,
                        "metric_name": alert.metric_name,
                        "threshold": alert.threshold,
                        "current_value": alert.current_value,
                        "triggered_at": alert.triggered_at.isoformat(),
                        "labels": alert.labels
                    })
            
            return active_alerts
            
        except Exception as e:
            logger.error(f"Error obteniendo alertas activas: {e}")
            return []
    
    async def get_alert_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Obtiene historial de alertas"""
        try:
            alerts = list(self.alerts.values())
            alerts.sort(key=lambda x: x.triggered_at, reverse=True)
            
            return [
                {
                    "id": alert.id,
                    "name": alert.name,
                    "description": alert.description,
                    "severity": alert.severity.value,
                    "status": alert.status.value,
                    "metric_name": alert.metric_name,
                    "threshold": alert.threshold,
                    "current_value": alert.current_value,
                    "triggered_at": alert.triggered_at.isoformat(),
                    "resolved_at": alert.resolved_at.isoformat() if alert.resolved_at else None,
                    "acknowledged_by": alert.acknowledged_by,
                    "acknowledged_at": alert.acknowledged_at.isoformat() if alert.acknowledged_at else None,
                    "labels": alert.labels
                }
                for alert in alerts[:limit]
            ]
            
        except Exception as e:
            logger.error(f"Error obteniendo historial de alertas: {e}")
            return []
    
    async def get_monitoring_dashboard_data(self) -> Dict[str, Any]:
        """Obtiene datos para dashboard de monitoreo"""
        try:
            # Salud del sistema actual
            current_health = await self.get_system_health()
            
            # Alertas activas
            active_alerts = await self.get_active_alerts()
            
            # Métricas clave
            key_metrics = {}
            for metric_name in ["system.cpu.usage", "system.memory.usage", "system.disk.usage", 
                              "app.response_time.avg", "app.error_rate", "app.active_connections"]:
                metrics = await self.get_metrics(metric_name, 3600)  # Última hora
                if metrics:
                    key_metrics[metric_name] = {
                        "current": metrics[-1]["value"],
                        "avg": sum(m["value"] for m in metrics) / len(metrics),
                        "min": min(m["value"] for m in metrics),
                        "max": max(m["value"] for m in metrics)
                    }
            
            # Estadísticas de alertas
            total_alerts = len(self.alerts)
            active_count = len(active_alerts)
            resolved_count = len([a for a in self.alerts.values() if a.status == AlertStatus.RESOLVED])
            
            # Distribución por severidad
            severity_distribution = {}
            for alert in self.alerts.values():
                severity = alert.severity.value
                severity_distribution[severity] = severity_distribution.get(severity, 0) + 1
            
            return {
                "system_health": current_health,
                "active_alerts": active_alerts,
                "key_metrics": key_metrics,
                "alert_statistics": {
                    "total": total_alerts,
                    "active": active_count,
                    "resolved": resolved_count,
                    "severity_distribution": severity_distribution
                },
                "monitoring_status": {
                    "active": self.monitoring_active,
                    "workers_running": len([w for w in [self.metrics_worker, self.health_worker, self.alert_worker] if w and not w.done()]),
                    "metrics_collected": sum(len(metrics) for metrics in self.metrics.values()),
                    "alert_rules": len(self.alert_rules)
                },
                "last_updated": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error obteniendo datos del dashboard de monitoreo: {e}")
            return {"error": str(e)}
    
    async def _save_monitoring_config(self):
        """Guarda configuración de monitoreo"""
        try:
            # Crear directorio de datos
            data_dir = Path("data")
            data_dir.mkdir(exist_ok=True)
            
            # Convertir reglas a formato serializable
            rules_data = []
            for rule in self.alert_rules.values():
                rules_data.append({
                    "id": rule.id,
                    "name": rule.name,
                    "description": rule.description,
                    "metric_name": rule.metric_name,
                    "condition": rule.condition,
                    "threshold": rule.threshold,
                    "severity": rule.severity.value,
                    "duration": rule.duration,
                    "enabled": rule.enabled,
                    "labels": rule.labels
                })
            
            config_data = {
                "alert_rules": rules_data,
                "last_updated": datetime.now().isoformat()
            }
            
            # Guardar archivo
            config_file = data_dir / "monitoring_config.json"
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False, default=str)
            
        except Exception as e:
            logger.error(f"Error guardando configuración de monitoreo: {e}")
    
    async def stop_monitoring(self):
        """Detiene monitoreo"""
        try:
            self.monitoring_active = False
            
            # Cancelar workers
            if self.metrics_worker:
                self.metrics_worker.cancel()
            if self.health_worker:
                self.health_worker.cancel()
            if self.alert_worker:
                self.alert_worker.cancel()
            
            logger.info("Monitoreo detenido")
            
        except Exception as e:
            logger.error(f"Error deteniendo monitoreo: {e}")
    
    async def get_metric_names(self) -> List[str]:
        """Obtiene nombres de métricas disponibles"""
        try:
            return list(self.metrics.keys())
        except Exception as e:
            logger.error(f"Error obteniendo nombres de métricas: {e}")
            return []
    
    async def get_alert_rules(self) -> List[Dict[str, Any]]:
        """Obtiene reglas de alerta"""
        try:
            return [
                {
                    "id": rule.id,
                    "name": rule.name,
                    "description": rule.description,
                    "metric_name": rule.metric_name,
                    "condition": rule.condition,
                    "threshold": rule.threshold,
                    "severity": rule.severity.value,
                    "duration": rule.duration,
                    "enabled": rule.enabled,
                    "labels": rule.labels
                }
                for rule in self.alert_rules.values()
            ]
        except Exception as e:
            logger.error(f"Error obteniendo reglas de alerta: {e}")
            return []

