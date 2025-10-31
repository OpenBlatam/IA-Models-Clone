"""
Advanced Monitoring and Alerting System for AI History Comparison
Sistema avanzado de monitoreo y alertas para análisis de historial de IA
"""

import asyncio
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import psutil
import time
from collections import deque, defaultdict
import threading
import queue

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AlertLevel(Enum):
    """Niveles de alerta"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class MetricType(Enum):
    """Tipos de métricas"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"
    CUSTOM = "custom"

class AlertCondition(Enum):
    """Condiciones de alerta"""
    GREATER_THAN = "greater_than"
    LESS_THAN = "less_than"
    EQUALS = "equals"
    NOT_EQUALS = "not_equals"
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"
    RATE_CHANGE = "rate_change"
    ANOMALY = "anomaly"

@dataclass
class Metric:
    """Métrica del sistema"""
    name: str
    value: float
    metric_type: MetricType
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    unit: Optional[str] = None

@dataclass
class AlertRule:
    """Regla de alerta"""
    id: str
    name: str
    metric_name: str
    condition: AlertCondition
    threshold: float
    alert_level: AlertLevel
    enabled: bool = True
    cooldown_period: int = 300  # seconds
    last_triggered: Optional[datetime] = None
    description: str = ""
    tags: Dict[str, str] = field(default_factory=dict)

@dataclass
class Alert:
    """Alerta generada"""
    id: str
    rule_id: str
    metric_name: str
    alert_level: AlertLevel
    message: str
    value: float
    threshold: float
    timestamp: datetime
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    tags: Dict[str, str] = field(default_factory=dict)

@dataclass
class SystemHealth:
    """Estado de salud del sistema"""
    overall_status: str
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: Dict[str, float]
    active_alerts: int
    total_metrics: int
    uptime: float
    timestamp: datetime = field(default_factory=datetime.now)

class AdvancedMonitoringSystem:
    """
    Sistema avanzado de monitoreo y alertas
    """
    
    def __init__(
        self,
        enable_system_metrics: bool = True,
        enable_application_metrics: bool = True,
        enable_alerting: bool = True,
        metrics_retention_days: int = 30
    ):
        self.enable_system_metrics = enable_system_metrics
        self.enable_application_metrics = enable_application_metrics
        self.enable_alerting = enable_alerting
        self.metrics_retention_days = metrics_retention_days
        
        # Almacenamiento de métricas
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.current_metrics: Dict[str, float] = {}
        
        # Reglas de alerta
        self.alert_rules: Dict[str, AlertRule] = {}
        
        # Alertas activas
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: deque = deque(maxlen=1000)
        
        # Callbacks de alerta
        self.alert_callbacks: List[Callable] = []
        
        # Estado del sistema
        self.system_start_time = datetime.now()
        self.monitoring_active = False
        
        # Configuración
        self.config = {
            "metrics_collection_interval": 10,  # seconds
            "alert_evaluation_interval": 5,  # seconds
            "health_check_interval": 30,  # seconds
            "max_metrics_per_name": 10000,
            "anomaly_detection_window": 100,
            "anomaly_threshold": 2.0  # standard deviations
        }
        
        # Inicializar reglas de alerta por defecto
        self._initialize_default_alert_rules()
        
        # Threads de monitoreo
        self.monitoring_threads: List[threading.Thread] = []
        self.stop_event = threading.Event()
    
    def _initialize_default_alert_rules(self):
        """Inicializar reglas de alerta por defecto"""
        
        # Reglas de sistema
        cpu_rule = AlertRule(
            id="high_cpu_usage",
            name="Alto uso de CPU",
            metric_name="system.cpu_percent",
            condition=AlertCondition.GREATER_THAN,
            threshold=80.0,
            alert_level=AlertLevel.WARNING,
            description="El uso de CPU ha excedido el 80%"
        )
        
        memory_rule = AlertRule(
            id="high_memory_usage",
            name="Alto uso de memoria",
            metric_name="system.memory_percent",
            condition=AlertCondition.GREATER_THAN,
            threshold=85.0,
            alert_level=AlertLevel.WARNING,
            description="El uso de memoria ha excedido el 85%"
        )
        
        disk_rule = AlertRule(
            id="high_disk_usage",
            name="Alto uso de disco",
            metric_name="system.disk_percent",
            condition=AlertCondition.GREATER_THAN,
            threshold=90.0,
            alert_level=AlertLevel.ERROR,
            description="El uso de disco ha excedido el 90%"
        )
        
        # Reglas de aplicación
        error_rate_rule = AlertRule(
            id="high_error_rate",
            name="Alta tasa de errores",
            metric_name="application.error_rate",
            condition=AlertCondition.GREATER_THAN,
            threshold=5.0,
            alert_level=AlertLevel.ERROR,
            description="La tasa de errores ha excedido el 5%"
        )
        
        response_time_rule = AlertRule(
            id="high_response_time",
            name="Tiempo de respuesta alto",
            metric_name="application.response_time",
            condition=AlertCondition.GREATER_THAN,
            threshold=5.0,
            alert_level=AlertLevel.WARNING,
            description="El tiempo de respuesta ha excedido 5 segundos"
        )
        
        self.alert_rules = {
            "high_cpu_usage": cpu_rule,
            "high_memory_usage": memory_rule,
            "high_disk_usage": disk_rule,
            "high_error_rate": error_rate_rule,
            "high_response_time": response_time_rule
        }
    
    async def start_monitoring(self):
        """Iniciar monitoreo del sistema"""
        if self.monitoring_active:
            logger.warning("Monitoring is already active")
            return
        
        self.monitoring_active = True
        self.stop_event.clear()
        
        logger.info("Starting advanced monitoring system")
        
        # Iniciar threads de monitoreo
        if self.enable_system_metrics:
            system_thread = threading.Thread(target=self._system_metrics_collector)
            system_thread.daemon = True
            system_thread.start()
            self.monitoring_threads.append(system_thread)
        
        if self.enable_alerting:
            alerting_thread = threading.Thread(target=self._alert_evaluator)
            alerting_thread.daemon = True
            alerting_thread.start()
            self.monitoring_threads.append(alerting_thread)
        
        health_thread = threading.Thread(target=self._health_checker)
        health_thread.daemon = True
        health_thread.start()
        self.monitoring_threads.append(health_thread)
        
        logger.info("Monitoring system started successfully")
    
    async def stop_monitoring(self):
        """Detener monitoreo del sistema"""
        if not self.monitoring_active:
            return
        
        logger.info("Stopping monitoring system")
        
        self.monitoring_active = False
        self.stop_event.set()
        
        # Esperar a que terminen los threads
        for thread in self.monitoring_threads:
            thread.join(timeout=5)
        
        self.monitoring_threads.clear()
        
        logger.info("Monitoring system stopped")
    
    def _system_metrics_collector(self):
        """Recolector de métricas del sistema"""
        while not self.stop_event.is_set():
            try:
                # Métricas de CPU
                cpu_percent = psutil.cpu_percent(interval=1)
                self._record_metric("system.cpu_percent", cpu_percent, MetricType.GAUGE)
                
                # Métricas de memoria
                memory = psutil.virtual_memory()
                self._record_metric("system.memory_percent", memory.percent, MetricType.GAUGE)
                self._record_metric("system.memory_available", memory.available, MetricType.GAUGE)
                self._record_metric("system.memory_used", memory.used, MetricType.GAUGE)
                
                # Métricas de disco
                disk = psutil.disk_usage('/')
                disk_percent = (disk.used / disk.total) * 100
                self._record_metric("system.disk_percent", disk_percent, MetricType.GAUGE)
                self._record_metric("system.disk_free", disk.free, MetricType.GAUGE)
                self._record_metric("system.disk_used", disk.used, MetricType.GAUGE)
                
                # Métricas de red
                network = psutil.net_io_counters()
                self._record_metric("system.network_bytes_sent", network.bytes_sent, MetricType.COUNTER)
                self._record_metric("system.network_bytes_recv", network.bytes_recv, MetricType.COUNTER)
                self._record_metric("system.network_packets_sent", network.packets_sent, MetricType.COUNTER)
                self._record_metric("system.network_packets_recv", network.packets_recv, MetricType.COUNTER)
                
                # Métricas de procesos
                process = psutil.Process()
                self._record_metric("system.process_cpu_percent", process.cpu_percent(), MetricType.GAUGE)
                self._record_metric("system.process_memory_percent", process.memory_percent(), MetricType.GAUGE)
                self._record_metric("system.process_memory_rss", process.memory_info().rss, MetricType.GAUGE)
                self._record_metric("system.process_memory_vms", process.memory_info().vms, MetricType.GAUGE)
                
                time.sleep(self.config["metrics_collection_interval"])
                
            except Exception as e:
                logger.error(f"Error in system metrics collector: {e}")
                time.sleep(5)
    
    def _alert_evaluator(self):
        """Evaluador de alertas"""
        while not self.stop_event.is_set():
            try:
                for rule_id, rule in self.alert_rules.items():
                    if not rule.enabled:
                        continue
                    
                    # Verificar cooldown
                    if (rule.last_triggered and 
                        (datetime.now() - rule.last_triggered).total_seconds() < rule.cooldown_period):
                        continue
                    
                    # Evaluar regla
                    if self._evaluate_alert_rule(rule):
                        self._trigger_alert(rule)
                
                time.sleep(self.config["alert_evaluation_interval"])
                
            except Exception as e:
                logger.error(f"Error in alert evaluator: {e}")
                time.sleep(5)
    
    def _health_checker(self):
        """Verificador de salud del sistema"""
        while not self.stop_event.is_set():
            try:
                # Verificar métricas críticas
                health_status = self._check_system_health()
                
                # Registrar métrica de salud
                self._record_metric("system.health_score", health_status, MetricType.GAUGE)
                
                time.sleep(self.config["health_check_interval"])
                
            except Exception as e:
                logger.error(f"Error in health checker: {e}")
                time.sleep(10)
    
    def _record_metric(
        self,
        name: str,
        value: float,
        metric_type: MetricType,
        tags: Optional[Dict[str, str]] = None,
        unit: Optional[str] = None
    ):
        """Registrar una métrica"""
        try:
            metric = Metric(
                name=name,
                value=value,
                metric_type=metric_type,
                timestamp=datetime.now(),
                tags=tags or {},
                unit=unit
            )
            
            self.metrics[name].append(metric)
            self.current_metrics[name] = value
            
        except Exception as e:
            logger.error(f"Error recording metric {name}: {e}")
    
    def _evaluate_alert_rule(self, rule: AlertRule) -> bool:
        """Evaluar una regla de alerta"""
        try:
            if rule.metric_name not in self.current_metrics:
                return False
            
            current_value = self.current_metrics[rule.metric_name]
            
            if rule.condition == AlertCondition.GREATER_THAN:
                return current_value > rule.threshold
            elif rule.condition == AlertCondition.LESS_THAN:
                return current_value < rule.threshold
            elif rule.condition == AlertCondition.EQUALS:
                return current_value == rule.threshold
            elif rule.condition == AlertCondition.NOT_EQUALS:
                return current_value != rule.threshold
            elif rule.condition == AlertCondition.RATE_CHANGE:
                return self._check_rate_change(rule.metric_name, rule.threshold)
            elif rule.condition == AlertCondition.ANOMALY:
                return self._check_anomaly(rule.metric_name, rule.threshold)
            else:
                return False
                
        except Exception as e:
            logger.error(f"Error evaluating alert rule {rule.id}: {e}")
            return False
    
    def _check_rate_change(self, metric_name: str, threshold: float) -> bool:
        """Verificar cambio de tasa"""
        try:
            if metric_name not in self.metrics or len(self.metrics[metric_name]) < 2:
                return False
            
            recent_metrics = list(self.metrics[metric_name])[-10:]  # Últimos 10 valores
            
            if len(recent_metrics) < 2:
                return False
            
            # Calcular tasa de cambio
            old_value = recent_metrics[0].value
            new_value = recent_metrics[-1].value
            
            if old_value == 0:
                return new_value > threshold
            
            rate_change = abs((new_value - old_value) / old_value) * 100
            return rate_change > threshold
            
        except Exception as e:
            logger.error(f"Error checking rate change for {metric_name}: {e}")
            return False
    
    def _check_anomaly(self, metric_name: str, threshold: float) -> bool:
        """Verificar anomalía"""
        try:
            if metric_name not in self.metrics or len(self.metrics[metric_name]) < 10:
                return False
            
            recent_metrics = list(self.metrics[metric_name])[-self.config["anomaly_detection_window"]:]
            values = [m.value for m in recent_metrics]
            
            if len(values) < 10:
                return False
            
            # Calcular z-score
            mean_value = np.mean(values[:-1])  # Excluir el último valor
            std_value = np.std(values[:-1])
            
            if std_value == 0:
                return False
            
            current_value = values[-1]
            z_score = abs((current_value - mean_value) / std_value)
            
            return z_score > threshold
            
        except Exception as e:
            logger.error(f"Error checking anomaly for {metric_name}: {e}")
            return False
    
    def _trigger_alert(self, rule: AlertRule):
        """Disparar una alerta"""
        try:
            current_value = self.current_metrics.get(rule.metric_name, 0)
            
            alert = Alert(
                id=f"alert_{rule.id}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
                rule_id=rule.id,
                metric_name=rule.metric_name,
                alert_level=rule.alert_level,
                message=rule.description,
                value=current_value,
                threshold=rule.threshold,
                timestamp=datetime.now(),
                tags=rule.tags
            )
            
            self.active_alerts[alert.id] = alert
            self.alert_history.append(alert)
            
            # Actualizar última vez que se disparó la regla
            rule.last_triggered = datetime.now()
            
            # Ejecutar callbacks
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    logger.error(f"Error in alert callback: {e}")
            
            logger.warning(f"Alert triggered: {alert.message} (Value: {current_value}, Threshold: {rule.threshold})")
            
        except Exception as e:
            logger.error(f"Error triggering alert for rule {rule.id}: {e}")
    
    def _check_system_health(self) -> float:
        """Verificar salud del sistema"""
        try:
            health_score = 100.0
            
            # Penalizar por CPU alto
            cpu_usage = self.current_metrics.get("system.cpu_percent", 0)
            if cpu_usage > 80:
                health_score -= (cpu_usage - 80) * 0.5
            
            # Penalizar por memoria alta
            memory_usage = self.current_metrics.get("system.memory_percent", 0)
            if memory_usage > 80:
                health_score -= (memory_usage - 80) * 0.5
            
            # Penalizar por disco alto
            disk_usage = self.current_metrics.get("system.disk_percent", 0)
            if disk_usage > 90:
                health_score -= (disk_usage - 90) * 1.0
            
            # Penalizar por alertas activas
            health_score -= len(self.active_alerts) * 5
            
            return max(0.0, health_score)
            
        except Exception as e:
            logger.error(f"Error checking system health: {e}")
            return 0.0
    
    def add_alert_rule(self, rule: AlertRule) -> bool:
        """Agregar regla de alerta"""
        try:
            self.alert_rules[rule.id] = rule
            logger.info(f"Alert rule added: {rule.name}")
            return True
        except Exception as e:
            logger.error(f"Error adding alert rule: {e}")
            return False
    
    def remove_alert_rule(self, rule_id: str) -> bool:
        """Remover regla de alerta"""
        try:
            if rule_id in self.alert_rules:
                del self.alert_rules[rule_id]
                logger.info(f"Alert rule removed: {rule_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error removing alert rule: {e}")
            return False
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolver una alerta"""
        try:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.resolved = True
                alert.resolved_at = datetime.now()
                
                del self.active_alerts[alert_id]
                logger.info(f"Alert resolved: {alert_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error resolving alert: {e}")
            return False
    
    def add_alert_callback(self, callback: Callable):
        """Agregar callback de alerta"""
        self.alert_callbacks.append(callback)
    
    def remove_alert_callback(self, callback: Callable):
        """Remover callback de alerta"""
        if callback in self.alert_callbacks:
            self.alert_callbacks.remove(callback)
    
    async def get_system_health(self) -> SystemHealth:
        """Obtener estado de salud del sistema"""
        try:
            uptime = (datetime.now() - self.system_start_time).total_seconds()
            health_score = self._check_system_health()
            
            # Determinar estado general
            if health_score >= 90:
                overall_status = "excellent"
            elif health_score >= 70:
                overall_status = "good"
            elif health_score >= 50:
                overall_status = "fair"
            elif health_score >= 30:
                overall_status = "poor"
            else:
                overall_status = "critical"
            
            # Obtener métricas de red
            network_io = {
                "bytes_sent": self.current_metrics.get("system.network_bytes_sent", 0),
                "bytes_recv": self.current_metrics.get("system.network_bytes_recv", 0),
                "packets_sent": self.current_metrics.get("system.network_packets_sent", 0),
                "packets_recv": self.current_metrics.get("system.network_packets_recv", 0)
            }
            
            return SystemHealth(
                overall_status=overall_status,
                cpu_usage=self.current_metrics.get("system.cpu_percent", 0),
                memory_usage=self.current_metrics.get("system.memory_percent", 0),
                disk_usage=self.current_metrics.get("system.disk_percent", 0),
                network_io=network_io,
                active_alerts=len(self.active_alerts),
                total_metrics=len(self.current_metrics),
                uptime=uptime
            )
            
        except Exception as e:
            logger.error(f"Error getting system health: {e}")
            raise
    
    async def get_metrics_summary(self) -> Dict[str, Any]:
        """Obtener resumen de métricas"""
        try:
            summary = {
                "total_metrics": len(self.current_metrics),
                "metrics_by_type": {},
                "top_metrics": {},
                "recent_metrics": {}
            }
            
            # Agrupar por tipo
            for metric_name, metric_deque in self.metrics.items():
                if metric_deque:
                    latest_metric = metric_deque[-1]
                    metric_type = latest_metric.metric_type.value
                    
                    if metric_type not in summary["metrics_by_type"]:
                        summary["metrics_by_type"][metric_type] = 0
                    summary["metrics_by_type"][metric_type] += 1
            
            # Top métricas por valor
            sorted_metrics = sorted(
                self.current_metrics.items(),
                key=lambda x: x[1],
                reverse=True
            )
            summary["top_metrics"] = dict(sorted_metrics[:10])
            
            # Métricas recientes
            recent_time = datetime.now() - timedelta(minutes=5)
            for metric_name, metric_deque in self.metrics.items():
                recent_count = sum(
                    1 for metric in metric_deque
                    if metric.timestamp > recent_time
                )
                if recent_count > 0:
                    summary["recent_metrics"][metric_name] = recent_count
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting metrics summary: {e}")
            raise
    
    async def get_alerts_summary(self) -> Dict[str, Any]:
        """Obtener resumen de alertas"""
        try:
            return {
                "active_alerts": len(self.active_alerts),
                "total_alert_rules": len(self.alert_rules),
                "enabled_rules": len([r for r in self.alert_rules.values() if r.enabled]),
                "alert_levels": {
                    level.value: len([a for a in self.active_alerts.values() if a.alert_level == level])
                    for level in AlertLevel
                },
                "recent_alerts": [
                    {
                        "id": alert.id,
                        "rule_id": alert.rule_id,
                        "message": alert.message,
                        "level": alert.alert_level.value,
                        "value": alert.value,
                        "threshold": alert.threshold,
                        "timestamp": alert.timestamp.isoformat()
                    }
                    for alert in list(self.alert_history)[-10:]
                ],
                "alert_rules": [
                    {
                        "id": rule.id,
                        "name": rule.name,
                        "metric_name": rule.metric_name,
                        "condition": rule.condition.value,
                        "threshold": rule.threshold,
                        "alert_level": rule.alert_level.value,
                        "enabled": rule.enabled,
                        "last_triggered": rule.last_triggered.isoformat() if rule.last_triggered else None
                    }
                    for rule in self.alert_rules.values()
                ]
            }
            
        except Exception as e:
            logger.error(f"Error getting alerts summary: {e}")
            raise
    
    async def export_monitoring_data(self, filepath: str = None) -> str:
        """Exportar datos de monitoreo"""
        try:
            if filepath is None:
                filepath = f"exports/monitoring_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            # Crear directorio si no existe
            import os
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Preparar datos para exportación
            export_data = {
                "system_health": await self.get_system_health(),
                "metrics_summary": await self.get_metrics_summary(),
                "alerts_summary": await self.get_alerts_summary(),
                "current_metrics": self.current_metrics,
                "exported_at": datetime.now().isoformat()
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"Monitoring data exported to {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error exporting monitoring data: {e}")
            raise

























