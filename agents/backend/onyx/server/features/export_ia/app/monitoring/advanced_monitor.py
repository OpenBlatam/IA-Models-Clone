"""
Advanced Monitor - Sistema de monitoreo y observabilidad avanzado
"""

import asyncio
import logging
import json
import time
from typing import Dict, List, Any, Optional, Callable, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import threading
from collections import defaultdict, deque
import uuid

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """Niveles de alerta."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MetricType(Enum):
    """Tipos de métricas."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


@dataclass
class Alert:
    """Alerta del sistema."""
    alert_id: str
    level: AlertLevel
    title: str
    message: str
    source: str
    timestamp: datetime
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Metric:
    """Métrica del sistema."""
    metric_id: str
    name: str
    value: float
    metric_type: MetricType
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HealthCheck:
    """Verificación de salud."""
    check_id: str
    name: str
    status: str
    message: str
    timestamp: datetime
    response_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class AdvancedMonitor:
    """
    Sistema de monitoreo y observabilidad avanzado.
    """
    
    def __init__(self):
        """Inicializar monitor avanzado."""
        self.alerts: Dict[str, Alert] = {}
        self.metrics: Dict[str, List[Metric]] = defaultdict(list)
        self.health_checks: Dict[str, HealthCheck] = {}
        self.alert_rules: Dict[str, Dict[str, Any]] = {}
        self.metric_collectors: Dict[str, Callable] = {}
        self.health_checkers: Dict[str, Callable] = {}
        
        # Configuración
        self.max_alerts = 10000
        self.max_metrics_per_name = 1000
        self.alert_retention_days = 30
        self.metric_retention_days = 7
        
        # Monitoreo activo
        self.monitoring_active = False
        self.collection_interval = 10  # segundos
        self.health_check_interval = 30  # segundos
        
        # Tareas de monitoreo
        self.monitoring_tasks: List[asyncio.Task] = []
        
        # Estadísticas
        self.stats = {
            "alerts_generated": 0,
            "metrics_collected": 0,
            "health_checks_performed": 0,
            "start_time": datetime.now()
        }
        
        logger.info("AdvancedMonitor inicializado")
    
    async def initialize(self):
        """Inicializar el monitor avanzado."""
        try:
            # Configurar reglas de alerta por defecto
            await self._setup_default_alert_rules()
            
            # Iniciar monitoreo
            await self.start_monitoring()
            
            logger.info("AdvancedMonitor inicializado exitosamente")
            
        except Exception as e:
            logger.error(f"Error al inicializar AdvancedMonitor: {e}")
            raise
    
    async def shutdown(self):
        """Cerrar el monitor avanzado."""
        try:
            # Detener monitoreo
            await self.stop_monitoring()
            
            # Limpiar datos antiguos
            await self._cleanup_old_data()
            
            logger.info("AdvancedMonitor cerrado")
            
        except Exception as e:
            logger.error(f"Error al cerrar AdvancedMonitor: {e}")
    
    async def start_monitoring(self):
        """Iniciar monitoreo del sistema."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        
        # Iniciar tareas de monitoreo
        self.monitoring_tasks = [
            asyncio.create_task(self._metrics_collector()),
            asyncio.create_task(self._health_checker()),
            asyncio.create_task(self._alert_processor()),
            asyncio.create_task(self._data_cleanup())
        ]
        
        logger.info("Monitoreo avanzado iniciado")
    
    async def stop_monitoring(self):
        """Detener monitoreo del sistema."""
        if not self.monitoring_active:
            return
        
        self.monitoring_active = False
        
        # Cancelar tareas de monitoreo
        for task in self.monitoring_tasks:
            task.cancel()
        
        # Esperar a que terminen
        await asyncio.gather(*self.monitoring_tasks, return_exceptions=True)
        
        self.monitoring_tasks.clear()
        logger.info("Monitoreo avanzado detenido")
    
    async def _setup_default_alert_rules(self):
        """Configurar reglas de alerta por defecto."""
        default_rules = {
            "high_cpu_usage": {
                "metric": "system.cpu.usage",
                "condition": ">",
                "threshold": 80.0,
                "level": AlertLevel.WARNING,
                "title": "Alto uso de CPU",
                "message": "El uso de CPU ha excedido el 80%"
            },
            "high_memory_usage": {
                "metric": "system.memory.usage",
                "condition": ">",
                "threshold": 85.0,
                "level": AlertLevel.WARNING,
                "title": "Alto uso de memoria",
                "message": "El uso de memoria ha excedido el 85%"
            },
            "low_disk_space": {
                "metric": "system.disk.usage",
                "condition": ">",
                "threshold": 90.0,
                "level": AlertLevel.ERROR,
                "title": "Espacio en disco bajo",
                "message": "El espacio en disco ha excedido el 90%"
            },
            "high_error_rate": {
                "metric": "application.error_rate",
                "condition": ">",
                "threshold": 5.0,
                "level": AlertLevel.ERROR,
                "title": "Alta tasa de errores",
                "message": "La tasa de errores ha excedido el 5%"
            },
            "slow_response_time": {
                "metric": "application.response_time",
                "condition": ">",
                "threshold": 5.0,
                "level": AlertLevel.WARNING,
                "title": "Tiempo de respuesta lento",
                "message": "El tiempo de respuesta ha excedido 5 segundos"
            }
        }
        
        for rule_name, rule_config in default_rules.items():
            self.alert_rules[rule_name] = rule_config
    
    async def _metrics_collector(self):
        """Recolector de métricas."""
        while self.monitoring_active:
            try:
                # Ejecutar recolectores de métricas
                for name, collector in self.metric_collectors.items():
                    try:
                        if asyncio.iscoroutinefunction(collector):
                            metrics = await collector()
                        else:
                            metrics = collector()
                        
                        if isinstance(metrics, list):
                            for metric in metrics:
                                await self.record_metric(metric)
                        else:
                            await self.record_metric(metrics)
                            
                    except Exception as e:
                        logger.error(f"Error en recolector de métricas {name}: {e}")
                
                await asyncio.sleep(self.collection_interval)
                
            except Exception as e:
                logger.error(f"Error en recolector de métricas: {e}")
                await asyncio.sleep(self.collection_interval)
    
    async def _health_checker(self):
        """Verificador de salud."""
        while self.monitoring_active:
            try:
                # Ejecutar verificadores de salud
                for name, checker in self.health_checkers.items():
                    try:
                        start_time = time.time()
                        
                        if asyncio.iscoroutinefunction(checker):
                            result = await checker()
                        else:
                            result = checker()
                        
                        response_time = time.time() - start_time
                        
                        await self.record_health_check(
                            name=name,
                            status=result.get("status", "unknown"),
                            message=result.get("message", ""),
                            response_time=response_time,
                            metadata=result.get("metadata", {})
                        )
                        
                    except Exception as e:
                        logger.error(f"Error en verificador de salud {name}: {e}")
                        await self.record_health_check(
                            name=name,
                            status="error",
                            message=str(e),
                            response_time=0.0
                        )
                
                await asyncio.sleep(self.health_check_interval)
                
            except Exception as e:
                logger.error(f"Error en verificador de salud: {e}")
                await asyncio.sleep(self.health_check_interval)
    
    async def _alert_processor(self):
        """Procesador de alertas."""
        while self.monitoring_active:
            try:
                # Procesar reglas de alerta
                for rule_name, rule in self.alert_rules.items():
                    await self._check_alert_rule(rule_name, rule)
                
                await asyncio.sleep(5)  # Verificar alertas cada 5 segundos
                
            except Exception as e:
                logger.error(f"Error en procesador de alertas: {e}")
                await asyncio.sleep(5)
    
    async def _data_cleanup(self):
        """Limpieza de datos antiguos."""
        while self.monitoring_active:
            try:
                await self._cleanup_old_data()
                await asyncio.sleep(3600)  # Limpiar cada hora
                
            except Exception as e:
                logger.error(f"Error en limpieza de datos: {e}")
                await asyncio.sleep(3600)
    
    async def _check_alert_rule(self, rule_name: str, rule: Dict[str, Any]):
        """Verificar regla de alerta."""
        try:
            metric_name = rule["metric"]
            condition = rule["condition"]
            threshold = rule["threshold"]
            
            # Obtener métrica más reciente
            if metric_name not in self.metrics or not self.metrics[metric_name]:
                return
            
            latest_metric = self.metrics[metric_name][-1]
            
            # Verificar condición
            should_alert = False
            if condition == ">":
                should_alert = latest_metric.value > threshold
            elif condition == ">=":
                should_alert = latest_metric.value >= threshold
            elif condition == "<":
                should_alert = latest_metric.value < threshold
            elif condition == "<=":
                should_alert = latest_metric.value <= threshold
            elif condition == "==":
                should_alert = latest_metric.value == threshold
            elif condition == "!=":
                should_alert = latest_metric.value != threshold
            
            if should_alert:
                # Verificar si ya existe una alerta activa para esta regla
                existing_alert = None
                for alert in self.alerts.values():
                    if (alert.source == rule_name and 
                        not alert.resolved and
                        (datetime.now() - alert.timestamp).total_seconds() < 300):  # 5 minutos
                        existing_alert = alert
                        break
                
                if not existing_alert:
                    # Crear nueva alerta
                    await self.create_alert(
                        level=rule["level"],
                        title=rule["title"],
                        message=f"{rule['message']} (Valor actual: {latest_metric.value})",
                        source=rule_name,
                        metadata={
                            "metric_name": metric_name,
                            "metric_value": latest_metric.value,
                            "threshold": threshold,
                            "condition": condition
                        }
                    )
            
        except Exception as e:
            logger.error(f"Error al verificar regla de alerta {rule_name}: {e}")
    
    async def record_metric(
        self,
        name: str,
        value: float,
        metric_type: MetricType = MetricType.GAUGE,
        tags: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Registrar una métrica."""
        try:
            metric_id = str(uuid.uuid4())
            
            metric = Metric(
                metric_id=metric_id,
                name=name,
                value=value,
                metric_type=metric_type,
                timestamp=datetime.now(),
                tags=tags or {},
                metadata=metadata or {}
            )
            
            # Agregar a la lista de métricas
            self.metrics[name].append(metric)
            
            # Limitar número de métricas por nombre
            if len(self.metrics[name]) > self.max_metrics_per_name:
                self.metrics[name] = self.metrics[name][-self.max_metrics_per_name:]
            
            self.stats["metrics_collected"] += 1
            
            return metric_id
            
        except Exception as e:
            logger.error(f"Error al registrar métrica: {e}")
            return ""
    
    async def create_alert(
        self,
        level: AlertLevel,
        title: str,
        message: str,
        source: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Crear una alerta."""
        try:
            alert_id = str(uuid.uuid4())
            
            alert = Alert(
                alert_id=alert_id,
                level=level,
                title=title,
                message=message,
                source=source,
                timestamp=datetime.now(),
                metadata=metadata or {}
            )
            
            self.alerts[alert_id] = alert
            
            # Limitar número de alertas
            if len(self.alerts) > self.max_alerts:
                # Eliminar alertas más antiguas
                oldest_alerts = sorted(
                    self.alerts.items(),
                    key=lambda x: x[1].timestamp
                )[:len(self.alerts) - self.max_alerts + 100]
                
                for alert_id_to_remove, _ in oldest_alerts:
                    del self.alerts[alert_id_to_remove]
            
            self.stats["alerts_generated"] += 1
            
            # Log de alerta
            logger.warning(f"ALERTA [{level.value.upper()}] {title}: {message}")
            
            return alert_id
            
        except Exception as e:
            logger.error(f"Error al crear alerta: {e}")
            return ""
    
    async def record_health_check(
        self,
        name: str,
        status: str,
        message: str,
        response_time: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Registrar verificación de salud."""
        try:
            check_id = str(uuid.uuid4())
            
            health_check = HealthCheck(
                check_id=check_id,
                name=name,
                status=status,
                message=message,
                timestamp=datetime.now(),
                response_time=response_time,
                metadata=metadata or {}
            )
            
            self.health_checks[name] = health_check
            self.stats["health_checks_performed"] += 1
            
            return check_id
            
        except Exception as e:
            logger.error(f"Error al registrar verificación de salud: {e}")
            return ""
    
    async def resolve_alert(self, alert_id: str) -> bool:
        """Resolver una alerta."""
        try:
            if alert_id in self.alerts:
                alert = self.alerts[alert_id]
                alert.resolved = True
                alert.resolved_at = datetime.now()
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error al resolver alerta: {e}")
            return False
    
    async def add_metric_collector(self, name: str, collector: Callable):
        """Agregar recolector de métricas."""
        self.metric_collectors[name] = collector
        logger.info(f"Recolector de métricas {name} agregado")
    
    async def add_health_checker(self, name: str, checker: Callable):
        """Agregar verificador de salud."""
        self.health_checkers[name] = checker
        logger.info(f"Verificador de salud {name} agregado")
    
    async def add_alert_rule(self, name: str, rule: Dict[str, Any]):
        """Agregar regla de alerta."""
        self.alert_rules[name] = rule
        logger.info(f"Regla de alerta {name} agregada")
    
    async def get_metrics(
        self,
        name: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Obtener métricas."""
        try:
            result = {}
            
            if name:
                # Obtener métricas específicas
                if name in self.metrics:
                    metrics = self.metrics[name]
                    
                    # Filtrar por tiempo si se especifica
                    if start_time or end_time:
                        filtered_metrics = []
                        for metric in metrics:
                            if start_time and metric.timestamp < start_time:
                                continue
                            if end_time and metric.timestamp > end_time:
                                continue
                            filtered_metrics.append(metric)
                        metrics = filtered_metrics
                    
                    # Limitar resultados
                    metrics = metrics[-limit:] if limit > 0 else metrics
                    
                    result[name] = [
                        {
                            "metric_id": m.metric_id,
                            "name": m.name,
                            "value": m.value,
                            "type": m.metric_type.value,
                            "timestamp": m.timestamp.isoformat(),
                            "tags": m.tags,
                            "metadata": m.metadata
                        }
                        for m in metrics
                    ]
            else:
                # Obtener todas las métricas
                for metric_name, metrics_list in self.metrics.items():
                    if start_time or end_time:
                        filtered_metrics = []
                        for metric in metrics_list:
                            if start_time and metric.timestamp < start_time:
                                continue
                            if end_time and metric.timestamp > end_time:
                                continue
                            filtered_metrics.append(metric)
                        metrics_list = filtered_metrics
                    
                    metrics_list = metrics_list[-limit:] if limit > 0 else metrics_list
                    
                    result[metric_name] = [
                        {
                            "metric_id": m.metric_id,
                            "name": m.name,
                            "value": m.value,
                            "type": m.metric_type.value,
                            "timestamp": m.timestamp.isoformat(),
                            "tags": m.tags,
                            "metadata": m.metadata
                        }
                        for m in metrics_list
                    ]
            
            return result
            
        except Exception as e:
            logger.error(f"Error al obtener métricas: {e}")
            return {}
    
    async def get_alerts(
        self,
        level: Optional[AlertLevel] = None,
        resolved: Optional[bool] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Obtener alertas."""
        try:
            alerts = list(self.alerts.values())
            
            # Filtrar por nivel
            if level:
                alerts = [a for a in alerts if a.level == level]
            
            # Filtrar por estado de resolución
            if resolved is not None:
                alerts = [a for a in alerts if a.resolved == resolved]
            
            # Filtrar por tiempo
            if start_time:
                alerts = [a for a in alerts if a.timestamp >= start_time]
            if end_time:
                alerts = [a for a in alerts if a.timestamp <= end_time]
            
            # Ordenar por timestamp (más recientes primero)
            alerts.sort(key=lambda x: x.timestamp, reverse=True)
            
            # Limitar resultados
            alerts = alerts[:limit] if limit > 0 else alerts
            
            return [
                {
                    "alert_id": a.alert_id,
                    "level": a.level.value,
                    "title": a.title,
                    "message": a.message,
                    "source": a.source,
                    "timestamp": a.timestamp.isoformat(),
                    "resolved": a.resolved,
                    "resolved_at": a.resolved_at.isoformat() if a.resolved_at else None,
                    "metadata": a.metadata
                }
                for a in alerts
            ]
            
        except Exception as e:
            logger.error(f"Error al obtener alertas: {e}")
            return []
    
    async def get_health_checks(self) -> Dict[str, Dict[str, Any]]:
        """Obtener verificaciones de salud."""
        try:
            return {
                name: {
                    "check_id": check.check_id,
                    "name": check.name,
                    "status": check.status,
                    "message": check.message,
                    "timestamp": check.timestamp.isoformat(),
                    "response_time": check.response_time,
                    "metadata": check.metadata
                }
                for name, check in self.health_checks.items()
            }
            
        except Exception as e:
            logger.error(f"Error al obtener verificaciones de salud: {e}")
            return {}
    
    async def get_dashboard_data(self) -> Dict[str, Any]:
        """Obtener datos para dashboard."""
        try:
            # Estadísticas generales
            total_alerts = len(self.alerts)
            active_alerts = len([a for a in self.alerts.values() if not a.resolved])
            
            # Alertas por nivel
            alerts_by_level = defaultdict(int)
            for alert in self.alerts.values():
                alerts_by_level[alert.level.value] += 1
            
            # Métricas más recientes
            recent_metrics = {}
            for name, metrics_list in self.metrics.items():
                if metrics_list:
                    latest = metrics_list[-1]
                    recent_metrics[name] = {
                        "value": latest.value,
                        "timestamp": latest.timestamp.isoformat(),
                        "type": latest.metric_type.value
                    }
            
            # Estado de verificaciones de salud
            health_status = {}
            for name, check in self.health_checks.items():
                health_status[name] = {
                    "status": check.status,
                    "response_time": check.response_time,
                    "last_check": check.timestamp.isoformat()
                }
            
            return {
                "stats": {
                    **self.stats,
                    "uptime_seconds": (datetime.now() - self.stats["start_time"]).total_seconds(),
                    "total_alerts": total_alerts,
                    "active_alerts": active_alerts,
                    "alerts_by_level": dict(alerts_by_level)
                },
                "recent_metrics": recent_metrics,
                "health_status": health_status,
                "monitoring": {
                    "active": self.monitoring_active,
                    "collection_interval": self.collection_interval,
                    "health_check_interval": self.health_check_interval
                },
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error al obtener datos del dashboard: {e}")
            return {"error": str(e)}
    
    async def _cleanup_old_data(self):
        """Limpiar datos antiguos."""
        try:
            now = datetime.now()
            
            # Limpiar alertas antiguas
            alerts_to_remove = []
            for alert_id, alert in self.alerts.items():
                if (now - alert.timestamp).days > self.alert_retention_days:
                    alerts_to_remove.append(alert_id)
            
            for alert_id in alerts_to_remove:
                del self.alerts[alert_id]
            
            # Limpiar métricas antiguas
            for metric_name in list(self.metrics.keys()):
                metrics_list = self.metrics[metric_name]
                metrics_list[:] = [
                    m for m in metrics_list
                    if (now - m.timestamp).days <= self.metric_retention_days
                ]
                
                # Eliminar listas vacías
                if not metrics_list:
                    del self.metrics[metric_name]
            
            if alerts_to_remove:
                logger.info(f"Limpiadas {len(alerts_to_remove)} alertas antiguas")
            
        except Exception as e:
            logger.error(f"Error en limpieza de datos antiguos: {e}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Verificar salud del monitor."""
        try:
            return {
                "status": "healthy",
                "monitoring_active": self.monitoring_active,
                "stats": self.stats,
                "active_tasks": len(self.monitoring_tasks),
                "alerts_count": len(self.alerts),
                "metrics_count": sum(len(metrics) for metrics in self.metrics.values()),
                "health_checks_count": len(self.health_checks),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error en health check del monitor: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }




