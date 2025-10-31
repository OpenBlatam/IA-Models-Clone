"""
Advanced Monitoring - Sistema de monitoreo avanzado
"""

import asyncio
import logging
import psutil
import time
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
import uuid
from collections import defaultdict, deque
import threading

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Tipos de métricas."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"
    CUSTOM = "custom"


class AlertLevel(Enum):
    """Niveles de alerta."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class Metric:
    """Métrica del sistema."""
    metric_id: str
    name: str
    metric_type: MetricType
    value: float
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Alert:
    """Alerta del sistema."""
    alert_id: str
    name: str
    level: AlertLevel
    message: str
    metric_name: str
    threshold: float
    current_value: float
    timestamp: datetime
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceSnapshot:
    """Instantánea de rendimiento."""
    snapshot_id: str
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    disk_usage_percent: float
    disk_free_gb: float
    network_bytes_sent: int
    network_bytes_recv: int
    active_connections: int
    process_count: int
    load_average: List[float] = field(default_factory=list)


class AdvancedMonitoring:
    """
    Sistema de monitoreo avanzado.
    """
    
    def __init__(self, collection_interval: int = 30):
        """Inicializar sistema de monitoreo."""
        self.collection_interval = collection_interval
        
        # Almacenamiento de métricas
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.alerts: List[Alert] = []
        self.performance_snapshots: deque = deque(maxlen=100)
        
        # Configuración de alertas
        self.alert_thresholds = {
            "cpu_percent": {"warning": 70.0, "critical": 90.0},
            "memory_percent": {"warning": 80.0, "critical": 95.0},
            "disk_usage_percent": {"warning": 85.0, "critical": 95.0},
            "response_time_ms": {"warning": 1000.0, "critical": 5000.0},
            "error_rate_percent": {"warning": 5.0, "critical": 10.0}
        }
        
        # Callbacks de alertas
        self.alert_callbacks: List[Callable] = []
        
        # Estado del sistema
        self.is_running = False
        self.collection_task: Optional[asyncio.Task] = None
        
        # Estadísticas
        self.stats = {
            "total_metrics_collected": 0,
            "total_alerts_triggered": 0,
            "total_alerts_resolved": 0,
            "start_time": datetime.now(),
            "last_collection": None
        }
        
        logger.info("AdvancedMonitoring inicializado")
    
    async def initialize(self):
        """Inicializar el sistema de monitoreo."""
        try:
            # Iniciar recolección de métricas
            self.is_running = True
            self.collection_task = asyncio.create_task(self._collect_metrics_loop())
            
            logger.info("AdvancedMonitoring inicializado exitosamente")
            
        except Exception as e:
            logger.error(f"Error al inicializar AdvancedMonitoring: {e}")
            raise
    
    async def shutdown(self):
        """Cerrar el sistema de monitoreo."""
        try:
            self.is_running = False
            
            if self.collection_task:
                self.collection_task.cancel()
                try:
                    await self.collection_task
                except asyncio.CancelledError:
                    pass
            
            logger.info("AdvancedMonitoring cerrado")
            
        except Exception as e:
            logger.error(f"Error al cerrar AdvancedMonitoring: {e}")
    
    async def _collect_metrics_loop(self):
        """Bucle de recolección de métricas."""
        while self.is_running:
            try:
                await self._collect_system_metrics()
                await self._check_alerts()
                self.stats["last_collection"] = datetime.now()
                
                await asyncio.sleep(self.collection_interval)
                
            except Exception as e:
                logger.error(f"Error en recolección de métricas: {e}")
                await asyncio.sleep(self.collection_interval)
    
    async def _collect_system_metrics(self):
        """Recolectar métricas del sistema."""
        try:
            now = datetime.now()
            
            # Métricas de CPU
            cpu_percent = psutil.cpu_percent(interval=1)
            await self._record_metric("cpu_percent", MetricType.GAUGE, cpu_percent, now)
            
            # Métricas de memoria
            memory = psutil.virtual_memory()
            await self._record_metric("memory_percent", MetricType.GAUGE, memory.percent, now)
            await self._record_metric("memory_used_mb", MetricType.GAUGE, memory.used / (1024 * 1024), now)
            await self._record_metric("memory_available_mb", MetricType.GAUGE, memory.available / (1024 * 1024), now)
            
            # Métricas de disco
            disk = psutil.disk_usage('/')
            await self._record_metric("disk_usage_percent", MetricType.GAUGE, (disk.used / disk.total) * 100, now)
            await self._record_metric("disk_free_gb", MetricType.GAUGE, disk.free / (1024 * 1024 * 1024), now)
            
            # Métricas de red
            network = psutil.net_io_counters()
            await self._record_metric("network_bytes_sent", MetricType.COUNTER, network.bytes_sent, now)
            await self._record_metric("network_bytes_recv", MetricType.COUNTER, network.bytes_recv, now)
            
            # Métricas de procesos
            process_count = len(psutil.pids())
            await self._record_metric("process_count", MetricType.GAUGE, process_count, now)
            
            # Métricas de carga del sistema
            try:
                load_avg = psutil.getloadavg()
                await self._record_metric("load_average_1m", MetricType.GAUGE, load_avg[0], now)
                await self._record_metric("load_average_5m", MetricType.GAUGE, load_avg[1], now)
                await self._record_metric("load_average_15m", MetricType.GAUGE, load_avg[2], now)
            except AttributeError:
                # Windows no tiene load average
                pass
            
            # Crear instantánea de rendimiento
            snapshot = PerformanceSnapshot(
                snapshot_id=str(uuid.uuid4()),
                timestamp=now,
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_used_mb=memory.used / (1024 * 1024),
                memory_available_mb=memory.available / (1024 * 1024),
                disk_usage_percent=(disk.used / disk.total) * 100,
                disk_free_gb=disk.free / (1024 * 1024 * 1024),
                network_bytes_sent=network.bytes_sent,
                network_bytes_recv=network.bytes_recv,
                active_connections=len(psutil.net_connections()),
                process_count=process_count,
                load_average=list(load_avg) if hasattr(psutil, 'getloadavg') else []
            )
            
            self.performance_snapshots.append(snapshot)
            
            self.stats["total_metrics_collected"] += 1
            
        except Exception as e:
            logger.error(f"Error al recolectar métricas del sistema: {e}")
    
    async def _record_metric(
        self,
        name: str,
        metric_type: MetricType,
        value: float,
        timestamp: datetime,
        tags: Dict[str, str] = None
    ):
        """Registrar una métrica."""
        try:
            metric = Metric(
                metric_id=str(uuid.uuid4()),
                name=name,
                metric_type=metric_type,
                value=value,
                timestamp=timestamp,
                tags=tags or {}
            )
            
            self.metrics[name].append(metric)
            
        except Exception as e:
            logger.error(f"Error al registrar métrica {name}: {e}")
    
    async def _check_alerts(self):
        """Verificar alertas."""
        try:
            for metric_name, thresholds in self.alert_thresholds.items():
                if metric_name not in self.metrics or not self.metrics[metric_name]:
                    continue
                
                latest_metric = self.metrics[metric_name][-1]
                current_value = latest_metric.value
                
                # Verificar umbrales
                for level_name, threshold in thresholds.items():
                    level = AlertLevel(level_name.upper())
                    
                    if current_value >= threshold:
                        # Verificar si ya existe una alerta activa
                        active_alert = next(
                            (alert for alert in self.alerts 
                             if alert.metric_name == metric_name and 
                                alert.level == level and 
                                not alert.resolved),
                            None
                        )
                        
                        if not active_alert:
                            # Crear nueva alerta
                            alert = Alert(
                                alert_id=str(uuid.uuid4()),
                                name=f"{metric_name}_{level_name}",
                                level=level,
                                message=f"{metric_name} excedió el umbral {level_name}: {current_value:.2f} >= {threshold}",
                                metric_name=metric_name,
                                threshold=threshold,
                                current_value=current_value,
                                timestamp=datetime.now()
                            )
                            
                            self.alerts.append(alert)
                            self.stats["total_alerts_triggered"] += 1
                            
                            # Ejecutar callbacks de alerta
                            await self._trigger_alert_callbacks(alert)
                            
                            logger.warning(f"ALERTA [{level.value.upper()}] {alert.message}")
                    else:
                        # Resolver alertas si el valor está por debajo del umbral
                        await self._resolve_alerts(metric_name, level, current_value)
            
        except Exception as e:
            logger.error(f"Error al verificar alertas: {e}")
    
    async def _resolve_alerts(self, metric_name: str, level: AlertLevel, current_value: float):
        """Resolver alertas."""
        try:
            for alert in self.alerts:
                if (alert.metric_name == metric_name and 
                    alert.level == level and 
                    not alert.resolved and 
                    current_value < alert.threshold):
                    
                    alert.resolved = True
                    alert.resolved_at = datetime.now()
                    self.stats["total_alerts_resolved"] += 1
                    
                    logger.info(f"Alerta resuelta: {alert.name}")
                    
        except Exception as e:
            logger.error(f"Error al resolver alertas: {e}")
    
    async def _trigger_alert_callbacks(self, alert: Alert):
        """Ejecutar callbacks de alerta."""
        try:
            for callback in self.alert_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(alert)
                    else:
                        callback(alert)
                except Exception as e:
                    logger.error(f"Error en callback de alerta: {e}")
                    
        except Exception as e:
            logger.error(f"Error al ejecutar callbacks de alerta: {e}")
    
    def add_alert_callback(self, callback: Callable):
        """Agregar callback de alerta."""
        self.alert_callbacks.append(callback)
    
    def remove_alert_callback(self, callback: Callable):
        """Remover callback de alerta."""
        if callback in self.alert_callbacks:
            self.alert_callbacks.remove(callback)
    
    async def record_custom_metric(
        self,
        name: str,
        value: float,
        metric_type: MetricType = MetricType.GAUGE,
        tags: Dict[str, str] = None
    ):
        """Registrar métrica personalizada."""
        await self._record_metric(name, metric_type, value, datetime.now(), tags)
    
    async def record_timing(self, name: str, duration_seconds: float, tags: Dict[str, str] = None):
        """Registrar tiempo de ejecución."""
        await self._record_metric(f"{name}_duration", MetricType.TIMER, duration_seconds, datetime.now(), tags)
        await self._record_metric(f"{name}_count", MetricType.COUNTER, 1, datetime.now(), tags)
    
    async def record_counter(self, name: str, increment: float = 1, tags: Dict[str, str] = None):
        """Registrar contador."""
        await self._record_metric(name, MetricType.COUNTER, increment, datetime.now(), tags)
    
    async def get_metrics(
        self,
        metric_name: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Obtener métricas."""
        try:
            results = []
            
            if metric_name:
                if metric_name in self.metrics:
                    metrics = list(self.metrics[metric_name])
                else:
                    return []
            else:
                # Obtener todas las métricas
                metrics = []
                for metric_list in self.metrics.values():
                    metrics.extend(metric_list)
            
            # Filtrar por tiempo
            if start_time or end_time:
                filtered_metrics = []
                for metric in metrics:
                    if start_time and metric.timestamp < start_time:
                        continue
                    if end_time and metric.timestamp > end_time:
                        continue
                    filtered_metrics.append(metric)
                metrics = filtered_metrics
            
            # Ordenar por timestamp
            metrics.sort(key=lambda x: x.timestamp, reverse=True)
            
            # Limitar resultados
            metrics = metrics[:limit]
            
            # Convertir a diccionario
            for metric in metrics:
                results.append({
                    "metric_id": metric.metric_id,
                    "name": metric.name,
                    "type": metric.metric_type.value,
                    "value": metric.value,
                    "timestamp": metric.timestamp.isoformat(),
                    "tags": metric.tags,
                    "metadata": metric.metadata
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error al obtener métricas: {e}")
            return []
    
    async def get_alerts(
        self,
        level: Optional[AlertLevel] = None,
        resolved: Optional[bool] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Obtener alertas."""
        try:
            alerts = self.alerts.copy()
            
            # Filtrar por nivel
            if level:
                alerts = [alert for alert in alerts if alert.level == level]
            
            # Filtrar por estado de resolución
            if resolved is not None:
                alerts = [alert for alert in alerts if alert.resolved == resolved]
            
            # Ordenar por timestamp
            alerts.sort(key=lambda x: x.timestamp, reverse=True)
            
            # Limitar resultados
            alerts = alerts[:limit]
            
            # Convertir a diccionario
            results = []
            for alert in alerts:
                results.append({
                    "alert_id": alert.alert_id,
                    "name": alert.name,
                    "level": alert.level.value,
                    "message": alert.message,
                    "metric_name": alert.metric_name,
                    "threshold": alert.threshold,
                    "current_value": alert.current_value,
                    "timestamp": alert.timestamp.isoformat(),
                    "resolved": alert.resolved,
                    "resolved_at": alert.resolved_at.isoformat() if alert.resolved_at else None,
                    "metadata": alert.metadata
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error al obtener alertas: {e}")
            return []
    
    async def get_performance_snapshots(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Obtener instantáneas de rendimiento."""
        try:
            snapshots = list(self.performance_snapshots)
            
            # Filtrar por tiempo
            if start_time or end_time:
                filtered_snapshots = []
                for snapshot in snapshots:
                    if start_time and snapshot.timestamp < start_time:
                        continue
                    if end_time and snapshot.timestamp > end_time:
                        continue
                    filtered_snapshots.append(snapshot)
                snapshots = filtered_snapshots
            
            # Ordenar por timestamp
            snapshots.sort(key=lambda x: x.timestamp, reverse=True)
            
            # Limitar resultados
            snapshots = snapshots[:limit]
            
            # Convertir a diccionario
            results = []
            for snapshot in snapshots:
                results.append({
                    "snapshot_id": snapshot.snapshot_id,
                    "timestamp": snapshot.timestamp.isoformat(),
                    "cpu_percent": snapshot.cpu_percent,
                    "memory_percent": snapshot.memory_percent,
                    "memory_used_mb": snapshot.memory_used_mb,
                    "memory_available_mb": snapshot.memory_available_mb,
                    "disk_usage_percent": snapshot.disk_usage_percent,
                    "disk_free_gb": snapshot.disk_free_gb,
                    "network_bytes_sent": snapshot.network_bytes_sent,
                    "network_bytes_recv": snapshot.network_bytes_recv,
                    "active_connections": snapshot.active_connections,
                    "process_count": snapshot.process_count,
                    "load_average": snapshot.load_average
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error al obtener instantáneas de rendimiento: {e}")
            return []
    
    async def get_monitoring_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de monitoreo."""
        try:
            # Calcular estadísticas de métricas
            total_metrics = sum(len(metrics) for metrics in self.metrics.values())
            unique_metrics = len(self.metrics)
            
            # Calcular estadísticas de alertas
            active_alerts = len([alert for alert in self.alerts if not alert.resolved])
            total_alerts = len(self.alerts)
            
            # Calcular uptime
            uptime_seconds = (datetime.now() - self.stats["start_time"]).total_seconds()
            
            return {
                **self.stats,
                "uptime_seconds": uptime_seconds,
                "uptime_hours": uptime_seconds / 3600,
                "total_metrics": total_metrics,
                "unique_metrics": unique_metrics,
                "active_alerts": active_alerts,
                "total_alerts": total_alerts,
                "performance_snapshots": len(self.performance_snapshots),
                "collection_interval": self.collection_interval,
                "is_running": self.is_running,
                "alert_callbacks": len(self.alert_callbacks),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error al obtener estadísticas de monitoreo: {e}")
            return {}
    
    async def health_check(self) -> Dict[str, Any]:
        """Verificar salud del sistema de monitoreo."""
        try:
            # Verificar si está recolectando métricas
            last_collection = self.stats.get("last_collection")
            collection_delay = None
            if last_collection:
                collection_delay = (datetime.now() - last_collection).total_seconds()
            
            # Verificar alertas críticas
            critical_alerts = len([alert for alert in self.alerts 
                                 if alert.level == AlertLevel.CRITICAL and not alert.resolved])
            
            # Determinar estado general
            status = "healthy"
            if not self.is_running:
                status = "stopped"
            elif collection_delay and collection_delay > self.collection_interval * 2:
                status = "degraded"
            elif critical_alerts > 0:
                status = "critical"
            
            return {
                "status": status,
                "is_running": self.is_running,
                "last_collection": last_collection.isoformat() if last_collection else None,
                "collection_delay_seconds": collection_delay,
                "critical_alerts": critical_alerts,
                "total_metrics": sum(len(metrics) for metrics in self.metrics.values()),
                "active_alerts": len([alert for alert in self.alerts if not alert.resolved]),
                "uptime_seconds": (datetime.now() - self.stats["start_time"]).total_seconds(),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error en health check de monitoreo: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }




