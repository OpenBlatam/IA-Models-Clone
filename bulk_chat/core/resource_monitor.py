"""
Resource Monitor - Monitor de Recursos
=======================================

Sistema de monitoreo de recursos en tiempo real con alertas y análisis de tendencias.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import deque
import statistics

logger = logging.getLogger(__name__)

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logger.warning("psutil not available, resource monitoring will be limited")


class ResourceType(Enum):
    """Tipo de recurso."""
    CPU = "cpu"
    MEMORY = "memory"
    DISK = "disk"
    NETWORK = "network"
    THREADS = "threads"
    CONNECTIONS = "connections"


class AlertLevel(Enum):
    """Nivel de alerta."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class ResourceMetric:
    """Métrica de recurso."""
    resource_type: ResourceType
    value: float
    unit: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResourceAlert:
    """Alerta de recurso."""
    alert_id: str
    resource_type: ResourceType
    level: AlertLevel
    message: str
    threshold: float
    current_value: float
    timestamp: datetime = field(default_factory=datetime.now)
    resolved: bool = False


class ResourceMonitor:
    """Monitor de recursos."""
    
    def __init__(self, history_size: int = 1000):
        self.history_size = history_size
        self.metrics: Dict[ResourceType, deque] = {
            rt: deque(maxlen=history_size)
            for rt in ResourceType
        }
        self.alerts: List[ResourceAlert] = []
        self.thresholds: Dict[ResourceType, Dict[str, float]] = {
            ResourceType.CPU: {"warning": 70.0, "critical": 90.0},
            ResourceType.MEMORY: {"warning": 80.0, "critical": 95.0},
            ResourceType.DISK: {"warning": 85.0, "critical": 95.0},
        }
        self._lock = asyncio.Lock()
        self._monitoring = False
    
    async def start_monitoring(self, interval: float = 5.0):
        """Iniciar monitoreo continuo."""
        if self._monitoring:
            return
        
        self._monitoring = True
        
        async def monitor_loop():
            while self._monitoring:
                try:
                    await self.collect_metrics()
                    await asyncio.sleep(interval)
                except Exception as e:
                    logger.error(f"Error in monitoring loop: {e}")
                    await asyncio.sleep(interval)
        
        asyncio.create_task(monitor_loop())
        logger.info("Resource monitoring started")
    
    async def stop_monitoring(self):
        """Detener monitoreo."""
        self._monitoring = False
        logger.info("Resource monitoring stopped")
    
    async def collect_metrics(self):
        """Recolectar métricas del sistema."""
        if not PSUTIL_AVAILABLE:
            return
        
        now = datetime.now()
        
        # CPU
        cpu_percent = psutil.cpu_percent(interval=0.1)
        await self.record_metric(
            ResourceType.CPU,
            cpu_percent,
            unit="%",
            metadata={"per_cpu": psutil.cpu_percent(interval=0.1, percpu=True)},
        )
        
        # Memoria
        memory = psutil.virtual_memory()
        await self.record_metric(
            ResourceType.MEMORY,
            memory.percent,
            unit="%",
            metadata={
                "total": memory.total,
                "available": memory.available,
                "used": memory.used,
            },
        )
        
        # Disco
        disk = psutil.disk_usage('/')
        await self.record_metric(
            ResourceType.DISK,
            disk.percent,
            unit="%",
            metadata={
                "total": disk.total,
                "used": disk.used,
                "free": disk.free,
            },
        )
        
        # Red
        net_io = psutil.net_io_counters()
        await self.record_metric(
            ResourceType.NETWORK,
            net_io.bytes_sent + net_io.bytes_recv,
            unit="bytes",
            metadata={
                "bytes_sent": net_io.bytes_sent,
                "bytes_recv": net_io.bytes_recv,
                "packets_sent": net_io.packets_sent,
                "packets_recv": net_io.packets_recv,
            },
        )
        
        # Procesos/Threads
        process_count = len(psutil.pids())
        await self.record_metric(
            ResourceType.THREADS,
            process_count,
            unit="count",
        )
    
    async def record_metric(
        self,
        resource_type: ResourceType,
        value: float,
        unit: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Registrar métrica."""
        metric = ResourceMetric(
            resource_type=resource_type,
            value=value,
            unit=unit,
            metadata=metadata or {},
        )
        
        async with self._lock:
            self.metrics[resource_type].append(metric)
        
        # Verificar umbrales
        await self._check_thresholds(resource_type, value)
    
    async def _check_thresholds(self, resource_type: ResourceType, value: float):
        """Verificar umbrales y generar alertas."""
        thresholds = self.thresholds.get(resource_type, {})
        
        if not thresholds:
            return
        
        critical_threshold = thresholds.get("critical")
        warning_threshold = thresholds.get("warning")
        
        level = None
        if critical_threshold and value >= critical_threshold:
            level = AlertLevel.CRITICAL
        elif warning_threshold and value >= warning_threshold:
            level = AlertLevel.WARNING
        
        if level:
            alert_id = f"alert_{resource_type.value}_{datetime.now().timestamp()}"
            
            alert = ResourceAlert(
                alert_id=alert_id,
                resource_type=resource_type,
                level=level,
                message=f"{resource_type.value} usage is at {value:.1f}%",
                threshold=critical_threshold or warning_threshold,
                current_value=value,
            )
            
            async with self._lock:
                self.alerts.append(alert)
            
            logger.warning(f"Resource alert: {alert.message}")
    
    def get_metric_history(
        self,
        resource_type: ResourceType,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Obtener historial de métricas."""
        metrics = list(self.metrics.get(resource_type, []))[-limit:]
        
        return [
            {
                "value": m.value,
                "unit": m.unit,
                "timestamp": m.timestamp.isoformat(),
                "metadata": m.metadata,
            }
            for m in metrics
        ]
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Obtener métricas actuales."""
        current = {}
        
        for resource_type, metrics_deque in self.metrics.items():
            if metrics_deque:
                latest = metrics_deque[-1]
                current[resource_type.value] = {
                    "value": latest.value,
                    "unit": latest.unit,
                    "timestamp": latest.timestamp.isoformat(),
                }
        
        return current
    
    def get_metric_statistics(
        self,
        resource_type: ResourceType,
        period_minutes: int = 60,
    ) -> Dict[str, Any]:
        """Obtener estadísticas de métricas."""
        cutoff = datetime.now() - timedelta(minutes=period_minutes)
        metrics = [
            m for m in self.metrics.get(resource_type, [])
            if m.timestamp >= cutoff
        ]
        
        if not metrics:
            return {}
        
        values = [m.value for m in metrics]
        
        return {
            "resource_type": resource_type.value,
            "period_minutes": period_minutes,
            "data_points": len(values),
            "average": statistics.mean(values),
            "min": min(values),
            "max": max(values),
            "median": statistics.median(values),
            "p95": sorted(values)[int(len(values) * 0.95)] if len(values) > 20 else max(values),
        }
    
    def get_active_alerts(
        self,
        level: Optional[AlertLevel] = None,
    ) -> List[Dict[str, Any]]:
        """Obtener alertas activas."""
        alerts = [a for a in self.alerts if not a.resolved]
        
        if level:
            alerts = [a for a in alerts if a.level == level]
        
        return [
            {
                "alert_id": a.alert_id,
                "resource_type": a.resource_type.value,
                "level": a.level.value,
                "message": a.message,
                "threshold": a.threshold,
                "current_value": a.current_value,
                "timestamp": a.timestamp.isoformat(),
            }
            for a in alerts
        ]
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolver alerta."""
        for alert in self.alerts:
            if alert.alert_id == alert_id:
                alert.resolved = True
                return True
        return False
    
    def get_resource_monitor_summary(self) -> Dict[str, Any]:
        """Obtener resumen del monitor."""
        by_level: Dict[str, int] = defaultdict(int)
        
        for alert in self.alerts:
            if not alert.resolved:
                by_level[alert.level.value] += 1
        
        return {
            "monitoring_active": self._monitoring,
            "total_metrics_collected": sum(len(m) for m in self.metrics.values()),
            "active_alerts": len([a for a in self.alerts if not a.resolved]),
            "alerts_by_level": dict(by_level),
            "current_metrics": self.get_current_metrics(),
        }
















