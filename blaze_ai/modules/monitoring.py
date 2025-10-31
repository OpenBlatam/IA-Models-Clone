"""
Monitoring Module for Blaze AI

Provides comprehensive system monitoring, metrics collection, and health tracking
as a modular component that can be used independently or as part of the system.
"""

import asyncio
import logging
import time
import json
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Callable, Union
from pathlib import Path

from .base import BaseModule, ModuleConfig, ModuleType, ModulePriority, ModuleStatus, HealthStatus

logger = logging.getLogger(__name__)

# ============================================================================
# MONITORING-SPECIFIC ENUMS AND CONSTANTS
# ============================================================================

class MetricType(Enum):
    """Types of metrics that can be collected."""
    COUNTER = auto()        # Incremental counter
    GAUGE = auto()          # Current value
    HISTOGRAM = auto()      # Distribution of values
    SUMMARY = auto()        # Statistical summary

class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = auto()
    WARNING = auto()
    ERROR = auto()
    CRITICAL = auto()

class MonitoringMode(Enum):
    """Monitoring operation modes."""
    PASSIVE = auto()        # Collect metrics when requested
    ACTIVE = auto()         # Actively collect metrics
    AGGRESSIVE = auto()     # High-frequency collection

# Default constants
DEFAULT_COLLECTION_INTERVAL = 10.0  # 10 seconds
DEFAULT_RETENTION_PERIOD = 86400    # 24 hours
DEFAULT_MAX_METRICS = 10000

# ============================================================================
# MONITORING-SPECIFIC DATACLASSES
# ============================================================================

@dataclass
class MonitoringConfig(ModuleConfig):
    """Configuration for monitoring modules."""
    collection_interval: float = DEFAULT_COLLECTION_INTERVAL
    retention_period: float = DEFAULT_RETENTION_PERIOD
    max_metrics: int = DEFAULT_MAX_METRICS
    monitoring_mode: MonitoringMode = MonitoringMode.ACTIVE
    enable_alerts: bool = True
    enable_persistence: bool = False
    persistence_path: Optional[str] = None
    alert_thresholds: Dict[str, float] = field(default_factory=dict)
    custom_metrics: Set[str] = field(default_factory=set)
    
    def __post_init__(self):
        """Set module type automatically."""
        self.module_type = ModuleType.MONITORING

@dataclass
class Metric:
    """Individual metric data point."""
    name: str
    value: Union[int, float, str]
    metric_type: MetricType
    timestamp: float = field(default_factory=time.time)
    labels: Dict[str, str] = field(default_factory=dict)
    description: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metric to dictionary."""
        return {
            "name": self.name,
            "value": self.value,
            "type": self.metric_type.name,
            "timestamp": self.timestamp,
            "labels": self.labels,
            "description": self.description
        }

@dataclass
class Alert:
    """System alert information."""
    id: str
    level: AlertLevel
    message: str
    timestamp: float = field(default_factory=time.time)
    source: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    acknowledged: bool = False
    resolved: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary."""
        return {
            "id": self.id,
            "level": self.level.name,
            "message": self.message,
            "timestamp": self.timestamp,
            "source": self.source,
            "details": self.details,
            "acknowledged": self.acknowledged,
            "resolved": self.resolved
        }

@dataclass
class SystemMetrics:
    """System-wide performance metrics."""
    timestamp: float = field(default_factory=time.time)
    cpu_usage_percent: float = 0.0
    memory_usage_mb: float = 0.0
    memory_available_mb: float = 0.0
    disk_usage_percent: float = 0.0
    network_io_mbps: float = 0.0
    active_connections: int = 0
    active_processes: int = 0
    system_load_1m: float = 0.0
    system_load_5m: float = 0.0
    system_load_15m: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp,
            "cpu_usage_percent": self.cpu_usage_percent,
            "memory_usage_mb": self.memory_usage_mb,
            "memory_available_mb": self.memory_available_mb,
            "disk_usage_percent": self.disk_usage_percent,
            "network_io_mbps": self.network_io_mbps,
            "active_connections": self.active_connections,
            "active_processes": self.active_processes,
            "system_load_1m": self.system_load_1m,
            "system_load_5m": self.system_load_5m,
            "system_load_15m": self.system_load_15m
        }

# ============================================================================
# METRIC COLLECTORS
# ============================================================================

class MetricCollector:
    """Base class for metric collectors."""
    
    def __init__(self, name: str):
        self.name = name
        self.metrics: List[Metric] = []
    
    async def collect(self) -> List[Metric]:
        """Collect metrics. Override in subclasses."""
        return []
    
    def add_metric(self, metric: Metric):
        """Add a metric to the collector."""
        self.metrics.append(metric)
    
    def get_metrics(self, limit: Optional[int] = None) -> List[Metric]:
        """Get collected metrics."""
        if limit is None:
            return self.metrics.copy()
        return self.metrics[-limit:]

class SystemMetricCollector(MetricCollector):
    """Collects system-level metrics."""
    
    def __init__(self):
        super().__init__("system")
    
    async def collect(self) -> List[Metric]:
        """Collect system metrics."""
        metrics = []
        current_time = time.time()
        
        try:
            # CPU usage
            try:
                import psutil
                cpu_percent = psutil.cpu_percent(interval=0.1)
                metrics.append(Metric(
                    name="cpu_usage_percent",
                    value=cpu_percent,
                    metric_type=MetricType.GAUGE,
                    timestamp=current_time,
                    labels={"collector": "system"}
                ))
            except ImportError:
                pass
            
            # Memory usage
            try:
                import psutil
                memory = psutil.virtual_memory()
                metrics.append(Metric(
                    name="memory_usage_mb",
                    value=memory.used / (1024 * 1024),
                    metric_type=MetricType.GAUGE,
                    timestamp=current_time,
                    labels={"collector": "system"}
                ))
                metrics.append(Metric(
                    name="memory_available_mb",
                    value=memory.available / (1024 * 1024),
                    metric_type=MetricType.GAUGE,
                    timestamp=current_time,
                    labels={"collector": "system"}
                ))
            except ImportError:
                pass
            
            # Disk usage
            try:
                import psutil
                disk = psutil.disk_usage('/')
                metrics.append(Metric(
                    name="disk_usage_percent",
                    value=disk.percent,
                    metric_type=MetricType.GAUGE,
                    timestamp=current_time,
                    labels={"collector": "system"}
                ))
            except ImportError:
                pass
            
            # Process count
            try:
                import psutil
                process_count = len(psutil.pids())
                metrics.append(Metric(
                    name="active_processes",
                    value=process_count,
                    metric_type=MetricType.GAUGE,
                    timestamp=current_time,
                    labels={"collector": "system"}
                ))
            except ImportError:
                pass
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
        
        return metrics

class CustomMetricCollector(MetricCollector):
    """Collects custom metrics from registered functions."""
    
    def __init__(self):
        super().__init__("custom")
        self.collectors: Dict[str, Callable] = {}
    
    def register_collector(self, name: str, collector_func: Callable):
        """Register a custom metric collector function."""
        self.collectors[name] = collector_func
    
    def unregister_collector(self, name: str):
        """Unregister a custom metric collector."""
        if name in self.collectors:
            del self.collectors[name]
    
    async def collect(self) -> List[Metric]:
        """Collect custom metrics."""
        metrics = []
        current_time = time.time()
        
        for name, collector_func in self.collectors.items():
            try:
                if asyncio.iscoroutinefunction(collector_func):
                    result = await collector_func()
                else:
                    result = collector_func()
                
                if isinstance(result, (int, float, str)):
                    metrics.append(Metric(
                        name=f"custom_{name}",
                        value=result,
                        metric_type=MetricType.GAUGE,
                        timestamp=current_time,
                        labels={"collector": "custom", "source": name}
                    ))
                elif isinstance(result, list):
                    for item in result:
                        if isinstance(item, Metric):
                            metrics.append(item)
                        else:
                            logger.warning(f"Custom collector {name} returned invalid metric: {item}")
                            
            except Exception as e:
                logger.error(f"Error in custom collector {name}: {e}")
        
        return metrics

# ============================================================================
# ALERT MANAGER
# ============================================================================

class AlertManager:
    """Manages system alerts and notifications."""
    
    def __init__(self):
        self.alerts: Dict[str, Alert] = {}
        self.alert_handlers: List[Callable] = []
        self.next_alert_id = 1
    
    def add_alert(self, level: AlertLevel, message: str, source: Optional[str] = None, **details) -> str:
        """Add a new alert."""
        alert_id = f"alert_{self.next_alert_id}"
        self.next_alert_id += 1
        
        alert = Alert(
            id=alert_id,
            level=level,
            message=message,
            source=source,
            details=details
        )
        
        self.alerts[alert_id] = alert
        
        # Notify handlers
        for handler in self.alert_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    asyncio.create_task(handler(alert))
                else:
                    handler(alert)
            except Exception as e:
                logger.error(f"Error in alert handler: {e}")
        
        return alert_id
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert."""
        if alert_id in self.alerts:
            self.alerts[alert_id].acknowledged = True
            return True
        return False
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Mark an alert as resolved."""
        if alert_id in self.alerts:
            self.alerts[alert_id].resolved = True
            return True
        return False
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active (unresolved) alerts."""
        return [alert for alert in self.alerts.values() if not alert.resolved]
    
    def get_alerts_by_level(self, level: AlertLevel) -> List[Alert]:
        """Get alerts by severity level."""
        return [alert for alert in self.alerts.values() if alert.level == level]
    
    def add_handler(self, handler: Callable):
        """Add an alert handler function."""
        self.alert_handlers.append(handler)
    
    def remove_handler(self, handler: Callable):
        """Remove an alert handler function."""
        if handler in self.alert_handlers:
            self.alert_handlers.remove(handler)

# ============================================================================
# MAIN MONITORING MODULE
# ============================================================================

class MonitoringModule(BaseModule):
    """Modular monitoring system for Blaze AI."""
    
    def __init__(self, config: MonitoringConfig):
        super().__init__(config)
        self.monitoring_config = config
        
        # Metric storage
        self.metrics: Dict[str, List[Metric]] = {}
        self.system_metrics: List[SystemMetrics] = []
        
        # Collectors
        self.system_collector = SystemMetricCollector()
        self.custom_collector = CustomMetricCollector()
        
        # Alert management
        self.alert_manager = AlertManager()
        
        # Background tasks
        self._collection_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        
        # Threshold monitoring
        self._threshold_checkers: Dict[str, Callable] = {}
        
        # Setup default threshold checkers
        self._setup_default_thresholds()
    
    def _setup_default_thresholds(self):
        """Setup default threshold monitoring."""
        # CPU usage threshold
        self._threshold_checkers["cpu_usage"] = lambda value: (
            AlertLevel.WARNING if value > 80 else
            AlertLevel.ERROR if value > 95 else
            None
        )
        
        # Memory usage threshold
        self._threshold_checkers["memory_usage"] = lambda value: (
            AlertLevel.WARNING if value > 80 else
            AlertLevel.ERROR if value > 95 else
            None
        )
        
        # Disk usage threshold
        self._threshold_checkers["disk_usage"] = lambda value: (
            AlertLevel.WARNING if value > 85 else
            AlertLevel.ERROR if value > 95 else
            None
        )
    
    # ============================================================================
    # ABSTRACT METHOD IMPLEMENTATIONS
    # ============================================================================
    
    async def _initialize_impl(self) -> bool:
        """Initialize the monitoring module."""
        try:
            self.logger.info(f"Initializing monitoring module: {self.config.name}")
            self.logger.info(f"Collection interval: {self.monitoring_config.collection_interval}s")
            self.logger.info(f"Monitoring mode: {self.monitoring_config.monitoring_mode.name}")
            
            # Start background tasks
            if self.monitoring_config.monitoring_mode != MonitoringMode.PASSIVE:
                self._start_collection_task()
            
            self._start_cleanup_task()
            
            return True
        except Exception as e:
            self.logger.error(f"Monitoring initialization failed: {e}")
            return False
    
    async def _shutdown_impl(self) -> bool:
        """Shutdown the monitoring module."""
        try:
            # Stop background tasks
            if self._collection_task:
                self._collection_task.cancel()
            
            if self._cleanup_task:
                self._cleanup_task.cancel()
            
            # Save final metrics if persistence is enabled
            if self.monitoring_config.enable_persistence:
                await self._persist_metrics()
            
            self.logger.info(f"Monitoring module shutdown completed: {self.config.name}")
            return True
        except Exception as e:
            self.logger.error(f"Monitoring shutdown failed: {e}")
            return False
    
    async def _health_check_impl(self) -> HealthStatus:
        """Perform monitoring-specific health check."""
        try:
            # Check if metrics are being collected
            total_metrics = sum(len(metrics) for metrics in self.metrics.values())
            active_alerts = len(self.alert_manager.get_active_alerts())
            
            if total_metrics == 0:
                message = "No metrics collected"
                status = ModuleStatus.IDLE
            elif active_alerts > 0:
                message = f"Active monitoring with {active_alerts} alerts"
                status = ModuleStatus.ACTIVE
            else:
                message = f"Active monitoring: {total_metrics} metrics"
                status = ModuleStatus.ACTIVE
            
            return HealthStatus(
                status=status,
                message=message,
                details={
                    "total_metrics": total_metrics,
                    "active_alerts": active_alerts,
                    "collection_interval": self.monitoring_config.collection_interval,
                    "monitoring_mode": self.monitoring_config.monitoring_mode.name
                }
            )
            
        except Exception as e:
            return HealthStatus(
                status=ModuleStatus.ERROR,
                message=f"Health check failed: {str(e)}",
                error=str(e)
            )
    
    # ============================================================================
    # BACKGROUND TASKS
    # ============================================================================
    
    def _start_collection_task(self):
        """Start metric collection task."""
        async def collection_loop():
            while self.status not in [ModuleStatus.SHUTDOWN, ModuleStatus.ERROR]:
                try:
                    await asyncio.sleep(self.monitoring_config.collection_interval)
                    await self._collect_all_metrics()
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    self.logger.error(f"Metric collection error: {e}")
                    await asyncio.sleep(5.0)
        
        self._collection_task = asyncio.create_task(collection_loop())
    
    def _start_cleanup_task(self):
        """Start cleanup task for old metrics."""
        async def cleanup_loop():
            while self.status not in [ModuleStatus.SHUTDOWN, ModuleStatus.ERROR]:
                try:
                    await asyncio.sleep(60.0)  # Cleanup every minute
                    await self._cleanup_old_metrics()
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    self.logger.error(f"Cleanup error: {e}")
                    await asyncio.sleep(5.0)
        
        self._cleanup_task = asyncio.create_task(cleanup_loop())
    
    # ============================================================================
    # METRIC COLLECTION
    # ============================================================================
    
    async def _collect_all_metrics(self):
        """Collect all metrics from all collectors."""
        try:
            # Collect system metrics
            system_metrics = await self.system_collector.collect()
            self._store_metrics("system", system_metrics)
            
            # Collect custom metrics
            custom_metrics = await self.custom_collector.collect()
            self._store_metrics("custom", custom_metrics)
            
            # Check thresholds and generate alerts
            await self._check_thresholds()
            
            # Record operation
            self.record_operation(True)
            
        except Exception as e:
            self.logger.error(f"Error collecting metrics: {e}")
            self.record_operation(False)
    
    def _store_metrics(self, source: str, metrics: List[Metric]):
        """Store metrics from a specific source."""
        if source not in self.metrics:
            self.metrics[source] = []
        
        self.metrics[source].extend(metrics)
        
        # Limit metrics per source
        if len(self.metrics[source]) > self.monitoring_config.max_metrics:
            self.metrics[source] = self.metrics[source][-self.monitoring_config.max_metrics:]
    
    async def _check_thresholds(self):
        """Check metrics against thresholds and generate alerts."""
        if not self.monitoring_config.enable_alerts:
            return
        
        for source, metrics in self.metrics.items():
            for metric in metrics:
                if metric.name in self._threshold_checkers:
                    threshold_func = self._threshold_checkers[metric.name]
                    try:
                        alert_level = threshold_func(metric.value)
                        if alert_level:
                            self.alert_manager.add_alert(
                                level=alert_level,
                                message=f"Metric {metric.name} exceeded threshold: {metric.value}",
                                source=source,
                                metric_name=metric.name,
                                metric_value=metric.value,
                                threshold_level=alert_level.name
                            )
                    except Exception as e:
                        logger.error(f"Error checking threshold for {metric.name}: {e}")
    
    async def _cleanup_old_metrics(self):
        """Remove metrics older than retention period."""
        current_time = time.time()
        cutoff_time = current_time - self.monitoring_config.retention_period
        
        for source in list(self.metrics.keys()):
            self.metrics[source] = [
                metric for metric in self.metrics[source]
                if metric.timestamp > cutoff_time
            ]
    
    # ============================================================================
    # PUBLIC INTERFACE
    # ============================================================================
    
    async def collect_metrics_now(self) -> Dict[str, List[Metric]]:
        """Manually trigger metric collection."""
        await self._collect_all_metrics()
        return self.metrics.copy()
    
    def get_metrics(self, source: Optional[str] = None, limit: Optional[int] = None) -> Dict[str, List[Metric]]:
        """Get collected metrics."""
        if source:
            if source in self.metrics:
                return {source: self.metrics[source][-limit:] if limit else self.metrics[source].copy()}
            return {}
        
        if limit:
            return {
                source: metrics[-limit:] if metrics else []
                for source, metrics in self.metrics.items()
            }
        
        return {source: metrics.copy() for source, metrics in self.metrics.items()}
    
    def get_metric_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics."""
        summary = {}
        
        for source, metrics in self.metrics.items():
            if not metrics:
                continue
            
            # Group by metric name
            metric_groups = {}
            for metric in metrics:
                if metric.name not in metric_groups:
                    metric_groups[metric.name] = []
                metric_groups[metric.name].append(metric)
            
            # Calculate statistics for each metric
            source_summary = {}
            for metric_name, metric_list in metric_groups.items():
                values = [m.value for m in metric_list if isinstance(m.value, (int, float))]
                if values:
                    source_summary[metric_name] = {
                        "count": len(values),
                        "min": min(values),
                        "max": max(values),
                        "avg": sum(values) / len(values),
                        "latest": values[-1],
                        "latest_timestamp": metric_list[-1].timestamp
                    }
            
            summary[source] = source_summary
        
        return summary
    
    def register_custom_collector(self, name: str, collector_func: Callable):
        """Register a custom metric collector."""
        self.custom_collector.register_collector(name, collector_func)
        self.logger.info(f"Registered custom metric collector: {name}")
    
    def unregister_custom_collector(self, name: str):
        """Unregister a custom metric collector."""
        self.custom_collector.unregister_collector(name)
        self.logger.info(f"Unregistered custom metric collector: {name}")
    
    def add_threshold_checker(self, metric_name: str, checker_func: Callable):
        """Add a custom threshold checker for a metric."""
        self._threshold_checkers[metric_name] = checker_func
        self.logger.info(f"Added threshold checker for metric: {metric_name}")
    
    def get_alerts(self, level: Optional[AlertLevel] = None, active_only: bool = True) -> List[Alert]:
        """Get system alerts."""
        if level:
            alerts = self.alert_manager.get_alerts_by_level(level)
        else:
            alerts = list(self.alert_manager.alerts.values())
        
        if active_only:
            alerts = [alert for alert in alerts if not alert.resolved]
        
        return sorted(alerts, key=lambda x: x.timestamp, reverse=True)
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert."""
        return self.alert_manager.acknowledge_alert(alert_id)
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Mark an alert as resolved."""
        return self.alert_manager.resolve_alert(alert_id)
    
    def add_alert_handler(self, handler: Callable):
        """Add an alert handler function."""
        self.alert_manager.add_handler(handler)
    
    # ============================================================================
    # PERSISTENCE
    # ============================================================================
    
    async def _persist_metrics(self):
        """Persist metrics to storage if enabled."""
        if not self.monitoring_config.enable_persistence or not self.monitoring_config.persistence_path:
            return
        
        try:
            persistence_path = Path(self.monitoring_config.persistence_path)
            persistence_path.mkdir(parents=True, exist_ok=True)
            
            # Save metrics
            metrics_file = persistence_path / f"metrics_{int(time.time())}.json"
            with open(metrics_file, 'w') as f:
                json.dump({
                    source: [metric.to_dict() for metric in metrics]
                    for source, metrics in self.metrics.items()
                }, f, indent=2)
            
            # Save alerts
            alerts_file = persistence_path / f"alerts_{int(time.time())}.json"
            with open(alerts_file, 'w') as f:
                json.dump([
                    alert.to_dict() for alert in self.alert_manager.alerts.values()
                ], f, indent=2)
            
            self.logger.info(f"Metrics persisted to {persistence_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to persist metrics: {e}")

# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

def create_monitoring_module(
    name: str = "monitoring",
    collection_interval: float = DEFAULT_COLLECTION_INTERVAL,
    monitoring_mode: MonitoringMode = MonitoringMode.ACTIVE,
    priority: ModulePriority = ModulePriority.HIGH
) -> MonitoringModule:
    """Create a new monitoring module."""
    config = MonitoringConfig(
        name=name,
        priority=priority,
        collection_interval=collection_interval,
        monitoring_mode=monitoring_mode
    )
    return MonitoringModule(config)

def create_passive_monitoring(name: str = "passive_monitoring") -> MonitoringModule:
    """Create a passive monitoring module."""
    return create_monitoring_module(
        name=name,
        collection_interval=60.0,  # 1 minute
        monitoring_mode=MonitoringMode.PASSIVE,
        priority=ModulePriority.LOW
    )

def create_aggressive_monitoring(name: str = "aggressive_monitoring") -> MonitoringModule:
    """Create an aggressive monitoring module."""
    return create_monitoring_module(
        name=name,
        collection_interval=1.0,  # 1 second
        monitoring_mode=MonitoringMode.AGGRESSIVE,
        priority=ModulePriority.CRITICAL
    )
