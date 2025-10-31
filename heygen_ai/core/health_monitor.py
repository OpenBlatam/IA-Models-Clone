"""
Advanced Health Monitoring System for HeyGen AI
==============================================

Comprehensive health monitoring that tracks:
- System performance metrics
- Component health status
- Resource utilization
- Error rates and trends
- Automated health checks
- Alerting and notifications
"""

import asyncio
import time
import psutil
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path

# Import core components
from .performance_optimizer import (
    MemoryCache
)
from .config_manager import get_config
from .logger_manager import get_logger

logger = get_logger(__name__)


class HealthStatus(Enum):
    """Health status enumeration"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"
    
    def __lt__(self, other):
        if not isinstance(other, HealthStatus):
            return NotImplemented
        order = {
            HealthStatus.UNKNOWN: 0,
            HealthStatus.HEALTHY: 1,
            HealthStatus.WARNING: 2,
            HealthStatus.CRITICAL: 3
        }
        return order[self] < order[other]
    
    def __le__(self, other):
        return self < other or self == other
    
    def __gt__(self, other):
        return not self <= other
    
    def __ge__(self, other):
        return not self < other


class ComponentType(Enum):
    """Component type enumeration"""
    CACHE = "cache"
    LOAD_BALANCER = "load_balancer"
    BACKGROUND_PROCESSOR = "background_processor"
    NETWORK = "network"
    SECURITY = "security"
    EXTERNAL_API = "external_api"


@dataclass
class HealthMetric:
    """Health metric data structure"""
    component: str
    metric_name: str
    value: float
    unit: str
    timestamp: datetime
    status: HealthStatus
    threshold_warning: Optional[float] = None
    threshold_critical: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['status'] = self.status.value
        return data


@dataclass
class ComponentHealth:
    """Component health status"""
    component_id: str
    component_type: ComponentType
    status: HealthStatus
    last_check: datetime
    metrics: List[HealthMetric]
    error_count: int = 0
    warning_count: int = 0
    uptime_seconds: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['component_type'] = self.component_type.value
        data['status'] = self.status.value
        data['last_check'] = self.last_check.isoformat()
        data['metrics'] = [metric.to_dict() for metric in self.metrics]
        return data


@dataclass
class SystemHealth:
    """Overall system health status"""
    timestamp: datetime
    overall_status: HealthStatus
    components: List[ComponentHealth]
    system_metrics: List[HealthMetric]
    alerts: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['overall_status'] = self.overall_status.value
        data['components'] = [comp.to_dict() for comp in self.components]
        data['system_metrics'] = [metric.to_dict() for metric in self.system_metrics]
        return data


class HealthCheck:
    """Individual health check definition"""
    
    def __init__(
        self,
        name: str,
        check_func: Callable,
        interval_seconds: float = 60.0,
        timeout_seconds: float = 30.0,
        critical: bool = False
    ):
        self.name = name
        self.check_func = check_func
        self.interval_seconds = interval_seconds
        self.timeout_seconds = timeout_seconds
        self.critical = critical
        self.last_run: Optional[datetime] = None
        self.last_result: Optional[HealthStatus] = None
        self.error_count = 0


class HealthMonitor:
    """Advanced health monitoring system"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = get_config()
        self.components: Dict[str, ComponentHealth] = {}
        self.health_checks: List[HealthCheck] = []
        self.metrics_history: List[HealthMetric] = []
        self.alerts: List[str] = []
        self.is_running = False
        self.monitoring_task: Optional[asyncio.Task] = None
        
        # Performance thresholds from configuration
        self.thresholds = {
            'cpu_usage': {'warning': 70.0, 'critical': 90.0},
            'memory_usage': {'warning': 80.0, 'critical': 95.0},
            'disk_usage': {'warning': 85.0, 'critical': 95.0},
            'response_time': {'warning': 1000.0, 'critical': 5000.0},  # ms
            'error_rate': {'warning': 5.0, 'critical': 15.0},  # percentage
        }
        
        # Initialize default health checks only if auto_init is True
        if self.config.monitoring.enable_health_checks:
            self._initialize_default_checks()
    
    def _initialize_default_checks(self):
        """Initialize default health checks"""
        # System resource checks
        self.add_health_check(
            "system_resources",
            self._check_system_resources,
            interval_seconds=self.config.monitoring.health_check_interval
        )
        
        # Component health checks
        self.add_health_check(
            "component_health",
            self._check_component_health,
            interval_seconds=self.config.monitoring.health_check_interval * 2
        )
        
        # Performance metrics
        self.add_health_check(
            "performance_metrics",
            self._check_performance_metrics,
            interval_seconds=self.config.monitoring.health_check_interval * 1.5
        )
    
    def add_health_check(
        self,
        name: str,
        check_func: Callable,
        interval_seconds: float = 60.0,
        timeout_seconds: float = 30.0,
        critical: bool = False
    ):
        """Add a new health check"""
        health_check = HealthCheck(
            name=name,
            check_func=check_func,
            interval_seconds=interval_seconds,
            timeout_seconds=timeout_seconds,
            critical=critical
        )
        self.health_checks.append(health_check)
        logger.info(f"Added health check: {name}")
    
    def register_component(
        self,
        component_id: str,
        component_type: ComponentType,
        initial_status: HealthStatus = HealthStatus.UNKNOWN
    ):
        """Register a component for health monitoring"""
        component_health = ComponentHealth(
            component_id=component_id,
            component_type=component_type,
            status=initial_status,
            last_check=datetime.now(),
            metrics=[]
        )
        self.components[component_id] = component_health
        logger.info(f"Registered component: {component_id} ({component_type.value})")
    
    def update_component_metric(
        self,
        component_id: str,
        metric_name: str,
        value: float,
        unit: str,
        threshold_warning: Optional[float] = None,
        threshold_critical: Optional[float] = None
    ):
        """Update a component metric"""
        if component_id not in self.components:
            logger.warning(f"Component {component_id} not registered")
            return
        
        # Determine status based on thresholds
        status = HealthStatus.HEALTHY
        if threshold_critical and value >= threshold_critical:
            status = HealthStatus.CRITICAL
        elif threshold_warning and value >= threshold_warning:
            status = HealthStatus.WARNING
        
        # Create metric
        metric = HealthMetric(
            component=component_id,
            metric_name=metric_name,
            value=value,
            unit=unit,
            timestamp=datetime.now(),
            status=status,
            threshold_warning=threshold_warning,
            threshold_critical=threshold_critical
        )
        
        # Update component
        component = self.components[component_id]
        component.metrics.append(metric)
        component.last_check = datetime.now()
        
        # Update component status based on worst metric status
        worst_status = max(metric.status for metric in component.metrics)
        component.status = worst_status
        
        # Count warnings and errors
        component.warning_count = sum(1 for m in component.metrics if m.status == HealthStatus.WARNING)
        component.error_count = sum(1 for m in component.metrics if m.status == HealthStatus.CRITICAL)
        
        # Store in history
        self.metrics_history.append(metric)
        
        # Keep only last 1000 metrics
        if len(self.metrics_history) > 1000:
            self.metrics_history = self.metrics_history[-1000:]
        
        logger.debug(f"Updated metric {metric_name} for {component_id}: {value} {unit}")
    
    async def _check_system_resources(self) -> HealthStatus:
        """Check system resource usage"""
        try:
            start_time = time.time()
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1.0)
            self._add_system_metric(
                "cpu_usage",
                cpu_percent,
                "%",
                self.thresholds['cpu_usage']['warning'],
                self.thresholds['cpu_usage']['critical']
            )
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            self._add_system_metric(
                "memory_usage",
                memory_percent,
                "%",
                self.thresholds['memory_usage']['warning'],
                self.thresholds['memory_usage']['critical']
            )
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            self._add_system_metric(
                "disk_usage",
                disk_percent,
                "%",
                self.thresholds['disk_usage']['warning'],
                self.thresholds['disk_usage']['critical']
            )
            
            # Network I/O
            network = psutil.net_io_counters()
            self._add_system_metric("network_bytes_sent", network.bytes_sent, "bytes")
            self._add_system_metric("network_bytes_recv", network.bytes_recv, "bytes")
            
            duration = time.time() - start_time
            logger.debug(f"System resources check completed in {duration:.3f}s")
            
            return HealthStatus.HEALTHY
            
        except Exception as e:
            logger.error(f"Error checking system resources: {e}")
            return HealthStatus.CRITICAL
    
    async def _check_component_health(self) -> HealthStatus:
        """Check component health status"""
        try:
            start_time = time.time()
            overall_status = HealthStatus.HEALTHY
            
            for component_id, component in self.components.items():
                # Check if component has recent metrics
                if not component.metrics:
                    component.status = HealthStatus.UNKNOWN
                    continue
                
                # Check last metric timestamp
                last_metric = max(component.metrics, key=lambda m: m.timestamp)
                time_since_last = (datetime.now() - last_metric.timestamp).total_seconds()
                
                # Mark as warning if no recent metrics
                if time_since_last > 300:  # 5 minutes
                    component.status = HealthStatus.WARNING
                    self.alerts.append(f"Component {component_id} has no recent metrics")
                
                # Update overall status
                if component.status == HealthStatus.CRITICAL:
                    overall_status = HealthStatus.CRITICAL
                elif component.status == HealthStatus.WARNING and overall_status == HealthStatus.HEALTHY:
                    overall_status = HealthStatus.WARNING
            
            duration = time.time() - start_time
            logger.debug(f"Component health check completed in {duration:.3f}s")
            
            return overall_status
            
        except Exception as e:
            logger.error(f"Error checking component health: {e}")
            return HealthStatus.CRITICAL
    
    async def _check_performance_metrics(self) -> HealthStatus:
        """Check performance metrics"""
        try:
            start_time = time.time()
            
            # Check response time trends
            recent_metrics = [
                m for m in self.metrics_history
                if m.metric_name == "response_time" and
                (datetime.now() - m.timestamp).total_seconds() < 300
            ]
            
            if recent_metrics:
                avg_response_time = sum(m.value for m in recent_metrics) / len(recent_metrics)
                self._add_system_metric(
                    "avg_response_time",
                    avg_response_time,
                    "ms",
                    self.thresholds['response_time']['warning'],
                    self.thresholds['response_time']['critical']
                )
            
            # Check error rates
            recent_errors = [
                m for m in self.metrics_history
                if m.status == HealthStatus.CRITICAL and
                (datetime.now() - m.timestamp).total_seconds() < 300
            ]
            
            if recent_metrics:
                error_rate = (len(recent_errors) / len(recent_metrics)) * 100
                self._add_system_metric(
                    "error_rate",
                    error_rate,
                    "%",
                    self.thresholds['error_rate']['warning'],
                    self.thresholds['error_rate']['critical']
                )
            
            duration = time.time() - start_time
            logger.debug(f"Performance metrics check completed in {duration:.3f}s")
            
            return HealthStatus.HEALTHY
            
        except Exception as e:
            logger.error(f"Error checking performance metrics: {e}")
            return HealthStatus.CRITICAL
    
    def _add_system_metric(
        self,
        metric_name: str,
        value: float,
        unit: str,
        threshold_warning: Optional[float] = None,
        threshold_critical: Optional[float] = None
    ):
        """Add a system-level metric"""
        metric = HealthMetric(
            component="system",
            metric_name=metric_name,
            value=value,
            unit=unit,
            timestamp=datetime.now(),
            status=HealthStatus.HEALTHY,
            threshold_warning=threshold_warning,
            threshold_critical=threshold_critical
        )
        
        # Determine status
        if threshold_critical and value >= threshold_critical:
            metric.status = HealthStatus.CRITICAL
            self.alerts.append(f"Critical threshold exceeded: {metric_name} = {value} {unit}")
        elif threshold_warning and value >= threshold_warning:
            metric.status = HealthStatus.WARNING
            self.alerts.append(f"Warning threshold exceeded: {metric_name} = {value} {unit}")
        
        self.metrics_history.append(metric)
    
    async def start_monitoring(self):
        """Start the health monitoring system"""
        if self.is_running:
            logger.warning("Health monitoring already running")
            return
        
        self.is_running = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Health monitoring started")
    
    async def stop_monitoring(self):
        """Stop the health monitoring system"""
        if not self.is_running:
            return
        
        self.is_running = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Health monitoring stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.is_running:
            try:
                # Run health checks
                for health_check in self.health_checks:
                    if (health_check.last_run is None or
                        (datetime.now() - health_check.last_run).total_seconds() >= health_check.interval_seconds):
                        
                        # Run check with timeout
                        try:
                            result = await asyncio.wait_for(
                                health_check.check_func(),
                                timeout=health_check.timeout_seconds
                            )
                            health_check.last_result = result
                            health_check.error_count = 0
                        except asyncio.TimeoutError:
                            logger.warning(f"Health check {health_check.name} timed out")
                            result = HealthStatus.WARNING
                            health_check.last_result = result
                        except Exception as e:
                            logger.error(f"Health check {health_check.name} failed: {e}")
                            result = HealthStatus.CRITICAL
                            health_check.last_result = result
                            health_check.error_count += 1
                        
                        health_check.last_run = datetime.now()
                        
                        # Handle critical failures
                        if health_check.critical and result == HealthStatus.CRITICAL:
                            self.alerts.append(f"Critical health check failed: {health_check.name}")
                
                # Wait before next iteration
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(30)  # Wait longer on error
    
    def get_system_health(self) -> SystemHealth:
        """Get current system health status"""
        # Determine overall status
        component_statuses = [comp.status for comp in self.components.values()]
        system_metric_statuses = [
            m.status for m in self.metrics_history
            if m.component == "system" and
            (datetime.now() - m.timestamp).total_seconds() < 300
        ]
        
        all_statuses = component_statuses + system_metric_statuses
        
        if HealthStatus.CRITICAL in all_statuses:
            overall_status = HealthStatus.CRITICAL
        elif HealthStatus.WARNING in all_statuses:
            overall_status = HealthStatus.WARNING
        elif all_statuses:
            overall_status = HealthStatus.HEALTHY
        else:
            overall_status = HealthStatus.UNKNOWN
        
        # Get recent system metrics
        recent_system_metrics = [
            m for m in self.metrics_history
            if m.component == "system" and
            (datetime.now() - m.timestamp).total_seconds() < 300
        ]
        
        return SystemHealth(
            timestamp=datetime.now(),
            overall_status=overall_status,
            components=list(self.components.values()),
            system_metrics=recent_system_metrics,
            alerts=self.alerts[-50:]  # Last 50 alerts
        )
    
    def export_health_data(self, output_path: Optional[Path] = None) -> Dict[str, Any]:
        """Export health data to JSON"""
        if output_path is None:
            output_path = Path("health_report.json")
        
        health_data = {
            'system_health': self.get_system_health().to_dict(),
            'metrics_history': [m.to_dict() for m in self.metrics_history[-1000:]],
            'health_checks': [
                {
                    'name': check.name,
                    'last_run': check.last_run.isoformat() if check.last_run else None,
                    'last_result': check.last_result.value if check.last_result else None,
                    'error_count': check.error_count
                }
                for check in self.health_checks
            ],
            'export_timestamp': datetime.now().isoformat()
        }
        
        # Save to file
        with open(output_path, 'w') as f:
            json.dump(health_data, f, indent=2, default=str)
        
        logger.info(f"Health data exported to {output_path}")
        return health_data
    
    def clear_alerts(self):
        """Clear all alerts"""
        self.alerts.clear()
        logger.info("All alerts cleared")
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get summary of current alerts"""
        critical_alerts = [alert for alert in self.alerts if "Critical" in alert]
        warning_alerts = [alert for alert in self.alerts if "Warning" in alert]
        
        return {
            'total_alerts': len(self.alerts),
            'critical_alerts': len(critical_alerts),
            'warning_alerts': len(warning_alerts),
            'recent_alerts': self.alerts[-10:],  # Last 10 alerts
            'timestamp': datetime.now().isoformat()
        }


# Convenience functions for quick health checks
async def quick_health_check() -> Dict[str, Any]:
    """Quick health check for the system"""
    monitor = HealthMonitor()
    
    # Register some basic components
    monitor.register_component("system", ComponentType.NETWORK)
    monitor.register_component("cache", ComponentType.CACHE)
    monitor.register_component("api", ComponentType.EXTERNAL_API)
    
    # Run a quick check
    await monitor._check_system_resources()
    
    # Get health status
    health = monitor.get_system_health()
    
    return {
        'status': health.overall_status.value,
        'components': len(health.components),
        'metrics': len(health.system_metrics),
        'alerts': len(health.alerts)
    }


if __name__ == "__main__":
    # Example usage
    async def main():
        monitor = HealthMonitor()
        
        # Register components
        monitor.register_component("test_cache", ComponentType.CACHE)
        monitor.register_component("test_lb", ComponentType.LOAD_BALANCER)
        
        # Start monitoring
        await monitor.start_monitoring()
        
        # Simulate some metrics
        monitor.update_component_metric("test_cache", "hit_rate", 95.5, "%", 80.0, 90.0)
        monitor.update_component_metric("test_lb", "response_time", 150.0, "ms", 200.0, 500.0)
        
        # Wait a bit
        await asyncio.sleep(5)
        
        # Get health status
        health = monitor.get_system_health()
        print(f"System Health: {health.overall_status.value}")
        
        # Stop monitoring
        await monitor.stop_monitoring()
    
    asyncio.run(main())
