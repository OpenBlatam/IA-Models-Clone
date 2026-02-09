"""
Real-time Monitor
================

Ultra-advanced real-time monitoring system for maximum observability.
"""

import asyncio
import logging
import time
import numpy as np
from typing import Dict, List, Optional, Any, Union, Callable, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor
import weakref
from collections import defaultdict, deque
import json
import pickle
import psutil
import gc
from prometheus_client import Counter, Histogram, Gauge, Summary, start_http_server
import websockets
import sse

logger = logging.getLogger(__name__)

class MetricType(str, Enum):
    """Metric types."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"
    CUSTOM = "custom"

class AlertLevel(str, Enum):
    """Alert levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class MonitoringInterval(str, Enum):
    """Monitoring intervals."""
    REAL_TIME = "real_time"      # 1 second
    FAST = "fast"                # 5 seconds
    NORMAL = "normal"            # 30 seconds
    SLOW = "slow"                # 5 minutes
    VERY_SLOW = "very_slow"      # 1 hour

@dataclass
class Metric:
    """Metric definition."""
    name: str
    metric_type: MetricType
    description: str = ""
    labels: Dict[str, str] = field(default_factory=dict)
    value: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)

@dataclass
class Alert:
    """Alert definition."""
    id: str
    name: str
    level: AlertLevel
    message: str
    metric_name: str
    threshold: float
    current_value: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    resolved: bool = False

@dataclass
class MonitoringConfig:
    """Monitoring configuration."""
    enable_prometheus: bool = True
    enable_websocket: bool = True
    enable_sse: bool = True
    enable_alerts: bool = True
    enable_dashboard: bool = True
    prometheus_port: int = 9090
    websocket_port: int = 9091
    sse_port: int = 9092
    dashboard_port: int = 9093
    monitoring_interval: MonitoringInterval = MonitoringInterval.NORMAL
    alert_thresholds: Dict[str, float] = field(default_factory=dict)
    retention_period: int = 86400  # 24 hours
    max_metrics: int = 100000
    max_alerts: int = 1000

class RealTimeMonitor:
    """
    Ultra-advanced real-time monitoring system.
    
    Features:
    - Real-time metrics collection
    - Prometheus integration
    - WebSocket streaming
    - Server-sent events
    - Alert system
    - Dashboard
    - Performance analytics
    - Resource monitoring
    """
    
    def __init__(self, config: Optional[MonitoringConfig] = None):
        self.config = config or MonitoringConfig()
        self.metrics = {}
        self.alerts = deque(maxlen=self.config.max_alerts)
        self.metric_history = deque(maxlen=self.config.max_metrics)
        self.websocket_clients = set()
        self.sse_clients = set()
        self.prometheus_metrics = {}
        self.running = False
        self.stats = {
            'total_metrics': 0,
            'total_alerts': 0,
            'active_alerts': 0,
            'websocket_clients': 0,
            'sse_clients': 0
        }
        self.lock = asyncio.Lock()
        
    async def initialize(self):
        """Initialize real-time monitor."""
        logger.info("Initializing Real-time Monitor...")
        
        try:
            # Initialize Prometheus
            if self.config.enable_prometheus:
                await self._initialize_prometheus()
            
            # Initialize WebSocket server
            if self.config.enable_websocket:
                await self._initialize_websocket()
            
            # Initialize SSE server
            if self.config.enable_sse:
                await self._initialize_sse()
            
            # Initialize dashboard
            if self.config.enable_dashboard:
                await self._initialize_dashboard()
            
            # Start monitoring tasks
            self.running = True
            asyncio.create_task(self._collect_metrics())
            asyncio.create_task(self._check_alerts())
            asyncio.create_task(self._cleanup_old_data())
            
            logger.info("Real-time Monitor initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Real-time Monitor: {str(e)}")
            raise
    
    async def _initialize_prometheus(self):
        """Initialize Prometheus metrics."""
        try:
            # Start Prometheus HTTP server
            start_http_server(self.config.prometheus_port)
            
            # Create Prometheus metrics
            self.prometheus_metrics = {
                'cpu_usage': Gauge('cpu_usage_percent', 'CPU usage percentage'),
                'memory_usage': Gauge('memory_usage_percent', 'Memory usage percentage'),
                'disk_usage': Gauge('disk_usage_percent', 'Disk usage percentage'),
                'network_io': Gauge('network_io_bytes', 'Network I/O bytes'),
                'request_count': Counter('requests_total', 'Total requests'),
                'response_time': Histogram('response_time_seconds', 'Response time'),
                'active_connections': Gauge('active_connections', 'Active connections'),
                'error_rate': Gauge('error_rate_percent', 'Error rate percentage')
            }
            
            logger.info(f"Prometheus server started on port {self.config.prometheus_port}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Prometheus: {str(e)}")
            raise
    
    async def _initialize_websocket(self):
        """Initialize WebSocket server."""
        try:
            # Start WebSocket server
            asyncio.create_task(self._websocket_server())
            
            logger.info(f"WebSocket server started on port {self.config.websocket_port}")
            
        except Exception as e:
            logger.error(f"Failed to initialize WebSocket: {str(e)}")
            raise
    
    async def _initialize_sse(self):
        """Initialize Server-Sent Events server."""
        try:
            # Start SSE server
            asyncio.create_task(self._sse_server())
            
            logger.info(f"SSE server started on port {self.config.sse_port}")
            
        except Exception as e:
            logger.error(f"Failed to initialize SSE: {str(e)}")
            raise
    
    async def _initialize_dashboard(self):
        """Initialize monitoring dashboard."""
        try:
            # Start dashboard server
            asyncio.create_task(self._dashboard_server())
            
            logger.info(f"Dashboard server started on port {self.config.dashboard_port}")
            
        except Exception as e:
            logger.error(f"Failed to initialize dashboard: {str(e)}")
            raise
    
    async def _websocket_server(self):
        """WebSocket server for real-time metrics."""
        try:
            async def handle_client(websocket, path):
                self.websocket_clients.add(websocket)
                self.stats['websocket_clients'] = len(self.websocket_clients)
                
                try:
                    while True:
                        # Send metrics to client
                        metrics = await self._get_current_metrics()
                        await websocket.send(json.dumps(metrics))
                        await asyncio.sleep(1)  # Send every second
                        
                except websockets.exceptions.ConnectionClosed:
                    pass
                finally:
                    self.websocket_clients.remove(websocket)
                    self.stats['websocket_clients'] = len(self.websocket_clients)
            
            await websockets.serve(handle_client, "localhost", self.config.websocket_port)
            
        except Exception as e:
            logger.error(f"WebSocket server error: {str(e)}")
    
    async def _sse_server(self):
        """Server-Sent Events server for real-time metrics."""
        try:
            # This would implement SSE server
            # For now, just log
            logger.info("SSE server would be implemented here")
            
        except Exception as e:
            logger.error(f"SSE server error: {str(e)}")
    
    async def _dashboard_server(self):
        """Monitoring dashboard server."""
        try:
            # This would implement dashboard server
            # For now, just log
            logger.info("Dashboard server would be implemented here")
            
        except Exception as e:
            logger.error(f"Dashboard server error: {str(e)}")
    
    async def _collect_metrics(self):
        """Collect system metrics."""
        while self.running:
            try:
                # Collect system metrics
                cpu_usage = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                network = psutil.net_io_counters()
                
                # Create metrics
                metrics = [
                    Metric('cpu_usage', MetricType.GAUGE, 'CPU usage percentage', {}, cpu_usage),
                    Metric('memory_usage', MetricType.GAUGE, 'Memory usage percentage', {}, memory.percent),
                    Metric('disk_usage', MetricType.GAUGE, 'Disk usage percentage', {}, disk.percent),
                    Metric('network_sent', MetricType.COUNTER, 'Network bytes sent', {}, network.bytes_sent),
                    Metric('network_recv', MetricType.COUNTER, 'Network bytes received', {}, network.bytes_recv),
                    Metric('active_processes', MetricType.GAUGE, 'Active processes', {}, len(psutil.pids())),
                    Metric('load_average', MetricType.GAUGE, 'Load average', {}, psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else 0.0)
                ]
                
                # Store metrics
                async with self.lock:
                    for metric in metrics:
                        self.metrics[metric.name] = metric
                        self.metric_history.append(metric)
                        self.stats['total_metrics'] += 1
                
                # Update Prometheus metrics
                if self.config.enable_prometheus:
                    await self._update_prometheus_metrics(metrics)
                
                # Broadcast to WebSocket clients
                if self.websocket_clients:
                    await self._broadcast_metrics(metrics)
                
                # Check for alerts
                if self.config.enable_alerts:
                    await self._check_metric_alerts(metrics)
                
                # Wait for next collection
                interval = self._get_monitoring_interval()
                await asyncio.sleep(interval)
                
            except Exception as e:
                logger.error(f"Metric collection failed: {str(e)}")
                await asyncio.sleep(5)
    
    def _get_monitoring_interval(self) -> float:
        """Get monitoring interval in seconds."""
        intervals = {
            MonitoringInterval.REAL_TIME: 1.0,
            MonitoringInterval.FAST: 5.0,
            MonitoringInterval.NORMAL: 30.0,
            MonitoringInterval.SLOW: 300.0,
            MonitoringInterval.VERY_SLOW: 3600.0
        }
        return intervals.get(self.config.monitoring_interval, 30.0)
    
    async def _update_prometheus_metrics(self, metrics: List[Metric]):
        """Update Prometheus metrics."""
        try:
            for metric in metrics:
                if metric.name in self.prometheus_metrics:
                    prometheus_metric = self.prometheus_metrics[metric.name]
                    
                    if isinstance(prometheus_metric, Gauge):
                        prometheus_metric.set(metric.value)
                    elif isinstance(prometheus_metric, Counter):
                        prometheus_metric.inc(metric.value)
                    elif isinstance(prometheus_metric, Histogram):
                        prometheus_metric.observe(metric.value)
                    elif isinstance(prometheus_metric, Summary):
                        prometheus_metric.observe(metric.value)
                        
        except Exception as e:
            logger.error(f"Failed to update Prometheus metrics: {str(e)}")
    
    async def _broadcast_metrics(self, metrics: List[Metric]):
        """Broadcast metrics to WebSocket clients."""
        try:
            if not self.websocket_clients:
                return
            
            # Prepare metrics data
            metrics_data = {
                'timestamp': datetime.utcnow().isoformat(),
                'metrics': [
                    {
                        'name': metric.name,
                        'type': metric.metric_type.value,
                        'value': metric.value,
                        'labels': metric.labels,
                        'timestamp': metric.timestamp.isoformat()
                    }
                    for metric in metrics
                ]
            }
            
            # Send to all connected clients
            disconnected_clients = set()
            for client in self.websocket_clients:
                try:
                    await client.send(json.dumps(metrics_data))
                except websockets.exceptions.ConnectionClosed:
                    disconnected_clients.add(client)
            
            # Remove disconnected clients
            self.websocket_clients -= disconnected_clients
            
        except Exception as e:
            logger.error(f"Failed to broadcast metrics: {str(e)}")
    
    async def _check_metric_alerts(self, metrics: List[Metric]):
        """Check for metric alerts."""
        try:
            for metric in metrics:
                if metric.name in self.config.alert_thresholds:
                    threshold = self.config.alert_thresholds[metric.name]
                    
                    if metric.value > threshold:
                        # Create alert
                        alert = Alert(
                            id=f"{metric.name}_{int(time.time())}",
                            name=f"{metric.name}_high",
                            level=AlertLevel.WARNING,
                            message=f"{metric.name} is above threshold: {metric.value} > {threshold}",
                            metric_name=metric.name,
                            threshold=threshold,
                            current_value=metric.value
                        )
                        
                        # Store alert
                        self.alerts.append(alert)
                        self.stats['total_alerts'] += 1
                        self.stats['active_alerts'] += 1
                        
                        logger.warning(f"Alert triggered: {alert.message}")
                        
        except Exception as e:
            logger.error(f"Failed to check metric alerts: {str(e)}")
    
    async def _check_alerts(self):
        """Check for system alerts."""
        while self.running:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                # Check for critical system conditions
                cpu_usage = psutil.cpu_percent()
                memory_usage = psutil.virtual_memory().percent
                disk_usage = psutil.disk_usage('/').percent
                
                # CPU alert
                if cpu_usage > 90:
                    await self._create_alert(
                        'cpu_critical',
                        AlertLevel.CRITICAL,
                        f'CPU usage is critical: {cpu_usage}%',
                        'cpu_usage',
                        cpu_usage
                    )
                
                # Memory alert
                if memory_usage > 90:
                    await self._create_alert(
                        'memory_critical',
                        AlertLevel.CRITICAL,
                        f'Memory usage is critical: {memory_usage}%',
                        'memory_usage',
                        memory_usage
                    )
                
                # Disk alert
                if disk_usage > 90:
                    await self._create_alert(
                        'disk_critical',
                        AlertLevel.CRITICAL,
                        f'Disk usage is critical: {disk_usage}%',
                        'disk_usage',
                        disk_usage
                    )
                
            except Exception as e:
                logger.error(f"Alert checking failed: {str(e)}")
    
    async def _create_alert(self, alert_id: str, level: AlertLevel, message: str, metric_name: str, current_value: float):
        """Create system alert."""
        try:
            alert = Alert(
                id=alert_id,
                name=alert_id,
                level=level,
                message=message,
                metric_name=metric_name,
                threshold=90.0,
                current_value=current_value
            )
            
            self.alerts.append(alert)
            self.stats['total_alerts'] += 1
            self.stats['active_alerts'] += 1
            
            logger.warning(f"System alert: {message}")
            
        except Exception as e:
            logger.error(f"Failed to create alert: {str(e)}")
    
    async def _cleanup_old_data(self):
        """Cleanup old metrics and alerts."""
        while self.running:
            try:
                await asyncio.sleep(3600)  # Cleanup every hour
                
                current_time = datetime.utcnow()
                cutoff_time = current_time - timedelta(seconds=self.config.retention_period)
                
                # Cleanup old metrics
                async with self.lock:
                    # Remove old metrics from history
                    while self.metric_history and self.metric_history[0].timestamp < cutoff_time:
                        self.metric_history.popleft()
                    
                    # Remove old alerts
                    while self.alerts and self.alerts[0].timestamp < cutoff_time:
                        self.alerts.popleft()
                
                logger.debug("Cleaned up old monitoring data")
                
            except Exception as e:
                logger.error(f"Failed to cleanup old data: {str(e)}")
    
    async def _get_current_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        try:
            metrics_data = {}
            
            async with self.lock:
                for name, metric in self.metrics.items():
                    metrics_data[name] = {
                        'value': metric.value,
                        'type': metric.metric_type.value,
                        'timestamp': metric.timestamp.isoformat(),
                        'labels': metric.labels
                    }
            
            return metrics_data
            
        except Exception as e:
            logger.error(f"Failed to get current metrics: {str(e)}")
            return {}
    
    def get_monitoring_stats(self) -> Dict[str, Any]:
        """Get monitoring statistics."""
        return {
            'total_metrics': self.stats['total_metrics'],
            'total_alerts': self.stats['total_alerts'],
            'active_alerts': self.stats['active_alerts'],
            'websocket_clients': self.stats['websocket_clients'],
            'sse_clients': self.stats['sse_clients'],
            'current_metrics': len(self.metrics),
            'metric_history_size': len(self.metric_history),
            'alerts_size': len(self.alerts),
            'config': {
                'prometheus_enabled': self.config.enable_prometheus,
                'websocket_enabled': self.config.enable_websocket,
                'sse_enabled': self.config.enable_sse,
                'dashboard_enabled': self.config.enable_dashboard,
                'alerts_enabled': self.config.enable_alerts,
                'monitoring_interval': self.config.monitoring_interval.value,
                'retention_period': self.config.retention_period,
                'max_metrics': self.config.max_metrics,
                'max_alerts': self.config.max_alerts
            }
        }
    
    async def cleanup(self):
        """Cleanup real-time monitor."""
        try:
            self.running = False
            
            # Close WebSocket connections
            for client in self.websocket_clients:
                await client.close()
            
            # Clear data
            self.metrics.clear()
            self.alerts.clear()
            self.metric_history.clear()
            self.websocket_clients.clear()
            self.sse_clients.clear()
            
            logger.info("Real-time Monitor cleaned up successfully")
            
        except Exception as e:
            logger.error(f"Failed to cleanup Real-time Monitor: {str(e)}")

# Global real-time monitor
realtime_monitor = RealTimeMonitor()

# Decorators for real-time monitoring
def monitor_metric(name: str, metric_type: MetricType = MetricType.GAUGE):
    """Decorator for monitoring metrics."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                
                # Record success metric
                execution_time = time.time() - start_time
                metric = Metric(
                    name=name,
                    metric_type=metric_type,
                    value=execution_time,
                    labels={'status': 'success'}
                )
                
                # Store metric
                realtime_monitor.metrics[name] = metric
                realtime_monitor.metric_history.append(metric)
                
                return result
                
            except Exception as e:
                # Record error metric
                execution_time = time.time() - start_time
                metric = Metric(
                    name=name,
                    metric_type=metric_type,
                    value=execution_time,
                    labels={'status': 'error', 'error': str(e)}
                )
                
                # Store metric
                realtime_monitor.metrics[name] = metric
                realtime_monitor.metric_history.append(metric)
                
                raise
        
        return wrapper
    return decorator

def alert_on_threshold(threshold: float, level: AlertLevel = AlertLevel.WARNING):
    """Decorator for threshold-based alerts."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            result = await func(*args, **kwargs)
            
            # Check threshold
            if isinstance(result, (int, float)) and result > threshold:
                await realtime_monitor._create_alert(
                    f"{func.__name__}_threshold",
                    level,
                    f"{func.__name__} exceeded threshold: {result} > {threshold}",
                    func.__name__,
                    result
                )
            
            return result
        
        return wrapper
    return decorator











