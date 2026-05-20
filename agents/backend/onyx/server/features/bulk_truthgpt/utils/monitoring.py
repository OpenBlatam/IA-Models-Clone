"""
Advanced Monitoring System
==========================

Ultra-advanced monitoring system with Prometheus, Grafana, and custom metrics.
"""

import time
import logging
import threading
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import psutil
import redis
from flask import current_app, g, request
from prometheus_client import Counter, Histogram, Gauge, Summary, CollectorRegistry, generate_latest
import json

logger = logging.getLogger(__name__)

class MetricType(str, Enum):
    """Metric types."""
    COUNTER = "counter"
    HISTOGRAM = "histogram"
    GAUGE = "gauge"
    SUMMARY = "summary"

@dataclass
class MetricConfig:
    """Metric configuration."""
    name: str
    description: str
    metric_type: MetricType
    labels: List[str] = field(default_factory=list)
    buckets: List[float] = field(default_factory=lambda: [0.1, 0.5, 1.0, 2.5, 5.0, 10.0])
    quantiles: List[float] = field(default_factory=lambda: [0.5, 0.9, 0.95, 0.99])

@dataclass
class AlertConfig:
    """Alert configuration."""
    name: str
    condition: str
    threshold: float
    severity: str
    message: str
    enabled: bool = True

class AdvancedMonitoring:
    """
    Ultra-advanced monitoring system.
    
    Features:
    - Prometheus metrics
    - Custom metrics
    - System monitoring
    - Performance tracking
    - Alerting
    - Dashboard integration
    """
    
    def __init__(self):
        self.registry = CollectorRegistry()
        self.metrics = {}
        self.alerts = {}
        self.running = False
        self.monitor_thread = None
        self.redis_client = None
        self.alert_callbacks = []
        
    def initialize(self):
        """Initialize monitoring system."""
        try:
            # Initialize Redis client
            if current_app.config.get('REDIS_URL'):
                self.redis_client = redis.from_url(current_app.config['REDIS_URL'])
            else:
                self.redis_client = redis.Redis(
                    host=current_app.config.get('REDIS_HOST', 'localhost'),
                    port=current_app.config.get('REDIS_PORT', 6379),
                    db=current_app.config.get('REDIS_DB', 0)
                )
            
            # Initialize default metrics
            self._initialize_default_metrics()
            
            # Initialize alerts
            self._initialize_default_alerts()
            
            # Start monitoring
            self.running = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop)
            self.monitor_thread.daemon = True
            self.monitor_thread.start()
            
            logger.info("Advanced monitoring system initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize monitoring system: {str(e)}")
            raise
    
    def _initialize_default_metrics(self):
        """Initialize default metrics."""
        try:
            # Request metrics
            self._create_metric('http_requests_total', 'Total HTTP requests', MetricType.COUNTER, ['method', 'endpoint', 'status'])
            self._create_metric('http_request_duration_seconds', 'HTTP request duration', MetricType.HISTOGRAM, ['method', 'endpoint'])
            self._create_metric('http_request_size_bytes', 'HTTP request size', MetricType.HISTOGRAM, ['method', 'endpoint'])
            self._create_metric('http_response_size_bytes', 'HTTP response size', MetricType.HISTOGRAM, ['method', 'endpoint'])
            
            # System metrics
            self._create_metric('system_cpu_usage_percent', 'System CPU usage', MetricType.GAUGE)
            self._create_metric('system_memory_usage_percent', 'System memory usage', MetricType.GAUGE)
            self._create_metric('system_disk_usage_percent', 'System disk usage', MetricType.GAUGE)
            self._create_metric('system_network_bytes_total', 'System network bytes', MetricType.COUNTER, ['direction'])
            
            # Application metrics
            self._create_metric('app_active_users', 'Active users', MetricType.GAUGE)
            self._create_metric('app_optimization_sessions_total', 'Total optimization sessions', MetricType.COUNTER, ['type', 'status'])
            self._create_metric('app_performance_metrics_total', 'Total performance metrics', MetricType.COUNTER, ['metric_name'])
            
            # Database metrics
            self._create_metric('db_connections_active', 'Active database connections', MetricType.GAUGE)
            self._create_metric('db_queries_total', 'Total database queries', MetricType.COUNTER, ['operation', 'table'])
            self._create_metric('db_query_duration_seconds', 'Database query duration', MetricType.HISTOGRAM, ['operation', 'table'])
            
            # Cache metrics
            self._create_metric('cache_hits_total', 'Cache hits', MetricType.COUNTER, ['cache_name'])
            self._create_metric('cache_misses_total', 'Cache misses', MetricType.COUNTER, ['cache_name'])
            self._create_metric('cache_size_bytes', 'Cache size', MetricType.GAUGE, ['cache_name'])
            
            logger.info("Default metrics initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize default metrics: {str(e)}")
            raise
    
    def _initialize_default_alerts(self):
        """Initialize default alerts."""
        try:
            # System alerts
            self._create_alert('high_cpu_usage', 'system_cpu_usage_percent > 80', 80, 'warning', 'High CPU usage detected')
            self._create_alert('high_memory_usage', 'system_memory_usage_percent > 85', 85, 'warning', 'High memory usage detected')
            self._create_alert('high_disk_usage', 'system_disk_usage_percent > 90', 90, 'critical', 'High disk usage detected')
            
            # Application alerts
            self._create_alert('high_error_rate', 'http_requests_total{status=5xx} / http_requests_total > 0.05', 0.05, 'critical', 'High error rate detected')
            self._create_alert('slow_response_time', 'http_request_duration_seconds{quantile=0.95} > 5', 5, 'warning', 'Slow response time detected')
            
            # Database alerts
            self._create_alert('high_db_connections', 'db_connections_active > 80', 80, 'warning', 'High database connections detected')
            self._create_alert('slow_db_queries', 'db_query_duration_seconds{quantile=0.95} > 2', 2, 'warning', 'Slow database queries detected')
            
            logger.info("Default alerts initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize default alerts: {str(e)}")
            raise
    
    def _create_metric(self, name: str, description: str, metric_type: MetricType, labels: List[str] = None):
        """Create a metric."""
        try:
            config = MetricConfig(
                name=name,
                description=description,
                metric_type=metric_type,
                labels=labels or []
            )
            
            if metric_type == MetricType.COUNTER:
                metric = Counter(name, description, labels, registry=self.registry)
            elif metric_type == MetricType.HISTOGRAM:
                metric = Histogram(name, description, labels, buckets=config.buckets, registry=self.registry)
            elif metric_type == MetricType.GAUGE:
                metric = Gauge(name, description, labels, registry=self.registry)
            elif metric_type == MetricType.SUMMARY:
                metric = Summary(name, description, labels, registry=self.registry)
            else:
                raise ValueError(f"Unknown metric type: {metric_type}")
            
            self.metrics[name] = metric
            logger.debug(f"Created metric: {name}")
            
        except Exception as e:
            logger.error(f"Failed to create metric {name}: {str(e)}")
            raise
    
    def _create_alert(self, name: str, condition: str, threshold: float, severity: str, message: str):
        """Create an alert."""
        try:
            alert = AlertConfig(
                name=name,
                condition=condition,
                threshold=threshold,
                severity=severity,
                message=message
            )
            
            self.alerts[name] = alert
            logger.debug(f"Created alert: {name}")
            
        except Exception as e:
            logger.error(f"Failed to create alert {name}: {str(e)}")
            raise
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.running:
            try:
                # Update system metrics
                self._update_system_metrics()
                
                # Check alerts
                self._check_alerts()
                
                # Store metrics in Redis
                self._store_metrics()
                
                # Sleep for monitoring interval
                time.sleep(30)  # Monitor every 30 seconds
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {str(e)}")
                time.sleep(60)  # Wait longer on error
    
    def _update_system_metrics(self):
        """Update system metrics."""
        try:
            # CPU usage
            cpu_usage = psutil.cpu_percent(interval=1)
            self.metrics['system_cpu_usage_percent'].set(cpu_usage)
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.metrics['system_memory_usage_percent'].set(memory.percent)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_usage = (disk.used / disk.total) * 100
            self.metrics['system_disk_usage_percent'].set(disk_usage)
            
            # Network usage
            network = psutil.net_io_counters()
            self.metrics['system_network_bytes_total'].labels(direction='sent').inc(network.bytes_sent)
            self.metrics['system_network_bytes_total'].labels(direction='received').inc(network.bytes_recv)
            
            logger.debug("System metrics updated")
            
        except Exception as e:
            logger.error(f"Failed to update system metrics: {str(e)}")
    
    def _check_alerts(self):
        """Check alerts."""
        try:
            for alert_name, alert in self.alerts.items():
                if not alert.enabled:
                    continue
                
                # This would implement actual alert checking
                # For now, just log
                logger.debug(f"Checking alert: {alert_name}")
                
        except Exception as e:
            logger.error(f"Failed to check alerts: {str(e)}")
    
    def _store_metrics(self):
        """Store metrics in Redis."""
        try:
            if not self.redis_client:
                return
            
            # Store metrics data
            metrics_data = {
                'timestamp': datetime.utcnow().isoformat(),
                'metrics': {}
            }
            
            for name, metric in self.metrics.items():
                # This would implement actual metric storage
                metrics_data['metrics'][name] = str(metric)
            
            # Store in Redis
            self.redis_client.setex('monitoring:metrics', 300, json.dumps(metrics_data))
            
        except Exception as e:
            logger.error(f"Failed to store metrics: {str(e)}")
    
    def record_request(self, method: str, endpoint: str, status_code: int, duration: float, request_size: int = 0, response_size: int = 0):
        """Record HTTP request metrics."""
        try:
            # Record request count
            self.metrics['http_requests_total'].labels(
                method=method,
                endpoint=endpoint,
                status=str(status_code)
            ).inc()
            
            # Record request duration
            self.metrics['http_request_duration_seconds'].labels(
                method=method,
                endpoint=endpoint
            ).observe(duration)
            
            # Record request size
            if request_size > 0:
                self.metrics['http_request_size_bytes'].labels(
                    method=method,
                    endpoint=endpoint
                ).observe(request_size)
            
            # Record response size
            if response_size > 0:
                self.metrics['http_response_size_bytes'].labels(
                    method=method,
                    endpoint=endpoint
                ).observe(response_size)
            
        except Exception as e:
            logger.error(f"Failed to record request metrics: {str(e)}")
    
    def record_optimization_session(self, session_type: str, status: str):
        """Record optimization session metrics."""
        try:
            self.metrics['app_optimization_sessions_total'].labels(
                type=session_type,
                status=status
            ).inc()
            
        except Exception as e:
            logger.error(f"Failed to record optimization session metrics: {str(e)}")
    
    def record_performance_metric(self, metric_name: str, value: float):
        """Record performance metric."""
        try:
            self.metrics['app_performance_metrics_total'].labels(
                metric_name=metric_name
            ).inc()
            
        except Exception as e:
            logger.error(f"Failed to record performance metric: {str(e)}")
    
    def record_database_query(self, operation: str, table: str, duration: float):
        """Record database query metrics."""
        try:
            self.metrics['db_queries_total'].labels(
                operation=operation,
                table=table
            ).inc()
            
            self.metrics['db_query_duration_seconds'].labels(
                operation=operation,
                table=table
            ).observe(duration)
            
        except Exception as e:
            logger.error(f"Failed to record database query metrics: {str(e)}")
    
    def record_cache_operation(self, cache_name: str, hit: bool):
        """Record cache operation metrics."""
        try:
            if hit:
                self.metrics['cache_hits_total'].labels(cache_name=cache_name).inc()
            else:
                self.metrics['cache_misses_total'].labels(cache_name=cache_name).inc()
            
        except Exception as e:
            logger.error(f"Failed to record cache operation metrics: {str(e)}")
    
    def get_metrics(self) -> str:
        """Get metrics in Prometheus format."""
        try:
            return generate_latest(self.registry)
        except Exception as e:
            logger.error(f"Failed to get metrics: {str(e)}")
            return ""
    
    def get_metrics_json(self) -> Dict[str, Any]:
        """Get metrics in JSON format."""
        try:
            metrics_data = {
                'timestamp': datetime.utcnow().isoformat(),
                'metrics': {}
            }
            
            for name, metric in self.metrics.items():
                # This would implement actual metric serialization
                metrics_data['metrics'][name] = str(metric)
            
            return metrics_data
            
        except Exception as e:
            logger.error(f"Failed to get metrics JSON: {str(e)}")
            return {}
    
    def add_alert_callback(self, callback: Callable):
        """Add alert callback."""
        try:
            self.alert_callbacks.append(callback)
            logger.debug("Alert callback added")
            
        except Exception as e:
            logger.error(f"Failed to add alert callback: {str(e)}")
    
    def cleanup(self):
        """Cleanup monitoring system."""
        try:
            self.running = False
            
            if self.monitor_thread:
                self.monitor_thread.join(timeout=5)
            
            if self.redis_client:
                self.redis_client.close()
            
            logger.info("Monitoring system cleaned up")
            
        except Exception as e:
            logger.error(f"Failed to cleanup monitoring system: {str(e)}")

# Global monitoring instance
monitoring = AdvancedMonitoring()

# Decorator for monitoring
def monitor_performance(metric_name: str):
    """Decorator for performance monitoring."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                # Record performance metric
                monitoring.record_performance_metric(metric_name, duration)
                
                return result
            except Exception as e:
                duration = time.time() - start_time
                monitoring.record_performance_metric(f"{metric_name}_error", duration)
                raise e
        
        return wrapper
    return decorator









