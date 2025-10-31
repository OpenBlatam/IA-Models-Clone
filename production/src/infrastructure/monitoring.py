from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

# Constants
BUFFER_SIZE = 1024

import asyncio
import logging
import time
import psutil
import gc
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
import structlog
from prometheus_client import (
from prometheus_client.openmetrics.exposition import generate_latest as generate_latest_openmetrics
from src.core.config import MonitoringSettings
from src.core.exceptions import BusinessException
from typing import Any, List, Dict, Optional
"""
📊 Ultra-Optimized Monitoring Service
=====================================

Production-grade monitoring with:
- Prometheus metrics
- Custom business metrics
- Performance tracking
- Alert management
- System health monitoring
"""


    Counter, Histogram, Gauge, Summary, Info,
    generate_latest, CONTENT_TYPE_LATEST
)



class MonitoringService:
    """
    Ultra-optimized monitoring service with comprehensive
    metrics collection and performance tracking.
    """
    
    def __init__(self, config: MonitoringSettings):
        
    """__init__ function."""
self.config = config
        self.logger = structlog.get_logger(__name__)
        
        # Performance tracking
        self.start_time = time.time()
        self.request_count = 0
        self.error_count = 0
        self.total_response_time = 0.0
        
        # System metrics
        self.system_metrics = {}
        self.last_system_check = 0
        
        # Business metrics
        self.business_metrics = {
            "content_generated": 0,
            "users_active": 0,
            "templates_created": 0,
            "ai_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0
        }
        
        # Alert management
        self.alerts = []
        self.alert_rules = self._initialize_alert_rules()
        
        # Prometheus metrics
        self._initialize_prometheus_metrics()
        
        # Background tasks
        self.metrics_collector_task = None
        self.alert_checker_task = None
        self.system_monitor_task = None
        
        # Health status
        self.is_healthy = False
        self.last_health_check = None
        
        self.logger.info("Monitoring Service initialized")
    
    def _initialize_prometheus_metrics(self) -> Any:
        """Initialize Prometheus metrics"""
        
        # HTTP metrics
        self.http_requests_total = Counter(
            'http_requests_total',
            'Total HTTP requests',
            ['method', 'endpoint', 'status']
        )
        
        self.http_request_duration = Histogram(
            'http_request_duration_seconds',
            'HTTP request duration',
            ['method', 'endpoint']
        )
        
        self.http_request_size = Histogram(
            'http_request_size_bytes',
            'HTTP request size',
            ['method', 'endpoint']
        )
        
        self.http_response_size = Histogram(
            'http_response_size_bytes',
            'HTTP response size',
            ['method', 'endpoint']
        )
        
        # Business metrics
        self.content_generated_total = Counter(
            'content_generated_total',
            'Total content generated',
            ['content_type', 'language']
        )
        
        self.ai_requests_total = Counter(
            'ai_requests_total',
            'Total AI requests',
            ['model', 'operation']
        )
        
        self.ai_request_duration = Histogram(
            'ai_request_duration_seconds',
            'AI request duration',
            ['model', 'operation']
        )
        
        self.cache_operations_total = Counter(
            'cache_operations_total',
            'Total cache operations',
            ['operation', 'status']
        )
        
        self.cache_hit_ratio = Gauge(
            'cache_hit_ratio',
            'Cache hit ratio'
        )
        
        # Database metrics
        self.db_connections = Gauge(
            'db_connections',
            'Database connections',
            ['status']
        )
        
        self.db_query_duration = Histogram(
            'db_query_duration_seconds',
            'Database query duration',
            ['operation']
        )
        
        self.db_queries_total = Counter(
            'db_queries_total',
            'Total database queries',
            ['operation', 'status']
        )
        
        # System metrics
        self.cpu_usage = Gauge(
            'cpu_usage_percent',
            'CPU usage percentage'
        )
        
        self.memory_usage = Gauge(
            'memory_usage_bytes',
            'Memory usage in bytes'
        )
        
        self.memory_usage_percent = Gauge(
            'memory_usage_percent',
            'Memory usage percentage'
        )
        
        self.disk_usage = Gauge(
            'disk_usage_bytes',
            'Disk usage in bytes',
            ['mount_point']
        )
        
        self.disk_usage_percent = Gauge(
            'disk_usage_percent',
            'Disk usage percentage',
            ['mount_point']
        )
        
        self.network_io = Counter(
            'network_io_bytes',
            'Network I/O in bytes',
            ['direction']
        )
        
        # Application metrics
        self.active_connections = Gauge(
            'active_connections',
            'Active connections'
        )
        
        self.queue_size = Gauge(
            'queue_size',
            'Queue size',
            ['queue_name']
        )
        
        self.worker_count = Gauge(
            'worker_count',
            'Number of workers',
            ['worker_type']
        )
        
        # Error metrics
        self.errors_total = Counter(
            'errors_total',
            'Total errors',
            ['type', 'severity']
        )
        
        self.error_rate = Gauge(
            'error_rate',
            'Error rate (errors per second)'
        )
        
        # Custom metrics
        self.custom_metrics = {}
        
        # Application info
        self.app_info = Info(
            'app',
            'Application information'
        )
        self.app_info.info({
            'name': 'Ultra-Optimized AI Copywriting System',
            'version': '2.0.0',
            'environment': self.config.ENVIRONMENT
        })
    
    def _initialize_alert_rules(self) -> Dict[str, Any]:
        """Initialize alert rules"""
        
        return {
            "high_error_rate": {
                "condition": lambda metrics: metrics.get("error_rate", 0) > 0.1,
                "severity": "critical",
                "message": "High error rate detected"
            },
            "high_cpu_usage": {
                "condition": lambda metrics: metrics.get("cpu_usage", 0) > 80,
                "severity": "warning",
                "message": "High CPU usage detected"
            },
            "high_memory_usage": {
                "condition": lambda metrics: metrics.get("memory_usage_percent", 0) > 85,
                "severity": "warning",
                "message": "High memory usage detected"
            },
            "low_cache_hit_ratio": {
                "condition": lambda metrics: metrics.get("cache_hit_ratio", 1) < 0.7,
                "severity": "warning",
                "message": "Low cache hit ratio detected"
            },
            "high_response_time": {
                "condition": lambda metrics: metrics.get("avg_response_time", 0) > 2.0,
                "severity": "warning",
                "message": "High response time detected"
            }
        }
    
    async def initialize(self) -> Any:
        """Initialize monitoring service"""
        
        self.logger.info("Initializing Monitoring Service...")
        
        try:
            # Start background tasks
            self.metrics_collector_task = asyncio.create_task(self._metrics_collector())
            self.alert_checker_task = asyncio.create_task(self._alert_checker())
            self.system_monitor_task = asyncio.create_task(self._system_monitor())
            
            # Set health status
            self.is_healthy = True
            self.last_health_check = time.time()
            
            self.logger.info("Monitoring Service initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Monitoring Service: {e}")
            raise BusinessException(f"Monitoring Service initialization failed: {e}")
    
    async def cleanup(self) -> Any:
        """Cleanup monitoring service"""
        
        self.logger.info("Cleaning up Monitoring Service...")
        
        # Stop background tasks
        tasks = [
            self.metrics_collector_task,
            self.alert_checker_task,
            self.system_monitor_task
        ]
        
        for task in tasks:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        self.logger.info("Monitoring Service cleanup completed")
    
    def record_request(self, method: str, endpoint: str, status: int, 
                      duration: float, request_size: int = 0, response_size: int = 0):
        """Record HTTP request metrics"""
        
        try:
            # Update counters
            self.http_requests_total.labels(
                method=method,
                endpoint=endpoint,
                status=status
            ).inc()
            
            # Update histograms
            self.http_request_duration.labels(
                method=method,
                endpoint=endpoint
            ).observe(duration)
            
            if request_size > 0:
                self.http_request_size.labels(
                    method=method,
                    endpoint=endpoint
                ).observe(request_size)
            
            if response_size > 0:
                self.http_response_size.labels(
                    method=method,
                    endpoint=endpoint
                ).observe(response_size)
            
            # Update internal metrics
            self.request_count += 1
            self.total_response_time += duration
            
            if status >= 400:
                self.error_count += 1
            
        except Exception as e:
            self.logger.error(f"Failed to record request metrics: {e}")
    
    def record_content_generation(self, content_type: str, language: str, duration: float):
        """Record content generation metrics"""
        
        try:
            # Update counters
            self.content_generated_total.labels(
                content_type=content_type,
                language=language
            ).inc()
            
            # Update business metrics
            self.business_metrics["content_generated"] += 1
            
        except Exception as e:
            self.logger.error(f"Failed to record content generation metrics: {e}")
    
    def record_ai_request(self, model: str, operation: str, duration: float, success: bool):
        """Record AI request metrics"""
        
        try:
            # Update counters
            self.ai_requests_total.labels(
                model=model,
                operation=operation
            ).inc()
            
            # Update histograms
            self.ai_request_duration.labels(
                model=model,
                operation=operation
            ).observe(duration)
            
            # Update business metrics
            self.business_metrics["ai_requests"] += 1
            
        except Exception as e:
            self.logger.error(f"Failed to record AI request metrics: {e}")
    
    def record_cache_operation(self, operation: str, hit: bool):
        """Record cache operation metrics"""
        
        try:
            # Update counters
            status = "hit" if hit else "miss"
            self.cache_operations_total.labels(
                operation=operation,
                status=status
            ).inc()
            
            # Update business metrics
            if hit:
                self.business_metrics["cache_hits"] += 1
            else:
                self.business_metrics["cache_misses"] += 1
            
            # Update hit ratio
            total_ops = self.business_metrics["cache_hits"] + self.business_metrics["cache_misses"]
            if total_ops > 0:
                hit_ratio = self.business_metrics["cache_hits"] / total_ops
                self.cache_hit_ratio.set(hit_ratio)
            
        except Exception as e:
            self.logger.error(f"Failed to record cache operation metrics: {e}")
    
    def record_database_operation(self, operation: str, duration: float, success: bool):
        """Record database operation metrics"""
        
        try:
            # Update counters
            status = "success" if success else "error"
            self.db_queries_total.labels(
                operation=operation,
                status=status
            ).inc()
            
            # Update histograms
            self.db_query_duration.labels(
                operation=operation
            ).observe(duration)
            
        except Exception as e:
            self.logger.error(f"Failed to record database operation metrics: {e}")
    
    def record_error(self, error_type: str, severity: str = "error"):
        """Record error metrics"""
        
        try:
            # Update counters
            self.errors_total.labels(
                type=error_type,
                severity=severity
            ).inc()
            
            # Update internal metrics
            self.error_count += 1
            
        except Exception as e:
            self.logger.error(f"Failed to record error metrics: {e}")
    
    def set_custom_metric(self, name: str, value: float, labels: Dict[str, str] = None):
        """Set custom metric"""
        
        try:
            if name not in self.custom_metrics:
                self.custom_metrics[name] = Gauge(
                    f'custom_{name}',
                    f'Custom metric: {name}',
                    list(labels.keys()) if labels else []
                )
            
            metric = self.custom_metrics[name]
            if labels:
                metric.labels(**labels).set(value)
            else:
                metric.set(value)
            
        except Exception as e:
            self.logger.error(f"Failed to set custom metric: {e}")
    
    async def _metrics_collector(self) -> Any:
        """Background task for collecting metrics"""
        
        while True:
            try:
                # Collect system metrics
                await self._collect_system_metrics()
                
                # Update error rate
                self._update_error_rate()
                
                # Wait before next collection
                await asyncio.sleep(30)  # 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Metrics collector error: {e}")
                await asyncio.sleep(60)  # Wait 1 minute on error
    
    async def _collect_system_metrics(self) -> Any:
        """Collect system metrics"""
        
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.cpu_usage.set(cpu_percent)
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.memory_usage.set(memory.used)
            self.memory_usage_percent.set(memory.percent)
            
            # Disk usage
            for partition in psutil.disk_partitions():
                try:
                    usage = psutil.disk_usage(partition.mountpoint)
                    mount_point = partition.mountpoint.replace('/', '_')
                    
                    self.disk_usage.labels(mount_point=mount_point).set(usage.used)
                    self.disk_usage_percent.labels(mount_point=mount_point).set(
                        (usage.used / usage.total) * 100
                    )
                except PermissionError:
                    continue
            
            # Network I/O
            net_io = psutil.net_io_counters()
            self.network_io.labels(direction="bytes_sent").inc(net_io.bytes_sent)
            self.network_io.labels(direction="bytes_recv").inc(net_io.bytes_recv)
            
            # Update system metrics cache
            self.system_metrics = {
                "cpu_usage": cpu_percent,
                "memory_usage": memory.used,
                "memory_usage_percent": memory.percent,
                "disk_usage": usage.used if 'usage' in locals() else 0,
                "network_sent": net_io.bytes_sent,
                "network_recv": net_io.bytes_recv
            }
            
            self.last_system_check = time.time()
            
        except Exception as e:
            self.logger.error(f"Failed to collect system metrics: {e}")
    
    def _update_error_rate(self) -> Any:
        """Update error rate metric"""
        
        try:
            if self.request_count > 0:
                error_rate = self.error_count / self.request_count
                self.error_rate.set(error_rate)
            
        except Exception as e:
            self.logger.error(f"Failed to update error rate: {e}")
    
    async def _alert_checker(self) -> Any:
        """Background task for checking alerts"""
        
        while True:
            try:
                # Get current metrics
                current_metrics = self.get_current_metrics()
                
                # Check alert rules
                await self._check_alert_rules(current_metrics)
                
                # Clean old alerts
                self._cleanup_old_alerts()
                
                # Wait before next check
                await asyncio.sleep(60)  # 1 minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Alert checker error: {e}")
                await asyncio.sleep(120)  # Wait 2 minutes on error
    
    async def _check_alert_rules(self, metrics: Dict[str, Any]):
        """Check alert rules against current metrics"""
        
        try:
            for rule_name, rule in self.alert_rules.items():
                if rule["condition"](metrics):
                    # Check if alert already exists
                    existing_alert = next(
                        (alert for alert in self.alerts 
                         if alert["rule"] == rule_name and not alert["acknowledged"]),
                        None
                    )
                    
                    if not existing_alert:
                        # Create new alert
                        alert = {
                            "id": f"{rule_name}_{int(time.time())}",
                            "rule": rule_name,
                            "severity": rule["severity"],
                            "message": rule["message"],
                            "timestamp": datetime.utcnow().isoformat(),
                            "acknowledged": False,
                            "metrics": metrics
                        }
                        
                        self.alerts.append(alert)
                        
                        self.logger.warning(
                            f"Alert triggered: {rule_name}",
                            severity=rule["severity"],
                            message=rule["message"]
                        )
                        
        except Exception as e:
            self.logger.error(f"Failed to check alert rules: {e}")
    
    def _cleanup_old_alerts(self) -> Any:
        """Clean up old alerts"""
        
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=24)
            
            self.alerts = [
                alert for alert in self.alerts
                if datetime.fromisoformat(alert["timestamp"]) > cutoff_time
            ]
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup old alerts: {e}")
    
    async def _system_monitor(self) -> Any:
        """Background task for system monitoring"""
        
        while True:
            try:
                # Monitor application health
                await self._monitor_application_health()
                
                # Monitor resource usage
                await self._monitor_resource_usage()
                
                # Wait before next check
                await asyncio.sleep(300)  # 5 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"System monitor error: {e}")
                await asyncio.sleep(600)  # Wait 10 minutes on error
    
    async def _monitor_application_health(self) -> Any:
        """Monitor application health"""
        
        try:
            # Check memory usage
            memory = psutil.virtual_memory()
            if memory.percent > 90:
                self.logger.warning("High memory usage detected")
            
            # Check CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            if cpu_percent > 90:
                self.logger.warning("High CPU usage detected")
            
            # Check disk usage
            for partition in psutil.disk_partitions():
                try:
                    usage = psutil.disk_usage(partition.mountpoint)
                    if (usage.used / usage.total) * 100 > 90:
                        self.logger.warning(f"High disk usage on {partition.mountpoint}")
                except PermissionError:
                    continue
            
        except Exception as e:
            self.logger.error(f"Failed to monitor application health: {e}")
    
    async def _monitor_resource_usage(self) -> Any:
        """Monitor resource usage"""
        
        try:
            # Monitor garbage collection
            gc.collect()
            
            # Log resource usage
            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=1)
            
            self.logger.info(
                "Resource usage",
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_used_gb=memory.used / (1024**3)
            )
            
        except Exception as e:
            self.logger.error(f"Failed to monitor resource usage: {e}")
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current metrics"""
        
        try:
            return {
                "request_count": self.request_count,
                "error_count": self.error_count,
                "error_rate": self.error_count / self.request_count if self.request_count > 0 else 0,
                "avg_response_time": self.total_response_time / self.request_count if self.request_count > 0 else 0,
                "uptime": time.time() - self.start_time,
                "business_metrics": self.business_metrics.copy(),
                "system_metrics": self.system_metrics.copy()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get current metrics: {e}")
            return {}
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        
        try:
            return {
                "uptime_seconds": time.time() - self.start_time,
                "requests_per_second": self.request_count / (time.time() - self.start_time),
                "error_rate": self.error_count / self.request_count if self.request_count > 0 else 0,
                "average_response_time": self.total_response_time / self.request_count if self.request_count > 0 else 0,
                "business_metrics": self.business_metrics,
                "system_metrics": self.system_metrics
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get performance summary: {e}")
            return {"error": str(e)}
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get system metrics"""
        
        try:
            return {
                "cpu_usage": psutil.cpu_percent(interval=1),
                "memory_usage": psutil.virtual_memory().percent,
                "disk_usage": {},
                "network_io": {},
                "process_info": {
                    "pid": psutil.Process().pid,
                    "memory_info": psutil.Process().memory_info()._asdict(),
                    "cpu_percent": psutil.Process().cpu_percent()
                }
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get system metrics: {e}")
            return {"error": str(e)}
    
    def get_alerts(self) -> List[Dict[str, Any]]:
        """Get current alerts"""
        
        try:
            return self.alerts.copy()
            
        except Exception as e:
            self.logger.error(f"Failed to get alerts: {e}")
            return []
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert"""
        
        try:
            for alert in self.alerts:
                if alert["id"] == alert_id:
                    alert["acknowledged"] = True
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to acknowledge alert: {e}")
            return False
    
    def get_metrics_prometheus(self) -> str:
        """Get Prometheus metrics"""
        
        try:
            return generate_latest()
            
        except Exception as e:
            self.logger.error(f"Failed to generate Prometheus metrics: {e}")
            return ""
    
    def get_metrics_openmetrics(self) -> str:
        """Get OpenMetrics format metrics"""
        
        try:
            return generate_latest_openmetrics()
            
        except Exception as e:
            self.logger.error(f"Failed to generate OpenMetrics: {e}")
            return ""
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check"""
        
        try:
            start_time = time.time()
            
            # Basic health checks
            checks = {
                "metrics_collection": self.metrics_collector_task and not self.metrics_collector_task.done(),
                "alert_checking": self.alert_checker_task and not self.alert_checker_task.done(),
                "system_monitoring": self.system_monitor_task and not self.system_monitor_task.done()
            }
            
            all_healthy = all(checks.values())
            
            response_time = time.time() - start_time
            
            # Update health status
            self.is_healthy = all_healthy
            self.last_health_check = time.time()
            
            return {
                "status": "healthy" if all_healthy else "unhealthy",
                "response_time": response_time,
                "checks": checks,
                "uptime": time.time() - self.start_time,
                "last_health_check": self.last_health_check
            }
            
        except Exception as e:
            self.is_healthy = False
            self.logger.error(f"Health check failed: {e}")
            
            return {
                "status": "unhealthy",
                "error": str(e),
                "last_health_check": self.last_health_check
            } 