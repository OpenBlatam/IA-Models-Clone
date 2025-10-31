from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import asyncio
import time
import json
import statistics
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict, deque
import weakref
from fastapi import FastAPI, Request, Response
from pydantic import BaseModel, Field
import structlog
from prometheus_client import Counter, Histogram, Gauge, Summary, generate_latest, REGISTRY
import redis.asyncio as redis
            import psutil
    from fastapi import FastAPI
from typing import Any, List, Dict, Optional
import logging
"""
📊 Middleware Monitoring System
==============================

Comprehensive monitoring system for middleware with real-time metrics,
alerting, analytics, and performance tracking.
"""



logger = structlog.get_logger(__name__)

# =============================================================================
# MONITORING CONFIGURATION
# =============================================================================

class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class AlertType(Enum):
    """Alert types."""
    ERROR_RATE = "error_rate"
    RESPONSE_TIME = "response_time"
    MEMORY_USAGE = "memory_usage"
    CPU_USAGE = "cpu_usage"
    RATE_LIMIT = "rate_limit"
    SLOW_REQUEST = "slow_request"
    CONNECTION_ERROR = "connection_error"
    CUSTOM = "custom"

class MonitoringConfig(BaseModel):
    """Configuration for monitoring system."""
    
    # General settings
    enabled: bool = Field(default=True, description="Enable monitoring")
    environment: str = Field(default="production", description="Environment")
    service_name: str = Field(default="blatam-academy", description="Service name")
    
    # Metrics collection
    collect_metrics: bool = Field(default=True, description="Collect metrics")
    metrics_prefix: str = Field(default="blatam_academy", description="Metrics prefix")
    metrics_interval: int = Field(default=60, description="Metrics collection interval in seconds")
    
    # Performance monitoring
    track_response_times: bool = Field(default=True, description="Track response times")
    track_memory_usage: bool = Field(default=True, description="Track memory usage")
    track_cpu_usage: bool = Field(default=True, description="Track CPU usage")
    track_request_rates: bool = Field(default=True, description="Track request rates")
    
    # Error monitoring
    track_error_rates: bool = Field(default=True, description="Track error rates")
    track_error_types: bool = Field(default=True, description="Track error types")
    track_slow_requests: bool = Field(default=True, description="Track slow requests")
    
    # Alerting
    enable_alerts: bool = Field(default=True, description="Enable alerting")
    alert_thresholds: Dict[str, float] = Field(
        default={
            "error_rate": 0.05,  # 5% error rate
            "response_time_p95": 2.0,  # 95th percentile response time
            "response_time_p99": 5.0,  # 99th percentile response time
            "memory_usage": 0.8,  # 80% memory usage
            "cpu_usage": 0.8,  # 80% CPU usage
            "slow_request_rate": 0.1,  # 10% slow requests
        },
        description="Alert thresholds"
    )
    
    # Storage
    redis_enabled: bool = Field(default=False, description="Enable Redis for metrics storage")
    redis_url: str = Field(default="redis://localhost:6379", description="Redis URL")
    redis_ttl: int = Field(default=86400, description="Redis TTL in seconds")
    
    # Retention
    metrics_retention_days: int = Field(default=30, description="Metrics retention in days")
    alert_retention_days: int = Field(default=7, description="Alert retention in days")

# =============================================================================
# METRICS COLLECTION
# =============================================================================

@dataclass
class PerformanceMetrics:
    """Performance metrics data structure."""
    timestamp: datetime
    response_time_p50: float
    response_time_p95: float
    response_time_p99: float
    request_rate: float
    error_rate: float
    memory_usage: float
    cpu_usage: float
    active_requests: int
    slow_request_rate: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ErrorMetrics:
    """Error metrics data structure."""
    timestamp: datetime
    total_errors: int
    error_rate: float
    error_types: Dict[str, int]
    error_endpoints: Dict[str, int]
    error_categories: Dict[str, int]
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Alert:
    """Alert data structure."""
    id: str
    type: AlertType
    level: AlertLevel
    message: str
    timestamp: datetime
    value: float
    threshold: float
    metadata: Dict[str, Any] = field(default_factory=dict)

class MetricsCollector:
    """Collector for various metrics."""
    
    def __init__(self, config: MonitoringConfig):
        
    """__init__ function."""
self.config = config
        self.prefix = config.metrics_prefix
        
        # Prometheus metrics
        self.request_total = Counter(
            f'{self.prefix}_requests_total',
            'Total requests',
            ['method', 'endpoint', 'status_code', 'error_type']
        )
        
        self.request_duration = Histogram(
            f'{self.prefix}_request_duration_seconds',
            'Request duration in seconds',
            ['method', 'endpoint', 'status_code']
        )
        
        self.error_total = Counter(
            f'{self.prefix}_errors_total',
            'Total errors',
            ['method', 'endpoint', 'error_category', 'error_code']
        )
        
        self.error_rate = Gauge(
            f'{self.prefix}_error_rate',
            'Error rate percentage',
            ['method', 'endpoint']
        )
        
        self.response_size = Histogram(
            f'{self.prefix}_response_size_bytes',
            'Response size in bytes',
            ['method', 'endpoint']
        )
        
        self.active_requests = Gauge(
            f'{self.prefix}_active_requests',
            'Number of active requests',
            ['method', 'endpoint']
        )
        
        self.memory_usage = Gauge(
            f'{self.prefix}_memory_usage_bytes',
            'Memory usage in bytes'
        )
        
        self.cpu_usage = Gauge(
            f'{self.prefix}_cpu_usage_percent',
            'CPU usage percentage'
        )
        
        self.slow_requests = Counter(
            f'{self.prefix}_slow_requests_total',
            'Total slow requests',
            ['method', 'endpoint', 'duration_range']
        )
        
        # Internal storage
        self.response_times = defaultdict(list)
        self.error_counts = defaultdict(int)
        self.request_counts = defaultdict(int)
        self.start_time = time.time()
        
        # Performance tracking
        self.performance_history = deque(maxlen=1000)
        self.error_history = deque(maxlen=1000)
    
    def record_request(self, method: str, endpoint: str, status_code: int, duration: float, response_size: int):
        """Record a request."""
        error_type = "error" if status_code >= 400 else "success"
        
        # Update Prometheus metrics
        self.request_total.labels(
            method=method,
            endpoint=endpoint,
            status_code=status_code,
            error_type=error_type
        ).inc()
        
        self.request_duration.labels(
            method=method,
            endpoint=endpoint,
            status_code=status_code
        ).observe(duration)
        
        self.response_size.labels(
            method=method,
            endpoint=endpoint
        ).observe(response_size)
        
        # Update internal metrics
        key = f"{method}:{endpoint}"
        self.response_times[key].append(duration)
        self.request_counts[key] += 1
        
        # Track slow requests
        if duration > self.config.alert_thresholds.get("response_time_p95", 2.0):
            duration_range = self._get_duration_range(duration)
            self.slow_requests.labels(
                method=method,
                endpoint=endpoint,
                duration_range=duration_range
            ).inc()
    
    def record_error(self, method: str, endpoint: str, error_category: str, error_code: str):
        """Record an error."""
        # Update Prometheus metrics
        self.error_total.labels(
            method=method,
            endpoint=endpoint,
            error_category=error_category,
            error_code=error_code
        ).inc()
        
        # Update internal metrics
        key = f"{method}:{endpoint}"
        self.error_counts[key] += 1
        
        # Update error rate
        total_requests = self.request_counts.get(key, 0)
        if total_requests > 0:
            error_rate = self.error_counts[key] / total_requests
            self.error_rate.labels(
                method=method,
                endpoint=endpoint
            ).set(error_rate)
    
    def record_system_metrics(self, memory_usage: float, cpu_usage: float):
        """Record system metrics."""
        self.memory_usage.set(memory_usage)
        self.cpu_usage.set(cpu_usage)
    
    def get_performance_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics."""
        current_time = datetime.utcnow()
        
        # Calculate response time percentiles
        all_response_times = []
        for times in self.response_times.values():
            all_response_times.extend(times)
        
        if all_response_times:
            response_time_p50 = statistics.median(all_response_times)
            response_time_p95 = statistics.quantiles(all_response_times, n=20)[18]  # 95th percentile
            response_time_p99 = statistics.quantiles(all_response_times, n=100)[98]  # 99th percentile
        else:
            response_time_p50 = response_time_p95 = response_time_p99 = 0.0
        
        # Calculate rates
        uptime = time.time() - self.start_time
        total_requests = sum(self.request_counts.values())
        total_errors = sum(self.error_counts.values())
        
        request_rate = total_requests / max(uptime, 1)
        error_rate = total_errors / max(total_requests, 1)
        
        # Calculate slow request rate
        slow_requests = sum(1 for times in self.response_times.values() 
                          for t in times if t > self.config.alert_thresholds.get("response_time_p95", 2.0))
        slow_request_rate = slow_requests / max(total_requests, 1)
        
        return PerformanceMetrics(
            timestamp=current_time,
            response_time_p50=response_time_p50,
            response_time_p95=response_time_p95,
            response_time_p99=response_time_p99,
            request_rate=request_rate,
            error_rate=error_rate,
            memory_usage=0.0,  # Will be updated separately
            cpu_usage=0.0,     # Will be updated separately
            active_requests=0,  # Will be updated separately
            slow_request_rate=slow_request_rate
        )
    
    def get_error_metrics(self) -> ErrorMetrics:
        """Get current error metrics."""
        current_time = datetime.utcnow()
        
        # Aggregate error types
        error_types = defaultdict(int)
        error_endpoints = defaultdict(int)
        error_categories = defaultdict(int)
        
        total_errors = sum(self.error_counts.values())
        total_requests = sum(self.request_counts.values())
        error_rate = total_errors / max(total_requests, 1)
        
        return ErrorMetrics(
            timestamp=current_time,
            total_errors=total_errors,
            error_rate=error_rate,
            error_types=dict(error_types),
            error_endpoints=dict(error_endpoints),
            error_categories=dict(error_categories)
        )
    
    def _get_duration_range(self, duration: float) -> str:
        """Get duration range for metrics."""
        if duration < 1.0:
            return "0-1s"
        elif duration < 5.0:
            return "1-5s"
        elif duration < 10.0:
            return "5-10s"
        else:
            return "10s+"

# =============================================================================
# ALERTING SYSTEM
# =============================================================================

class AlertManager:
    """Manager for handling alerts and notifications."""
    
    def __init__(self, config: MonitoringConfig):
        
    """__init__ function."""
self.config = config
        self.logger = structlog.get_logger(__name__)
        self.alerts = deque(maxlen=1000)
        self.alert_handlers: List[Callable[[Alert], None]] = []
        self.silenced_alerts: Dict[str, datetime] = {}
    
    def add_alert_handler(self, handler: Callable[[Alert], None]):
        """Add an alert handler."""
        self.alert_handlers.append(handler)
    
    def check_alerts(self, metrics: PerformanceMetrics, error_metrics: ErrorMetrics) -> List[Alert]:
        """Check for alerts based on current metrics."""
        alerts = []
        
        # Check error rate
        if error_metrics.error_rate > self.config.alert_thresholds.get("error_rate", 0.05):
            alert = Alert(
                id=f"error_rate_{int(time.time())}",
                type=AlertType.ERROR_RATE,
                level=AlertLevel.ERROR if error_metrics.error_rate > 0.1 else AlertLevel.WARNING,
                message=f"High error rate detected: {error_metrics.error_rate:.2%}",
                timestamp=datetime.utcnow(),
                value=error_metrics.error_rate,
                threshold=self.config.alert_thresholds.get("error_rate", 0.05)
            )
            alerts.append(alert)
        
        # Check response time
        if metrics.response_time_p95 > self.config.alert_thresholds.get("response_time_p95", 2.0):
            alert = Alert(
                id=f"response_time_{int(time.time())}",
                type=AlertType.RESPONSE_TIME,
                level=AlertLevel.WARNING,
                message=f"High response time detected: {metrics.response_time_p95:.2f}s (95th percentile)",
                timestamp=datetime.utcnow(),
                value=metrics.response_time_p95,
                threshold=self.config.alert_thresholds.get("response_time_p95", 2.0)
            )
            alerts.append(alert)
        
        # Check slow request rate
        if metrics.slow_request_rate > self.config.alert_thresholds.get("slow_request_rate", 0.1):
            alert = Alert(
                id=f"slow_requests_{int(time.time())}",
                type=AlertType.SLOW_REQUEST,
                level=AlertLevel.WARNING,
                message=f"High slow request rate: {metrics.slow_request_rate:.2%}",
                timestamp=datetime.utcnow(),
                value=metrics.slow_request_rate,
                threshold=self.config.alert_thresholds.get("slow_request_rate", 0.1)
            )
            alerts.append(alert)
        
        # Check memory usage
        if metrics.memory_usage > self.config.alert_thresholds.get("memory_usage", 0.8):
            alert = Alert(
                id=f"memory_usage_{int(time.time())}",
                type=AlertType.MEMORY_USAGE,
                level=AlertLevel.WARNING,
                message=f"High memory usage: {metrics.memory_usage:.2%}",
                timestamp=datetime.utcnow(),
                value=metrics.memory_usage,
                threshold=self.config.alert_thresholds.get("memory_usage", 0.8)
            )
            alerts.append(alert)
        
        # Check CPU usage
        if metrics.cpu_usage > self.config.alert_thresholds.get("cpu_usage", 0.8):
            alert = Alert(
                id=f"cpu_usage_{int(time.time())}",
                type=AlertType.CPU_USAGE,
                level=AlertLevel.WARNING,
                message=f"High CPU usage: {metrics.cpu_usage:.2%}",
                timestamp=datetime.utcnow(),
                value=metrics.cpu_usage,
                threshold=self.config.alert_thresholds.get("cpu_usage", 0.8)
            )
            alerts.append(alert)
        
        # Process alerts
        for alert in alerts:
            self._process_alert(alert)
        
        return alerts
    
    def _process_alert(self, alert: Alert):
        """Process an alert."""
        # Check if alert is silenced
        if alert.id in self.silenced_alerts:
            if datetime.utcnow() < self.silenced_alerts[alert.id]:
                return  # Alert is silenced
            else:
                del self.silenced_alerts[alert.id]
        
        # Add to alert history
        self.alerts.append(alert)
        
        # Log alert
        self.logger.warning(
            "Alert triggered",
            alert_id=alert.id,
            alert_type=alert.type.value,
            alert_level=alert.level.value,
            message=alert.message,
            value=alert.value,
            threshold=alert.threshold
        )
        
        # Send to handlers
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                self.logger.error("Alert handler failed", handler=handler.__name__, error=str(e))
    
    def silence_alert(self, alert_id: str, duration_minutes: int = 60):
        """Silence an alert for a specified duration."""
        self.silenced_alerts[alert_id] = datetime.utcnow() + timedelta(minutes=duration_minutes)
        self.logger.info("Alert silenced", alert_id=alert_id, duration_minutes=duration_minutes)
    
    def get_alerts(self, hours: int = 24) -> List[Alert]:
        """Get alerts from the last N hours."""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        return [alert for alert in self.alerts if alert.timestamp > cutoff]
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get alert summary."""
        total_alerts = len(self.alerts)
        alerts_by_level = defaultdict(int)
        alerts_by_type = defaultdict(int)
        
        for alert in self.alerts:
            alerts_by_level[alert.level.value] += 1
            alerts_by_type[alert.type.value] += 1
        
        return {
            "total_alerts": total_alerts,
            "alerts_by_level": dict(alerts_by_level),
            "alerts_by_type": dict(alerts_by_type),
            "silenced_alerts": len(self.silenced_alerts)
        }

# =============================================================================
# MONITORING MIDDLEWARE
# =============================================================================

class MonitoringMiddleware:
    """Middleware for comprehensive monitoring."""
    
    def __init__(self, config: MonitoringConfig, redis_client: Optional[redis.Redis] = None):
        
    """__init__ function."""
self.config = config
        self.redis_client = redis_client
        self.logger = structlog.get_logger(__name__)
        
        # Initialize components
        self.metrics_collector = MetricsCollector(config)
        self.alert_manager = AlertManager(config)
        
        # Performance tracking
        self.start_time = time.time()
        self.request_count = 0
        self.error_count = 0
        
        # Setup alert handlers
        self._setup_alert_handlers()
    
    def _setup_alert_handlers(self) -> Any:
        """Setup default alert handlers."""
        # Log alert handler
        def log_alert_handler(alert: Alert):
            
    """log_alert_handler function."""
log_level = {
                AlertLevel.INFO: self.logger.info,
                AlertLevel.WARNING: self.logger.warning,
                AlertLevel.ERROR: self.logger.error,
                AlertLevel.CRITICAL: self.logger.critical
            }
            
            log_func = log_level.get(alert.level, self.logger.warning)
            log_func(
                "Alert triggered",
                alert_id=alert.id,
                alert_type=alert.type.value,
                alert_level=alert.level.value,
                message=alert.message,
                value=alert.value,
                threshold=alert.threshold
            )
        
        self.alert_manager.add_alert_handler(log_alert_handler)
        
        # Redis alert handler (if Redis is available)
        if self.redis_client:
            async def redis_alert_handler(alert: Alert):
                
    """redis_alert_handler function."""
try:
                    alert_data = {
                        "id": alert.id,
                        "type": alert.type.value,
                        "level": alert.level.value,
                        "message": alert.message,
                        "timestamp": alert.timestamp.isoformat(),
                        "value": alert.value,
                        "threshold": alert.threshold,
                        "metadata": alert.metadata
                    }
                    
                    await self.redis_client.lpush(
                        f"alerts:{self.config.service_name}",
                        json.dumps(alert_data)
                    )
                    await self.redis_client.expire(
                        f"alerts:{self.config.service_name}",
                        self.config.alert_retention_days * 86400
                    )
                except Exception as e:
                    self.logger.error("Failed to store alert in Redis", error=str(e))
            
            self.alert_manager.add_alert_handler(redis_alert_handler)
    
    async def record_request(self, request: Request, response: Response, duration: float):
        """Record a request for monitoring."""
        if not self.config.enabled:
            return
        
        self.request_count += 1
        
        # Extract request info
        method = request.method
        endpoint = request.url.path
        status_code = response.status_code
        response_size = len(response.body) if hasattr(response, 'body') and response.body else 0
        
        # Record metrics
        self.metrics_collector.record_request(method, endpoint, status_code, duration, response_size)
        
        # Check for errors
        if status_code >= 400:
            self.error_count += 1
            error_category = self._get_error_category(status_code)
            error_code = f"HTTP_{status_code}"
            self.metrics_collector.record_error(method, endpoint, error_category, error_code)
    
    async def record_error(self, request: Request, error: Exception, duration: float):
        """Record an error for monitoring."""
        if not self.config.enabled:
            return
        
        self.error_count += 1
        
        # Extract error info
        method = request.method
        endpoint = request.url.path
        error_category = self._get_error_category_from_exception(error)
        error_code = type(error).__name__
        
        # Record metrics
        self.metrics_collector.record_error(method, endpoint, error_category, error_code)
    
    def _get_error_category(self, status_code: int) -> str:
        """Get error category from status code."""
        if 400 <= status_code < 500:
            return "client_error"
        elif 500 <= status_code < 600:
            return "server_error"
        else:
            return "unknown"
    
    def _get_error_category_from_exception(self, error: Exception) -> str:
        """Get error category from exception."""
        if isinstance(error, (ValueError, TypeError)):
            return "validation_error"
        elif isinstance(error, (ConnectionError, TimeoutError)):
            return "connection_error"
        elif isinstance(error, (PermissionError, OSError)):
            return "system_error"
        else:
            return "unknown"
    
    async def collect_system_metrics(self) -> Any:
        """Collect system metrics."""
        if not self.config.enabled:
            return
        
        try:
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_usage = memory.percent / 100.0
            
            # CPU usage
            cpu_usage = psutil.cpu_percent(interval=1) / 100.0
            
            # Record metrics
            self.metrics_collector.record_system_metrics(memory_usage, cpu_usage)
            
        except ImportError:
            self.logger.warning("psutil not available, system metrics not collected")
        except Exception as e:
            self.logger.error("Failed to collect system metrics", error=str(e))
    
    async def check_alerts(self) -> Any:
        """Check for alerts based on current metrics."""
        if not self.config.enabled or not self.config.enable_alerts:
            return []
        
        # Get current metrics
        performance_metrics = self.metrics_collector.get_performance_metrics()
        error_metrics = self.metrics_collector.get_error_metrics()
        
        # Update system metrics
        await self.collect_system_metrics()
        
        # Check for alerts
        alerts = self.alert_manager.check_alerts(performance_metrics, error_metrics)
        
        return alerts
    
    def get_metrics(self) -> str:
        """Get Prometheus metrics."""
        return generate_latest(REGISTRY)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        if not self.config.enabled:
            return {}
        
        uptime = time.time() - self.start_time
        performance_metrics = self.metrics_collector.get_performance_metrics()
        error_metrics = self.metrics_collector.get_error_metrics()
        
        return {
            "uptime_seconds": uptime,
            "uptime_formatted": str(timedelta(seconds=int(uptime))),
            "request_count": self.request_count,
            "error_count": self.error_count,
            "error_rate": error_metrics.error_rate,
            "request_rate": performance_metrics.request_rate,
            "response_time_p50": performance_metrics.response_time_p50,
            "response_time_p95": performance_metrics.response_time_p95,
            "response_time_p99": performance_metrics.response_time_p99,
            "slow_request_rate": performance_metrics.slow_request_rate,
            "memory_usage": performance_metrics.memory_usage,
            "cpu_usage": performance_metrics.cpu_usage
        }
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get error summary."""
        if not self.config.enabled:
            return {}
        
        error_metrics = self.metrics_collector.get_error_metrics()
        alert_summary = self.alert_manager.get_alert_summary()
        
        return {
            "total_errors": error_metrics.total_errors,
            "error_rate": error_metrics.error_rate,
            "error_types": error_metrics.error_types,
            "error_endpoints": error_metrics.error_endpoints,
            "error_categories": error_metrics.error_categories,
            "alerts": alert_summary
        }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status."""
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "uptime": time.time() - self.start_time,
            "monitoring_enabled": self.config.enabled,
            "metrics_enabled": self.config.collect_metrics,
            "alerting_enabled": self.config.enable_alerts,
            "redis_connected": self.redis_client is not None
        }

# =============================================================================
# MONITORING MANAGER
# =============================================================================

class MonitoringManager:
    """Manager for the complete monitoring system."""
    
    def __init__(self, config: MonitoringConfig, redis_client: Optional[redis.Redis] = None):
        
    """__init__ function."""
self.config = config
        self.redis_client = redis_client
        self.logger = structlog.get_logger(__name__)
        
        # Initialize monitoring middleware
        self.monitoring_middleware = MonitoringMiddleware(config, redis_client)
        
        # Background tasks
        self.metrics_task: Optional[asyncio.Task] = None
        self.alert_task: Optional[asyncio.Task] = None
        
        # Performance tracking
        self.start_time = time.time()
    
    async def start(self) -> Any:
        """Start monitoring system."""
        if not self.config.enabled:
            self.logger.info("Monitoring disabled")
            return
        
        self.logger.info("Starting monitoring system", config=self.config.dict())
        
        # Start background tasks
        if self.config.collect_metrics:
            self.metrics_task = asyncio.create_task(self._metrics_collector())
        
        if self.config.enable_alerts:
            self.alert_task = asyncio.create_task(self._alert_checker())
        
        self.logger.info("Monitoring system started")
    
    async def stop(self) -> Any:
        """Stop monitoring system."""
        if not self.config.enabled:
            return
        
        self.logger.info("Stopping monitoring system")
        
        # Cancel background tasks
        if self.metrics_task:
            self.metrics_task.cancel()
            try:
                await self.metrics_task
            except asyncio.CancelledError:
                pass
        
        if self.alert_task:
            self.alert_task.cancel()
            try:
                await self.alert_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Monitoring system stopped")
    
    async def _metrics_collector(self) -> Any:
        """Background task for collecting metrics."""
        while True:
            try:
                await self.monitoring_middleware.collect_system_metrics()
                await asyncio.sleep(self.config.metrics_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Metrics collection failed", error=str(e))
                await asyncio.sleep(60)  # Wait before retrying
    
    async def _alert_checker(self) -> Any:
        """Background task for checking alerts."""
        while True:
            try:
                await self.monitoring_middleware.check_alerts()
                await asyncio.sleep(60)  # Check alerts every minute
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Alert checking failed", error=str(e))
                await asyncio.sleep(60)  # Wait before retrying
    
    def get_metrics(self) -> str:
        """Get Prometheus metrics."""
        return self.monitoring_middleware.get_metrics()
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        return self.monitoring_middleware.get_performance_summary()
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get error summary."""
        return self.monitoring_middleware.get_error_summary()
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status."""
        return self.monitoring_middleware.get_health_status()
    
    def get_alerts(self, hours: int = 24) -> List[Alert]:
        """Get alerts from the last N hours."""
        return self.monitoring_middleware.alert_manager.get_alerts(hours)
    
    def silence_alert(self, alert_id: str, duration_minutes: int = 60):
        """Silence an alert for a specified duration."""
        self.monitoring_middleware.alert_manager.silence_alert(alert_id, duration_minutes)

# =============================================================================
# CONFIGURATION FACTORIES
# =============================================================================

def create_monitoring_config(
    environment: str = "production",
    collect_metrics: bool = True,
    enable_alerts: bool = True,
    **kwargs
) -> MonitoringConfig:
    """Create monitoring configuration with sensible defaults."""
    return MonitoringConfig(
        environment=environment,
        collect_metrics=collect_metrics,
        enable_alerts=enable_alerts,
        **kwargs
    )

def create_production_monitoring_config() -> MonitoringConfig:
    """Create production-optimized monitoring configuration."""
    return MonitoringConfig(
        environment="production",
        collect_metrics=True,
        enable_alerts=True,
        alert_thresholds={
            "error_rate": 0.05,
            "response_time_p95": 2.0,
            "response_time_p99": 5.0,
            "memory_usage": 0.8,
            "cpu_usage": 0.8,
            "slow_request_rate": 0.1,
        },
        metrics_interval=60,
        redis_enabled=True
    )

def create_development_monitoring_config() -> MonitoringConfig:
    """Create development-optimized monitoring configuration."""
    return MonitoringConfig(
        environment="development",
        collect_metrics=True,
        enable_alerts=False,  # No alerts in development
        alert_thresholds={
            "error_rate": 0.1,  # Higher threshold in development
            "response_time_p95": 5.0,  # Higher threshold in development
            "response_time_p99": 10.0,
            "memory_usage": 0.9,
            "cpu_usage": 0.9,
            "slow_request_rate": 0.2,
        },
        metrics_interval=30,  # More frequent in development
        redis_enabled=False
    )

# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

async def setup_monitoring(
    app: FastAPI,
    config: Optional[MonitoringConfig] = None,
    redis_client: Optional[redis.Redis] = None
) -> MonitoringManager:
    """
    Setup monitoring for a FastAPI application.
    
    Args:
        app: FastAPI application
        config: Monitoring configuration (optional)
        redis_client: Redis client for metrics storage (optional)
        
    Returns:
        MonitoringManager instance
    """
    if config is None:
        config = create_monitoring_config()
    
    manager = MonitoringManager(config, redis_client)
    await manager.start()
    
    # Add monitoring endpoints
    @app.get("/health")
    async def health():
        
    """health function."""
return manager.get_health_status()
    
    @app.get("/metrics")
    async def metrics():
        
    """metrics function."""
return manager.get_metrics()
    
    @app.get("/monitoring/performance")
    async def performance():
        
    """performance function."""
return manager.get_performance_summary()
    
    @app.get("/monitoring/errors")
    async def errors():
        
    """errors function."""
return manager.get_error_summary()
    
    @app.get("/monitoring/alerts")
    async def alerts(hours: int = 24):
        
    """alerts function."""
return manager.get_alerts(hours)
    
    @app.post("/monitoring/alerts/{alert_id}/silence")
    async def silence_alert(alert_id: str, duration_minutes: int = 60):
        
    """silence_alert function."""
manager.silence_alert(alert_id, duration_minutes)
        return {"message": f"Alert {alert_id} silenced for {duration_minutes} minutes"}
    
    return manager

# Example usage
async def example_usage():
    """Example of how to use the monitoring system."""
    
    
    # Create app
    app = FastAPI(title="Monitoring Example")
    
    # Create configuration
    config = create_production_monitoring_config()
    
    # Setup monitoring
    manager = await setup_monitoring(app, config)
    
    # Add test endpoints
    @app.get("/")
    async def root():
        
    """root function."""
return {"message": "Monitoring example"}
    
    @app.get("/test/error")
    async def test_error():
        
    """test_error function."""
raise ValueError("Test error for monitoring")
    
    @app.get("/test/slow")
    async def test_slow():
        
    """test_slow function."""
await asyncio.sleep(3)  # Simulate slow request
        return {"message": "Slow request completed"}
    
    return app

match __name__:
    case "__main__":
    asyncio.run(example_usage()) 