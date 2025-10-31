from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

# Constants
BUFFER_SIZE = 1024

import asyncio
import time
import statistics
from typing import Dict, List, Any, Optional, Union, Callable, Awaitable
from datetime import datetime, timezone, timedelta
from contextlib import asynccontextmanager
import structlog
from dataclasses import dataclass, asdict, field
from enum import Enum
import weakref
from functools import lru_cache, wraps
import inspect
import gc
import json
from collections import defaultdict, deque
import threading
from concurrent.futures import ThreadPoolExecutor
import psutil
import aiohttp
from prometheus_client import Counter, Histogram, Gauge, Summary, generate_latest
from fastapi import Request, Response
from fastapi.middleware.base import BaseHTTPMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import Any, List, Dict, Optional
import logging
#!/usr/bin/env python3
"""
API Performance Monitor for HeyGen AI FastAPI
Comprehensive performance monitoring system for tracking response time, latency, and throughput.
"""



logger = structlog.get_logger()

# =============================================================================
# Performance Types
# =============================================================================

class MetricType(Enum):
    """Metric type enumeration."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"
    TIMER = "timer"

class PerformanceLevel(Enum):
    """Performance level enumeration."""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    CRITICAL = "critical"

@dataclass
class PerformanceThresholds:
    """Performance thresholds configuration."""
    response_time_ms: Dict[PerformanceLevel, float] = field(default_factory=lambda: {
        PerformanceLevel.EXCELLENT: 100,
        PerformanceLevel.GOOD: 300,
        PerformanceLevel.ACCEPTABLE: 500,
        PerformanceLevel.POOR: 1000,
        PerformanceLevel.CRITICAL: 2000
    })
    throughput_rps: Dict[PerformanceLevel, float] = field(default_factory=lambda: {
        PerformanceLevel.EXCELLENT: 1000,
        PerformanceLevel.GOOD: 500,
        PerformanceLevel.ACCEPTABLE: 100,
        PerformanceLevel.POOR: 50,
        PerformanceLevel.CRITICAL: 10
    })
    error_rate_percent: Dict[PerformanceLevel, float] = field(default_factory=lambda: {
        PerformanceLevel.EXCELLENT: 0.1,
        PerformanceLevel.GOOD: 1.0,
        PerformanceLevel.ACCEPTABLE: 5.0,
        PerformanceLevel.POOR: 10.0,
        PerformanceLevel.CRITICAL: 25.0
    })
    cpu_usage_percent: Dict[PerformanceLevel, float] = field(default_factory=lambda: {
        PerformanceLevel.EXCELLENT: 20,
        PerformanceLevel.GOOD: 50,
        PerformanceLevel.ACCEPTABLE: 70,
        PerformanceLevel.POOR: 85,
        PerformanceLevel.CRITICAL: 95
    })
    memory_usage_percent: Dict[PerformanceLevel, float] = field(default_factory=lambda: {
        PerformanceLevel.EXCELLENT: 30,
        PerformanceLevel.GOOD: 60,
        PerformanceLevel.ACCEPTABLE: 80,
        PerformanceLevel.POOR: 90,
        PerformanceLevel.CRITICAL: 95
    })

@dataclass
class RequestMetrics:
    """Request-level performance metrics."""
    request_id: str
    method: str
    path: str
    status_code: int
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    request_size_bytes: int = 0
    response_size_bytes: int = 0
    user_agent: Optional[str] = None
    ip_address: Optional[str] = None
    user_id: Optional[str] = None
    error_message: Optional[str] = None
    database_queries: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    external_api_calls: int = 0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0

@dataclass
class EndpointMetrics:
    """Endpoint-level performance metrics."""
    method: str
    path: str
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_duration_ms: float = 0.0
    min_duration_ms: float = float('inf')
    max_duration_ms: float = 0.0
    avg_duration_ms: float = 0.0
    p50_duration_ms: float = 0.0
    p95_duration_ms: float = 0.0
    p99_duration_ms: float = 0.0
    total_request_size_bytes: int = 0
    total_response_size_bytes: int = 0
    avg_request_size_bytes: float = 0.0
    avg_response_size_bytes: float = 0.0
    total_database_queries: int = 0
    total_cache_hits: int = 0
    total_cache_misses: int = 0
    total_external_api_calls: int = 0
    last_request_time: Optional[datetime] = None
    first_request_time: Optional[datetime] = None

@dataclass
class SystemMetrics:
    """System-level performance metrics."""
    timestamp: datetime
    cpu_usage_percent: float
    memory_usage_percent: float
    memory_available_mb: float
    disk_usage_percent: float
    network_bytes_sent: int
    network_bytes_received: int
    active_connections: int
    total_requests_per_second: float
    avg_response_time_ms: float
    error_rate_percent: float
    throughput_rps: float

# =============================================================================
# Performance Monitor
# =============================================================================

class PerformanceMonitor:
    """Main performance monitoring system."""
    
    def __init__(self, thresholds: Optional[PerformanceThresholds] = None):
        
    """__init__ function."""
self.thresholds = thresholds or PerformanceThresholds()
        self.request_metrics: Dict[str, RequestMetrics] = {}
        self.endpoint_metrics: Dict[str, EndpointMetrics] = {}
        self.system_metrics: List[SystemMetrics] = []
        self.performance_alerts: List[Dict[str, Any]] = []
        
        # Thread-safe collections
        self._request_metrics_lock = threading.Lock()
        self._endpoint_metrics_lock = threading.Lock()
        self._system_metrics_lock = threading.Lock()
        
        # Rolling windows for real-time metrics
        self._request_window = deque(maxlen=1000)
        self._system_window = deque(maxlen=100)
        
        # Prometheus metrics
        self._setup_prometheus_metrics()
        
        # Background tasks
        self._monitoring_task: Optional[asyncio.Task] = None
        self._alerting_task: Optional[asyncio.Task] = None
        self._is_running = False
    
    def _setup_prometheus_metrics(self) -> Any:
        """Setup Prometheus metrics."""
        # Request metrics
        self.request_duration = Histogram(
            'http_request_duration_seconds',
            'HTTP request duration in seconds',
            ['method', 'endpoint', 'status_code']
        )
        
        self.request_total = Counter(
            'http_requests_total',
            'Total HTTP requests',
            ['method', 'endpoint', 'status_code']
        )
        
        self.request_size = Histogram(
            'http_request_size_bytes',
            'HTTP request size in bytes',
            ['method', 'endpoint']
        )
        
        self.response_size = Histogram(
            'http_response_size_bytes',
            'HTTP response size in bytes',
            ['method', 'endpoint', 'status_code']
        )
        
        # System metrics
        self.cpu_usage = Gauge(
            'system_cpu_usage_percent',
            'System CPU usage percentage'
        )
        
        self.memory_usage = Gauge(
            'system_memory_usage_percent',
            'System memory usage percentage'
        )
        
        self.active_connections = Gauge(
            'system_active_connections',
            'Number of active connections'
        )
        
        self.throughput = Gauge(
            'system_throughput_rps',
            'System throughput in requests per second'
        )
        
        self.error_rate = Gauge(
            'system_error_rate_percent',
            'System error rate percentage'
        )
        
        # Database metrics
        self.database_queries = Counter(
            'database_queries_total',
            'Total database queries',
            ['operation', 'table']
        )
        
        self.database_duration = Histogram(
            'database_query_duration_seconds',
            'Database query duration in seconds',
            ['operation', 'table']
        )
        
        # Cache metrics
        self.cache_hits = Counter(
            'cache_hits_total',
            'Total cache hits',
            ['cache_name']
        )
        
        self.cache_misses = Counter(
            'cache_misses_total',
            'Total cache misses',
            ['cache_name']
        )
        
        # External API metrics
        self.external_api_calls = Counter(
            'external_api_calls_total',
            'Total external API calls',
            ['api_name', 'endpoint']
        )
        
        self.external_api_duration = Histogram(
            'external_api_duration_seconds',
            'External API call duration in seconds',
            ['api_name', 'endpoint']
        )
    
    async def start_monitoring(self) -> Any:
        """Start performance monitoring."""
        if self._is_running:
            return
        
        self._is_running = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        self._alerting_task = asyncio.create_task(self._alerting_loop())
        
        logger.info("Performance monitoring started")
    
    async def stop_monitoring(self) -> Any:
        """Stop performance monitoring."""
        if not self._is_running:
            return
        
        self._is_running = False
        
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        if self._alerting_task:
            self._alerting_task.cancel()
            try:
                await self._alerting_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Performance monitoring stopped")
    
    async def _monitoring_loop(self) -> Any:
        """System monitoring loop."""
        while self._is_running:
            try:
                system_metrics = await self._collect_system_metrics()
                
                with self._system_metrics_lock:
                    self.system_metrics.append(system_metrics)
                    self._system_window.append(system_metrics)
                
                # Update Prometheus metrics
                self._update_prometheus_metrics(system_metrics)
                
                await asyncio.sleep(10)  # Collect metrics every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(10)
    
    async def _alerting_loop(self) -> Any:
        """Performance alerting loop."""
        while self._is_running:
            try:
                await self._check_performance_alerts()
                await asyncio.sleep(30)  # Check alerts every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in alerting loop: {e}")
                await asyncio.sleep(30)
    
    async def _collect_system_metrics(self) -> SystemMetrics:
        """Collect system-level metrics."""
        # CPU and memory usage
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        # Disk usage
        disk = psutil.disk_usage('/')
        
        # Network usage
        network = psutil.net_io_counters()
        
        # Calculate request metrics
        total_rps = self._calculate_throughput()
        avg_response_time = self._calculate_avg_response_time()
        error_rate = self._calculate_error_rate()
        
        return SystemMetrics(
            timestamp=datetime.now(timezone.utc),
            cpu_usage_percent=cpu_percent,
            memory_usage_percent=memory.percent,
            memory_available_mb=memory.available / (1024 * 1024),
            disk_usage_percent=disk.percent,
            network_bytes_sent=network.bytes_sent,
            network_bytes_received=network.bytes_recv,
            active_connections=len(self.request_metrics),
            total_requests_per_second=total_rps,
            avg_response_time_ms=avg_response_time,
            error_rate_percent=error_rate,
            throughput_rps=total_rps
        )
    
    def _calculate_throughput(self) -> float:
        """Calculate current throughput (requests per second)."""
        if not self._request_window:
            return 0.0
        
        # Calculate RPS based on last minute
        now = datetime.now(timezone.utc)
        one_minute_ago = now - timedelta(minutes=1)
        
        recent_requests = [
            req for req in self._request_window
            if req.end_time and req.end_time > one_minute_ago
        ]
        
        return len(recent_requests) / 60.0
    
    def _calculate_avg_response_time(self) -> float:
        """Calculate average response time."""
        if not self._request_window:
            return 0.0
        
        recent_requests = list(self._request_window)[-100:]  # Last 100 requests
        durations = [req.duration_ms for req in recent_requests if req.duration_ms is not None]
        
        return statistics.mean(durations) if durations else 0.0
    
    def _calculate_error_rate(self) -> float:
        """Calculate error rate percentage."""
        if not self._request_window:
            return 0.0
        
        recent_requests = list(self._request_window)[-100:]  # Last 100 requests
        total_requests = len(recent_requests)
        
        if total_requests == 0:
            return 0.0
        
        error_requests = sum(1 for req in recent_requests if req.status_code >= 400)
        return (error_requests / total_requests) * 100.0
    
    def _update_prometheus_metrics(self, system_metrics: SystemMetrics):
        """Update Prometheus metrics."""
        self.cpu_usage.set(system_metrics.cpu_usage_percent)
        self.memory_usage.set(system_metrics.memory_usage_percent)
        self.active_connections.set(system_metrics.active_connections)
        self.throughput.set(system_metrics.throughput_rps)
        self.error_rate.set(system_metrics.error_rate_percent)
    
    async def _check_performance_alerts(self) -> Any:
        """Check for performance alerts."""
        if not self.system_metrics:
            return
        
        latest_metrics = self.system_metrics[-1]
        
        # Check response time
        if latest_metrics.avg_response_time_ms > self.thresholds.response_time_ms[PerformanceLevel.POOR]:
            await self._create_alert(
                "High Response Time",
                f"Average response time is {latest_metrics.avg_response_time_ms:.2f}ms",
                "warning"
            )
        
        # Check throughput
        if latest_metrics.throughput_rps < self.thresholds.throughput_rps[PerformanceLevel.POOR]:
            await self._create_alert(
                "Low Throughput",
                f"Throughput is {latest_metrics.throughput_rps:.2f} RPS",
                "warning"
            )
        
        # Check error rate
        if latest_metrics.error_rate_percent > self.thresholds.error_rate_percent[PerformanceLevel.POOR]:
            await self._create_alert(
                "High Error Rate",
                f"Error rate is {latest_metrics.error_rate_percent:.2f}%",
                "error"
            )
        
        # Check CPU usage
        if latest_metrics.cpu_usage_percent > self.thresholds.cpu_usage_percent[PerformanceLevel.POOR]:
            await self._create_alert(
                "High CPU Usage",
                f"CPU usage is {latest_metrics.cpu_usage_percent:.2f}%",
                "warning"
            )
        
        # Check memory usage
        if latest_metrics.memory_usage_percent > self.thresholds.memory_usage_percent[PerformanceLevel.POOR]:
            await self._create_alert(
                "High Memory Usage",
                f"Memory usage is {latest_metrics.memory_usage_percent:.2f}%",
                "warning"
            )
    
    async def _create_alert(self, title: str, message: str, level: str):
        """Create a performance alert."""
        alert = {
            "id": f"alert_{int(time.time())}",
            "title": title,
            "message": message,
            "level": level,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "acknowledged": False
        }
        
        self.performance_alerts.append(alert)
        logger.warning(f"Performance Alert: {title} - {message}")
    
    async def start_request(self, request: Request) -> str:
        """Start monitoring a request."""
        request_id = f"req_{int(time.time() * 1000)}_{id(request)}"
        
        metrics = RequestMetrics(
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            status_code=0,
            start_time=datetime.now(timezone.utc),
            request_size_bytes=len(request.body) if request.body else 0,
            user_agent=request.headers.get("user-agent"),
            ip_address=request.client.host if request.client else None
        )
        
        with self._request_metrics_lock:
            self.request_metrics[request_id] = metrics
        
        return request_id
    
    def end_request(self, request_id: str, response: Response, duration_ms: float):
        """End monitoring a request."""
        with self._request_metrics_lock:
            if request_id in self.request_metrics:
                metrics = self.request_metrics[request_id]
                metrics.end_time = datetime.now(timezone.utc)
                metrics.duration_ms = duration_ms
                metrics.status_code = response.status_code
                metrics.response_size_bytes = len(response.body) if response.body else 0
                
                # Update Prometheus metrics
                self.request_duration.observe(
                    duration_ms / 1000.0,
                    method=metrics.method,
                    endpoint=metrics.path,
                    status_code=str(metrics.status_code)
                )
                
                self.request_total.labels(
                    method=metrics.method,
                    endpoint=metrics.path,
                    status_code=str(metrics.status_code)
                ).inc()
                
                self.request_size.observe(
                    metrics.request_size_bytes,
                    method=metrics.method,
                    endpoint=metrics.path
                )
                
                self.response_size.observe(
                    metrics.response_size_bytes,
                    method=metrics.method,
                    endpoint=metrics.path,
                    status_code=str(metrics.status_code)
                )
                
                # Update endpoint metrics
                self._update_endpoint_metrics(metrics)
                
                # Add to rolling window
                self._request_window.append(metrics)
                
                # Clean up old metrics
                if len(self.request_metrics) > 10000:
                    oldest_key = min(self.request_metrics.keys(), key=lambda k: self.request_metrics[k].start_time)
                    del self.request_metrics[oldest_key]
    
    def _update_endpoint_metrics(self, request_metrics: RequestMetrics):
        """Update endpoint-level metrics."""
        endpoint_key = f"{request_metrics.method}:{request_metrics.path}"
        
        with self._endpoint_metrics_lock:
            if endpoint_key not in self.endpoint_metrics:
                self.endpoint_metrics[endpoint_key] = EndpointMetrics(
                    method=request_metrics.method,
                    path=request_metrics.path,
                    first_request_time=request_metrics.start_time
                )
            
            endpoint = self.endpoint_metrics[endpoint_key]
            
            # Update basic metrics
            endpoint.total_requests += 1
            endpoint.last_request_time = request_metrics.start_time
            
            if 200 <= request_metrics.status_code < 400:
                endpoint.successful_requests += 1
            else:
                endpoint.failed_requests += 1
            
            # Update duration metrics
            if request_metrics.duration_ms is not None:
                endpoint.total_duration_ms += request_metrics.duration_ms
                endpoint.min_duration_ms = min(endpoint.min_duration_ms, request_metrics.duration_ms)
                endpoint.max_duration_ms = max(endpoint.max_duration_ms, request_metrics.duration_ms)
                endpoint.avg_duration_ms = endpoint.total_duration_ms / endpoint.total_requests
            
            # Update size metrics
            endpoint.total_request_size_bytes += request_metrics.request_size_bytes
            endpoint.total_response_size_bytes += request_metrics.response_size_bytes
            endpoint.avg_request_size_bytes = endpoint.total_request_size_bytes / endpoint.total_requests
            endpoint.avg_response_size_bytes = endpoint.total_response_size_bytes / endpoint.total_requests
            
            # Update operation metrics
            endpoint.total_database_queries += request_metrics.database_queries
            endpoint.total_cache_hits += request_metrics.cache_hits
            endpoint.total_cache_misses += request_metrics.cache_misses
            endpoint.total_external_api_calls += request_metrics.external_api_calls
    
    def record_database_query(self, operation: str, table: str, duration_ms: float):
        """Record a database query."""
        self.database_queries.labels(operation=operation, table=table).inc()
        self.database_duration.observe(duration_ms / 1000.0, operation=operation, table=table)
    
    def record_cache_operation(self, cache_name: str, hit: bool):
        """Record a cache operation."""
        if hit:
            self.cache_hits.labels(cache_name=cache_name).inc()
        else:
            self.cache_misses.labels(cache_name=cache_name).inc()
    
    def record_external_api_call(self, api_name: str, endpoint: str, duration_ms: float):
        """Record an external API call."""
        self.external_api_calls.labels(api_name=api_name, endpoint=endpoint).inc()
        self.external_api_duration.observe(duration_ms / 1000.0, api_name=api_name, endpoint=endpoint)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        if not self.system_metrics:
            return {}
        
        latest_metrics = self.system_metrics[-1]
        
        # Calculate percentiles for response time
        recent_requests = list(self._request_window)[-1000:]  # Last 1000 requests
        durations = [req.duration_ms for req in recent_requests if req.duration_ms is not None]
        
        if durations:
            durations.sort()
            p50 = durations[len(durations) // 2]
            p95 = durations[int(len(durations) * 0.95)]
            p99 = durations[int(len(durations) * 0.99)]
        else:
            p50 = p95 = p99 = 0.0
        
        return {
            "system": {
                "cpu_usage_percent": latest_metrics.cpu_usage_percent,
                "memory_usage_percent": latest_metrics.memory_usage_percent,
                "memory_available_mb": latest_metrics.memory_available_mb,
                "disk_usage_percent": latest_metrics.disk_usage_percent,
                "active_connections": latest_metrics.active_connections,
                "throughput_rps": latest_metrics.throughput_rps,
                "avg_response_time_ms": latest_metrics.avg_response_time_ms,
                "error_rate_percent": latest_metrics.error_rate_percent
            },
            "response_time": {
                "avg_ms": latest_metrics.avg_response_time_ms,
                "min_ms": min(durations) if durations else 0.0,
                "max_ms": max(durations) if durations else 0.0,
                "p50_ms": p50,
                "p95_ms": p95,
                "p99_ms": p99
            },
            "endpoints": {
                endpoint: {
                    "total_requests": metrics.total_requests,
                    "successful_requests": metrics.successful_requests,
                    "failed_requests": metrics.failed_requests,
                    "avg_duration_ms": metrics.avg_duration_ms,
                    "avg_request_size_bytes": metrics.avg_request_size_bytes,
                    "avg_response_size_bytes": metrics.avg_response_size_bytes,
                    "total_database_queries": metrics.total_database_queries,
                    "total_cache_hits": metrics.total_cache_hits,
                    "total_cache_misses": metrics.total_cache_misses,
                    "total_external_api_calls": metrics.total_external_api_calls
                }
                for endpoint, metrics in self.endpoint_metrics.items()
            },
            "alerts": [
                {
                    "title": alert["title"],
                    "message": alert["message"],
                    "level": alert["level"],
                    "timestamp": alert["timestamp"]
                }
                for alert in self.performance_alerts[-10:]  # Last 10 alerts
            ]
        }
    
    def get_prometheus_metrics(self) -> str:
        """Get Prometheus metrics."""
        return generate_latest()
    
    def get_endpoint_performance(self, method: str, path: str) -> Optional[EndpointMetrics]:
        """Get performance metrics for a specific endpoint."""
        endpoint_key = f"{method}:{path}"
        return self.endpoint_metrics.get(endpoint_key)
    
    def get_system_metrics_history(self, minutes: int = 60) -> List[SystemMetrics]:
        """Get system metrics history."""
        cutoff_time = datetime.now(timezone.utc) - timedelta(minutes=minutes)
        
        with self._system_metrics_lock:
            return [
                metrics for metrics in self.system_metrics
                if metrics.timestamp > cutoff_time
            ]

# =============================================================================
# Performance Middleware
# =============================================================================

class PerformanceMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware for performance monitoring."""
    
    def __init__(self, app, monitor: PerformanceMonitor):
        
    """__init__ function."""
super().__init__(app)
        self.monitor = monitor
    
    async def dispatch(self, request: Request, call_next):
        """Process request with performance monitoring."""
        start_time = time.time()
        
        # Start monitoring
        request_id = self.monitor.start_request(request)
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate duration
            duration_ms = (time.time() - start_time) * 1000
            
            # End monitoring
            self.monitor.end_request(request_id, response, duration_ms)
            
            return response
            
        except Exception as e:
            # Handle errors
            duration_ms = (time.time() - start_time) * 1000
            
            error_response = JSONResponse(
                status_code=500,
                content={"error": "Internal server error"}
            )
            
            self.monitor.end_request(request_id, error_response, duration_ms)
            raise

# =============================================================================
# Performance Decorators
# =============================================================================

def monitor_performance(operation_name: str = None):
    """Decorator for monitoring function performance."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                duration_ms = (time.time() - start_time) * 1000
                
                # Record performance (you would inject the monitor)
                # monitor.record_operation(operation_name or func.__name__, duration_ms)
                
                return result
                
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                # monitor.record_error(operation_name or func.__name__, duration_ms, str(e))
                raise
        
        return wrapper
    return decorator

def monitor_database_query(table: str = None):
    """Decorator for monitoring database queries."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                duration_ms = (time.time() - start_time) * 1000
                
                # Record database query (you would inject the monitor)
                # monitor.record_database_query(func.__name__, table or "unknown", duration_ms)
                
                return result
                
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                # monitor.record_database_error(func.__name__, table or "unknown", duration_ms, str(e))
                raise
        
        return wrapper
    return decorator

# =============================================================================
# Export
# =============================================================================

__all__ = [
    "MetricType",
    "PerformanceLevel",
    "PerformanceThresholds",
    "RequestMetrics",
    "EndpointMetrics",
    "SystemMetrics",
    "PerformanceMonitor",
    "PerformanceMiddleware",
    "monitor_performance",
    "monitor_database_query",
] 