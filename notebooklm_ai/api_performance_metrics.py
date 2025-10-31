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
import statistics
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import deque, defaultdict
from datetime import datetime, timedelta
import json
import logging
from contextlib import asynccontextmanager
from functools import wraps
from prometheus_client import (
import psutil
import aiohttp
from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.responses import JSONResponse
import redis.asyncio as redis
    import uvicorn
from typing import Any, List, Dict, Optional
"""
API Performance Metrics System for notebooklm_ai
- Response time monitoring
- Latency tracking
- Throughput measurement
- Real-time performance analytics
- Performance alerts and thresholds
"""


# Performance monitoring libraries
    Counter, Histogram, Gauge, Summary, 
    generate_latest, CONTENT_TYPE_LATEST
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# PERFORMANCE METRICS DATA STRUCTURES
# ============================================================================

@dataclass
class RequestMetrics:
    """Individual request performance metrics."""
    request_id: str
    endpoint: str
    method: str
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    status_code: Optional[int] = None
    response_size: Optional[int] = None
    cache_hit: bool = False
    error: Optional[str] = None
    
    def __post_init__(self) -> Any:
        if self.end_time and self.start_time:
            self.duration = self.end_time - self.start_time

@dataclass
class EndpointMetrics:
    """Aggregated metrics for a specific endpoint."""
    endpoint: str
    method: str
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_duration: float = 0.0
    min_duration: float = float('inf')
    max_duration: float = 0.0
    avg_duration: float = 0.0
    p50_duration: float = 0.0
    p95_duration: float = 0.0
    p99_duration: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    total_response_size: int = 0
    avg_response_size: float = 0.0
    error_count: int = 0
    last_updated: datetime = field(default_factory=datetime.utcnow)
    
    # Rolling window for recent metrics
    recent_durations: deque = field(default_factory=lambda: deque(maxlen=1000))
    recent_response_sizes: deque = field(default_factory=lambda: deque(maxlen=1000))

@dataclass
class SystemMetrics:
    """System-level performance metrics."""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    network_io: Dict[str, float]
    active_connections: int
    total_requests_per_second: float
    avg_response_time: float
    error_rate: float

# ============================================================================
# PERFORMANCE MONITORING CLASSES
# ============================================================================

class APIPerformanceMonitor:
    """Comprehensive API performance monitoring system."""
    
    def __init__(self, window_size: int = 1000, alert_thresholds: Optional[Dict] = None):
        
    """__init__ function."""
self.window_size = window_size
        self.alert_thresholds = alert_thresholds or {
            'response_time_p95': 2.0,  # 2 seconds
            'error_rate': 0.05,        # 5%
            'cpu_usage': 80.0,         # 80%
            'memory_usage': 85.0,      # 85%
        }
        
        # Metrics storage
        self.endpoint_metrics: Dict[str, EndpointMetrics] = defaultdict(
            lambda: EndpointMetrics("", "")
        )
        self.system_metrics: List[SystemMetrics] = []
        self.active_requests: Dict[str, RequestMetrics] = {}
        
        # Prometheus metrics
        self._setup_prometheus_metrics()
        
        # Performance alerts
        self.alerts: List[Dict] = []
        
        # Throughput tracking
        self.request_timestamps: deque = deque(maxlen=10000)
        self.throughput_window = 60  # 60 seconds
        
    def _setup_prometheus_metrics(self) -> Any:
        """Setup Prometheus metrics."""
        # Request metrics
        self.request_counter = Counter(
            'api_requests_total',
            'Total API requests',
            ['method', 'endpoint', 'status_code']
        )
        
        self.request_duration = Histogram(
            'api_request_duration_seconds',
            'API request duration',
            ['method', 'endpoint'],
            buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0, 100.0]
        )
        
        self.response_size = Histogram(
            'api_response_size_bytes',
            'API response size',
            ['method', 'endpoint'],
            buckets=[100, 1000, 10000, 100000, 1000000, 10000000]
        )
        
        # Throughput metrics
        self.requests_per_second = Gauge(
            'api_requests_per_second',
            'Requests per second'
        )
        
        self.active_requests_gauge = Gauge(
            'api_active_requests',
            'Number of active requests'
        )
        
        # Cache metrics
        self.cache_hits = Counter(
            'api_cache_hits_total',
            'Total cache hits'
        )
        
        self.cache_misses = Counter(
            'api_cache_misses_total',
            'Total cache misses'
        )
        
        # Error metrics
        self.error_counter = Counter(
            'api_errors_total',
            'Total API errors',
            ['method', 'endpoint', 'error_type']
        )
        
        # System metrics
        self.cpu_usage = Gauge(
            'system_cpu_usage_percent',
            'CPU usage percentage'
        )
        
        self.memory_usage = Gauge(
            'system_memory_usage_percent',
            'Memory usage percentage'
        )
        
        self.disk_usage = Gauge(
            'system_disk_usage_percent',
            'Disk usage percentage'
        )

    async def start_request(self, request_id: str, endpoint: str, method: str) -> RequestMetrics:
        """Start tracking a new request."""
        metrics = RequestMetrics(
            request_id=request_id,
            endpoint=endpoint,
            method=method,
            start_time=time.time()
        )
        
        self.active_requests[request_id] = metrics
        self.active_requests_gauge.inc()
        self.request_timestamps.append(time.time())
        
        return metrics

    def end_request(self, request_id: str, status_code: int, response_size: int = 0, 
                   cache_hit: bool = False, error: Optional[str] = None):
        """End tracking a request and update metrics."""
        if request_id not in self.active_requests:
            return
        
        metrics = self.active_requests[request_id]
        metrics.end_time = time.time()
        metrics.status_code = status_code
        metrics.response_size = response_size
        metrics.cache_hit = cache_hit
        metrics.error = error
        
        # Update Prometheus metrics
        self.request_counter.labels(
            method=metrics.method,
            endpoint=metrics.endpoint,
            status_code=status_code
        ).inc()
        
        if metrics.duration:
            self.request_duration.labels(
                method=metrics.method,
                endpoint=metrics.endpoint
            ).observe(metrics.duration)
            
            self.response_size.labels(
                method=metrics.method,
                endpoint=metrics.endpoint
            ).observe(response_size)
        
        # Update cache metrics
        if cache_hit:
            self.cache_hits.inc()
        else:
            self.cache_misses.inc()
        
        # Update error metrics
        if error or status_code >= 400:
            self.error_counter.labels(
                method=metrics.method,
                endpoint=metrics.endpoint,
                error_type=error or f"http_{status_code}"
            ).inc()
        
        # Update endpoint metrics
        self._update_endpoint_metrics(metrics)
        
        # Remove from active requests
        del self.active_requests[request_id]
        self.active_requests_gauge.dec()

    def _update_endpoint_metrics(self, metrics: RequestMetrics):
        """Update aggregated endpoint metrics."""
        key = f"{metrics.method}:{metrics.endpoint}"
        endpoint_metrics = self.endpoint_metrics[key]
        
        if endpoint_metrics.endpoint == "":
            endpoint_metrics.endpoint = metrics.endpoint
            endpoint_metrics.method = metrics.method
        
        endpoint_metrics.total_requests += 1
        endpoint_metrics.total_duration += metrics.duration or 0
        endpoint_metrics.total_response_size += metrics.response_size or 0
        
        if metrics.duration:
            endpoint_metrics.min_duration = min(endpoint_metrics.min_duration, metrics.duration)
            endpoint_metrics.max_duration = max(endpoint_metrics.max_duration, metrics.duration)
            endpoint_metrics.recent_durations.append(metrics.duration)
        
        if metrics.response_size:
            endpoint_metrics.recent_response_sizes.append(metrics.response_size)
        
        if metrics.cache_hit:
            endpoint_metrics.cache_hits += 1
        else:
            endpoint_metrics.cache_misses += 1
        
        if metrics.error or (metrics.status_code and metrics.status_code >= 400):
            endpoint_metrics.failed_requests += 1
            endpoint_metrics.error_count += 1
        else:
            endpoint_metrics.successful_requests += 1
        
        # Calculate percentiles
        if endpoint_metrics.recent_durations:
            durations = list(endpoint_metrics.recent_durations)
            endpoint_metrics.avg_duration = statistics.mean(durations)
            endpoint_metrics.p50_duration = statistics.quantiles(durations, n=2)[0]
            endpoint_metrics.p95_duration = statistics.quantiles(durations, n=20)[18]
            endpoint_metrics.p99_duration = statistics.quantiles(durations, n=100)[98]
        
        if endpoint_metrics.recent_response_sizes:
            response_sizes = list(endpoint_metrics.recent_response_sizes)
            endpoint_metrics.avg_response_size = statistics.mean(response_sizes)
        
        endpoint_metrics.last_updated = datetime.utcnow()

    def get_throughput(self) -> float:
        """Calculate current requests per second."""
        now = time.time()
        cutoff = now - self.throughput_window
        
        # Remove old timestamps
        while self.request_timestamps and self.request_timestamps[0] < cutoff:
            self.request_timestamps.popleft()
        
        if not self.request_timestamps:
            return 0.0
        
        return len(self.request_timestamps) / self.throughput_window

    def get_system_metrics(self) -> SystemMetrics:
        """Get current system metrics."""
        # CPU and memory
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Network I/O
        network_io = psutil.net_io_counters()
        network_metrics = {
            'bytes_sent': network_io.bytes_sent,
            'bytes_recv': network_io.bytes_recv,
            'packets_sent': network_io.packets_sent,
            'packets_recv': network_io.packets_recv
        }
        
        # Calculate overall metrics
        total_requests_per_second = self.get_throughput()
        
        # Calculate average response time across all endpoints
        total_duration = 0
        total_requests = 0
        total_errors = 0
        
        for metrics in self.endpoint_metrics.values():
            total_duration += metrics.total_duration
            total_requests += metrics.total_requests
            total_errors += metrics.error_count
        
        avg_response_time = total_duration / total_requests if total_requests > 0 else 0
        error_rate = total_errors / total_requests if total_requests > 0 else 0
        
        system_metrics = SystemMetrics(
            timestamp=datetime.utcnow(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            disk_percent=disk.percent,
            network_io=network_metrics,
            active_connections=len(self.active_requests),
            total_requests_per_second=total_requests_per_second,
            avg_response_time=avg_response_time,
            error_rate=error_rate
        )
        
        # Update Prometheus system metrics
        self.cpu_usage.set(cpu_percent)
        self.memory_usage.set(memory.percent)
        self.disk_usage.set(disk.percent)
        self.requests_per_second.set(total_requests_per_second)
        
        # Store for historical tracking
        self.system_metrics.append(system_metrics)
        
        # Keep only last 1000 system metrics
        if len(self.system_metrics) > 1000:
            self.system_metrics = self.system_metrics[-1000:]
        
        return system_metrics

    def check_alerts(self) -> List[Dict]:
        """Check for performance alerts."""
        alerts = []
        system_metrics = self.get_system_metrics()
        
        # Check response time P95
        for endpoint_metrics in self.endpoint_metrics.values():
            if endpoint_metrics.p95_duration > self.alert_thresholds['response_time_p95']:
                alerts.append({
                    'type': 'high_response_time',
                    'endpoint': endpoint_metrics.endpoint,
                    'method': endpoint_metrics.method,
                    'value': endpoint_metrics.p95_duration,
                    'threshold': self.alert_thresholds['response_time_p95'],
                    'timestamp': datetime.utcnow()
                })
        
        # Check error rate
        if system_metrics.error_rate > self.alert_thresholds['error_rate']:
            alerts.append({
                'type': 'high_error_rate',
                'value': system_metrics.error_rate,
                'threshold': self.alert_thresholds['error_rate'],
                'timestamp': datetime.utcnow()
            })
        
        # Check CPU usage
        if system_metrics.cpu_percent > self.alert_thresholds['cpu_usage']:
            alerts.append({
                'type': 'high_cpu_usage',
                'value': system_metrics.cpu_percent,
                'threshold': self.alert_thresholds['cpu_usage'],
                'timestamp': datetime.utcnow()
            })
        
        # Check memory usage
        if system_metrics.memory_percent > self.alert_thresholds['memory_usage']:
            alerts.append({
                'type': 'high_memory_usage',
                'value': system_metrics.memory_percent,
                'threshold': self.alert_thresholds['memory_usage'],
                'timestamp': datetime.utcnow()
            })
        
        self.alerts.extend(alerts)
        return alerts

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        system_metrics = self.get_system_metrics()
        
        # Top endpoints by request count
        top_endpoints = sorted(
            self.endpoint_metrics.values(),
            key=lambda x: x.total_requests,
            reverse=True
        )[:10]
        
        # Slowest endpoints
        slowest_endpoints = sorted(
            self.endpoint_metrics.values(),
            key=lambda x: x.avg_duration,
            reverse=True
        )[:10]
        
        # Endpoints with highest error rates
        error_endpoints = sorted(
            self.endpoint_metrics.values(),
            key=lambda x: x.error_count / max(x.total_requests, 1),
            reverse=True
        )[:10]
        
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'system_metrics': {
                'cpu_percent': system_metrics.cpu_percent,
                'memory_percent': system_metrics.memory_percent,
                'disk_percent': system_metrics.disk_percent,
                'active_connections': system_metrics.active_connections,
                'requests_per_second': system_metrics.total_requests_per_second,
                'avg_response_time': system_metrics.avg_response_time,
                'error_rate': system_metrics.error_rate
            },
            'top_endpoints': [
                {
                    'endpoint': m.endpoint,
                    'method': m.method,
                    'total_requests': m.total_requests,
                    'avg_duration': m.avg_duration,
                    'p95_duration': m.p95_duration,
                    'error_rate': m.error_count / max(m.total_requests, 1)
                }
                for m in top_endpoints
            ],
            'slowest_endpoints': [
                {
                    'endpoint': m.endpoint,
                    'method': m.method,
                    'avg_duration': m.avg_duration,
                    'p95_duration': m.p95_duration,
                    'total_requests': m.total_requests
                }
                for m in slowest_endpoints
            ],
            'error_endpoints': [
                {
                    'endpoint': m.endpoint,
                    'method': m.method,
                    'error_rate': m.error_count / max(m.total_requests, 1),
                    'total_requests': m.total_requests,
                    'error_count': m.error_count
                }
                for m in error_endpoints
            ],
            'recent_alerts': self.alerts[-10:],
            'total_endpoints': len(self.endpoint_metrics),
            'total_requests': sum(m.total_requests for m in self.endpoint_metrics.values())
        }

# ============================================================================
# PERFORMANCE MIDDLEWARE
# ============================================================================

class PerformanceMiddleware:
    """FastAPI middleware for performance monitoring."""
    
    def __init__(self, monitor: APIPerformanceMonitor):
        
    """__init__ function."""
self.monitor = monitor
    
    async def __call__(self, request: Request, call_next):
        """Middleware implementation."""
        request_id = f"{request.method}_{request.url.path}_{int(time.time() * 1000000)}"
        
        # Start request tracking
        metrics = self.monitor.start_request(
            request_id=request_id,
            endpoint=request.url.path,
            method=request.method
        )
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate response size
            response_size = 0
            if hasattr(response, 'body'):
                response_size = len(response.body)
            
            # End request tracking
            self.monitor.end_request(
                request_id=request_id,
                status_code=response.status_code,
                response_size=response_size,
                cache_hit=False  # Will be updated by cache middleware
            )
            
            # Add performance headers
            if metrics.duration:
                response.headers["X-Processing-Time"] = f"{metrics.duration:.4f}"
                response.headers["X-Request-ID"] = request_id
            
            return response
            
        except Exception as e:
            # Track error
            self.monitor.end_request(
                request_id=request_id,
                status_code=500,
                response_size=0,
                cache_hit=False,
                error=str(e)
            )
            raise

# ============================================================================
# PERFORMANCE DECORATORS
# ============================================================================

def monitor_performance(monitor: APIPerformanceMonitor):
    """Decorator for monitoring specific functions."""
    def decorator(func) -> Any:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            request_id = f"{func.__name__}_{int(time.time() * 1000000)}"
            
            # Start tracking
            metrics = monitor.start_request(
                request_id=request_id,
                endpoint=func.__name__,
                method="FUNCTION"
            )
            
            try:
                result = await func(*args, **kwargs)
                
                # End tracking
                monitor.end_request(
                    request_id=request_id,
                    status_code=200,
                    response_size=0
                )
                
                return result
                
            except Exception as e:
                monitor.end_request(
                    request_id=request_id,
                    status_code=500,
                    response_size=0,
                    error=str(e)
                )
                raise
        
        return wrapper
    return decorator

# ============================================================================
# PERFORMANCE ANALYTICS
# ============================================================================

class PerformanceAnalytics:
    """Advanced performance analytics and insights."""
    
    def __init__(self, monitor: APIPerformanceMonitor):
        
    """__init__ function."""
self.monitor = monitor
    
    def get_trend_analysis(self, hours: int = 24) -> Dict[str, Any]:
        """Analyze performance trends over time."""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        
        # Filter recent system metrics
        recent_metrics = [
            m for m in self.monitor.system_metrics
            if m.timestamp > cutoff
        ]
        
        if not recent_metrics:
            return {"error": "No data available"}
        
        # Calculate trends
        response_times = [m.avg_response_time for m in recent_metrics]
        throughput = [m.total_requests_per_second for m in recent_metrics]
        error_rates = [m.error_rate for m in recent_metrics]
        cpu_usage = [m.cpu_percent for m in recent_metrics]
        
        return {
            'time_period': f"{hours} hours",
            'data_points': len(recent_metrics),
            'response_time_trend': {
                'current': response_times[-1] if response_times else 0,
                'average': statistics.mean(response_times),
                'trend': 'increasing' if len(response_times) > 1 and response_times[-1] > response_times[0] else 'stable'
            },
            'throughput_trend': {
                'current': throughput[-1] if throughput else 0,
                'average': statistics.mean(throughput),
                'trend': 'increasing' if len(throughput) > 1 and throughput[-1] > throughput[0] else 'stable'
            },
            'error_rate_trend': {
                'current': error_rates[-1] if error_rates else 0,
                'average': statistics.mean(error_rates),
                'trend': 'increasing' if len(error_rates) > 1 and error_rates[-1] > error_rates[0] else 'stable'
            },
            'cpu_usage_trend': {
                'current': cpu_usage[-1] if cpu_usage else 0,
                'average': statistics.mean(cpu_usage),
                'trend': 'increasing' if len(cpu_usage) > 1 and cpu_usage[-1] > cpu_usage[0] else 'stable'
            }
        }
    
    def get_bottleneck_analysis(self) -> Dict[str, Any]:
        """Identify performance bottlenecks."""
        bottlenecks = []
        
        for endpoint_metrics in self.monitor.endpoint_metrics.values():
            if endpoint_metrics.avg_duration > 1.0:  # Slow endpoints
                bottlenecks.append({
                    'type': 'slow_endpoint',
                    'endpoint': endpoint_metrics.endpoint,
                    'method': endpoint_metrics.method,
                    'avg_duration': endpoint_metrics.avg_duration,
                    'p95_duration': endpoint_metrics.p95_duration,
                    'total_requests': endpoint_metrics.total_requests
                })
            
            error_rate = endpoint_metrics.error_count / max(endpoint_metrics.total_requests, 1)
            if error_rate > 0.1:  # High error rate
                bottlenecks.append({
                    'type': 'high_error_rate',
                    'endpoint': endpoint_metrics.endpoint,
                    'method': endpoint_metrics.method,
                    'error_rate': error_rate,
                    'total_requests': endpoint_metrics.total_requests,
                    'error_count': endpoint_metrics.error_count
                })
        
        # Sort by severity
        bottlenecks.sort(key=lambda x: x.get('avg_duration', 0) or x.get('error_rate', 0), reverse=True)
        
        return {
            'bottlenecks': bottlenecks[:10],
            'total_bottlenecks': len(bottlenecks),
            'recommendations': self._generate_recommendations(bottlenecks)
        }
    
    def _generate_recommendations(self, bottlenecks: List[Dict]) -> List[str]:
        """Generate performance improvement recommendations."""
        recommendations = []
        
        slow_endpoints = [b for b in bottlenecks if b['type'] == 'slow_endpoint']
        error_endpoints = [b for b in bottlenecks if b['type'] == 'high_error_rate']
        
        if slow_endpoints:
            recommendations.append(
                f"Consider caching for {len(slow_endpoints)} slow endpoints"
            )
            recommendations.append(
                "Implement database query optimization for slow endpoints"
            )
        
        if error_endpoints:
            recommendations.append(
                f"Investigate error handling for {len(error_endpoints)} high-error endpoints"
            )
        
        if len(bottlenecks) > 5:
            recommendations.append(
                "Consider implementing rate limiting and load balancing"
            )
        
        return recommendations

# ============================================================================
# FASTAPI INTEGRATION
# ============================================================================

def create_performance_app() -> FastAPI:
    """Create FastAPI app with performance monitoring."""
    app = FastAPI(title="Performance Monitoring API")
    
    # Initialize performance monitor
    monitor = APIPerformanceMonitor()
    analytics = PerformanceAnalytics(monitor)
    
    # Add performance middleware
    app.add_middleware(PerformanceMiddleware, monitor=monitor)
    
    @app.get("/metrics")
    async def prometheus_metrics():
        """Prometheus metrics endpoint."""
        return Response(
            generate_latest(),
            media_type=CONTENT_TYPE_LATEST
        )
    
    @app.get("/performance/summary")
    async def performance_summary():
        """Get performance summary."""
        return monitor.get_performance_summary()
    
    @app.get("/performance/trends")
    async def performance_trends(hours: int = 24):
        """Get performance trends."""
        return analytics.get_trend_analysis(hours)
    
    @app.get("/performance/bottlenecks")
    async def performance_bottlenecks():
        """Get bottleneck analysis."""
        return analytics.get_bottleneck_analysis()
    
    @app.get("/performance/alerts")
    async def performance_alerts():
        """Get current alerts."""
        alerts = monitor.check_alerts()
        return {
            'alerts': alerts,
            'total_alerts': len(alerts),
            'timestamp': datetime.utcnow().isoformat()
        }
    
    @app.get("/performance/endpoints/{endpoint}")
    async def endpoint_metrics(endpoint: str):
        """Get metrics for specific endpoint."""
        endpoint_data = {}
        for key, metrics in monitor.endpoint_metrics.items():
            if endpoint in key:
                endpoint_data[key] = {
                    'total_requests': metrics.total_requests,
                    'avg_duration': metrics.avg_duration,
                    'p95_duration': metrics.p95_duration,
                    'p99_duration': metrics.p99_duration,
                    'error_rate': metrics.error_count / max(metrics.total_requests, 1),
                    'cache_hit_rate': metrics.cache_hits / max(metrics.total_requests, 1),
                    'avg_response_size': metrics.avg_response_size,
                    'last_updated': metrics.last_updated.isoformat()
                }
        
        return endpoint_data
    
    return app

# ============================================================================
# BACKGROUND MONITORING TASKS
# ============================================================================

async def background_performance_monitoring(monitor: APIPerformanceMonitor):
    """Background task for continuous performance monitoring."""
    while True:
        try:
            # Update system metrics
            monitor.get_system_metrics()
            
            # Check for alerts
            alerts = monitor.check_alerts()
            if alerts:
                logger.warning(f"Performance alerts detected: {len(alerts)} alerts")
                for alert in alerts:
                    logger.warning(f"Alert: {alert['type']} - {alert.get('value', 'N/A')}")
            
            # Log performance summary every 5 minutes
            if int(time.time()) % 300 == 0:
                summary = monitor.get_performance_summary()
                logger.info(f"Performance Summary: {summary['system_metrics']['requests_per_second']:.2f} req/s, "
                          f"Avg RT: {summary['system_metrics']['avg_response_time']:.3f}s, "
                          f"Error Rate: {summary['system_metrics']['error_rate']:.2%}")
            
            await asyncio.sleep(10)  # Check every 10 seconds
            
        except Exception as e:
            logger.error(f"Performance monitoring error: {e}")
            await asyncio.sleep(30)

# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    
    # Create performance monitoring app
    app = create_performance_app()
    
    # Start background monitoring
    monitor = APIPerformanceMonitor()
    asyncio.create_task(background_performance_monitoring(monitor))
    
    # Run the application
    uvicorn.run(app, host="0.0.0.0", port=8000) 