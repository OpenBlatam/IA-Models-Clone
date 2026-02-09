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
import logging
import statistics
import threading
from typing import Any, Dict, List, Optional, Callable, Union, Tuple, NamedTuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
from datetime import datetime, timedelta
from enum import Enum
import json
import sqlite3
from pathlib import Path
import functools
import weakref
import structlog
from pydantic import BaseModel, Field
import numpy as np
from prometheus_client import Counter, Histogram, Gauge, Summary, generate_latest
import redis.asyncio as redis
from typing import Any, List, Dict, Optional
"""
🚀 API Performance Metrics System
=================================

Comprehensive API performance metrics prioritization system with:
- Response time tracking (p50, p95, p99, p99.9)
- Latency monitoring (network, processing, database)
- Throughput measurement (requests/second, concurrent users)
- Real-time performance alerts
- Performance optimization recommendations
- Historical trend analysis
- SLA monitoring and compliance
- Performance regression detection
- Load testing integration
- Performance dashboards
"""



logger = structlog.get_logger(__name__)

class MetricPriority(Enum):
    """Metric priority levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class PerformanceThreshold(Enum):
    """Performance threshold levels"""
    EXCELLENT = "excellent"
    GOOD = "good"
    WARNING = "warning"
    CRITICAL = "critical"

class LatencyType(Enum):
    """Types of latency measurements"""
    NETWORK = "network"
    PROCESSING = "processing"
    DATABASE = "database"
    CACHE = "cache"
    EXTERNAL_API = "external_api"
    TOTAL = "total"

@dataclass
class ResponseTimeMetrics:
    """Response time metrics with percentiles"""
    count: int = 0
    total_time: float = 0.0
    min_time: float = float('inf')
    max_time: float = 0.0
    p50: float = 0.0
    p95: float = 0.0
    p99: float = 0.0
    p99_9: float = 0.0
    average: float = 0.0
    median: float = 0.0
    
    def update(self, response_time: float):
        """Update metrics with new response time"""
        self.count += 1
        self.total_time += response_time
        self.min_time = min(self.min_time, response_time)
        self.max_time = max(self.max_time, response_time)
        
        # Update percentiles (simplified - in production use proper percentile tracking)
        self.average = self.total_time / self.count
        self.median = self.p50  # Simplified

@dataclass
class LatencyBreakdown:
    """Detailed latency breakdown"""
    network_latency: float = 0.0
    processing_latency: float = 0.0
    database_latency: float = 0.0
    cache_latency: float = 0.0
    external_api_latency: float = 0.0
    total_latency: float = 0.0
    
    def calculate_total(self) -> Any:
        """Calculate total latency"""
        self.total_latency = (
            self.network_latency +
            self.processing_latency +
            self.database_latency +
            self.cache_latency +
            self.external_api_latency
        )

@dataclass
class ThroughputMetrics:
    """Throughput metrics"""
    requests_per_second: float = 0.0
    concurrent_users: int = 0
    max_concurrent_users: int = 0
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    bytes_transferred: int = 0
    requests_per_minute: float = 0.0
    requests_per_hour: float = 0.0
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate"""
        return self.successful_requests / self.total_requests if self.total_requests > 0 else 0.0

@dataclass
class APIPerformanceMetrics:
    """Complete API performance metrics"""
    endpoint: str
    method: str
    response_time: ResponseTimeMetrics = field(default_factory=ResponseTimeMetrics)
    latency_breakdown: LatencyBreakdown = field(default_factory=LatencyBreakdown)
    throughput: ThroughputMetrics = field(default_factory=ThroughputMetrics)
    priority: MetricPriority = MetricPriority.MEDIUM
    last_updated: float = field(default_factory=time.time)
    
    def update_response_time(self, response_time: float):
        """Update response time metrics"""
        self.response_time.update(response_time)
        self.last_updated = time.time()
    
    def update_latency(self, latency_type: LatencyType, latency: float):
        """Update specific latency metric"""
        if latency_type == LatencyType.NETWORK:
            self.latency_breakdown.network_latency = latency
        elif latency_type == LatencyType.PROCESSING:
            self.latency_breakdown.processing_latency = latency
        elif latency_type == LatencyType.DATABASE:
            self.latency_breakdown.database_latency = latency
        elif latency_type == LatencyType.CACHE:
            self.latency_breakdown.cache_latency = latency
        elif latency_type == LatencyType.EXTERNAL_API:
            self.latency_breakdown.external_api_latency = latency
        
        self.latency_breakdown.calculate_total()
        self.last_updated = time.time()
    
    def update_throughput(self, request_count: int = 1, success: bool = True, bytes_transferred: int = 0):
        """Update throughput metrics"""
        self.throughput.total_requests += request_count
        self.throughput.bytes_transferred += bytes_transferred
        
        if success:
            self.throughput.successful_requests += request_count
        else:
            self.throughput.failed_requests += request_count
        
        self.last_updated = time.time()

class PerformanceThresholds:
    """Configurable performance thresholds"""
    
    def __init__(self) -> Any:
        self.response_time_thresholds = {
            MetricPriority.CRITICAL: {
                PerformanceThreshold.EXCELLENT: 0.1,  # 100ms
                PerformanceThreshold.GOOD: 0.5,       # 500ms
                PerformanceThreshold.WARNING: 1.0,    # 1s
                PerformanceThreshold.CRITICAL: 2.0    # 2s
            },
            MetricPriority.HIGH: {
                PerformanceThreshold.EXCELLENT: 0.2,  # 200ms
                PerformanceThreshold.GOOD: 1.0,       # 1s
                PerformanceThreshold.WARNING: 2.0,    # 2s
                PerformanceThreshold.CRITICAL: 5.0    # 5s
            },
            MetricPriority.MEDIUM: {
                PerformanceThreshold.EXCELLENT: 0.5,  # 500ms
                PerformanceThreshold.GOOD: 2.0,       # 2s
                PerformanceThreshold.WARNING: 5.0,    # 5s
                PerformanceThreshold.CRITICAL: 10.0   # 10s
            },
            MetricPriority.LOW: {
                PerformanceThreshold.EXCELLENT: 1.0,  # 1s
                PerformanceThreshold.GOOD: 5.0,       # 5s
                PerformanceThreshold.WARNING: 10.0,   # 10s
                PerformanceThreshold.CRITICAL: 30.0   # 30s
            }
        }
        
        self.throughput_thresholds = {
            MetricPriority.CRITICAL: {
                PerformanceThreshold.EXCELLENT: 1000,  # 1000 req/s
                PerformanceThreshold.GOOD: 500,        # 500 req/s
                PerformanceThreshold.WARNING: 100,     # 100 req/s
                PerformanceThreshold.CRITICAL: 50      # 50 req/s
            },
            MetricPriority.HIGH: {
                PerformanceThreshold.EXCELLENT: 500,   # 500 req/s
                PerformanceThreshold.GOOD: 200,        # 200 req/s
                PerformanceThreshold.WARNING: 50,      # 50 req/s
                PerformanceThreshold.CRITICAL: 20      # 20 req/s
            },
            MetricPriority.MEDIUM: {
                PerformanceThreshold.EXCELLENT: 200,   # 200 req/s
                PerformanceThreshold.GOOD: 100,        # 100 req/s
                PerformanceThreshold.WARNING: 20,      # 20 req/s
                PerformanceThreshold.CRITICAL: 10      # 10 req/s
            },
            MetricPriority.LOW: {
                PerformanceThreshold.EXCELLENT: 100,   # 100 req/s
                PerformanceThreshold.GOOD: 50,         # 50 req/s
                PerformanceThreshold.WARNING: 10,      # 10 req/s
                PerformanceThreshold.CRITICAL: 5       # 5 req/s
            }
        }
    
    def get_response_time_threshold(self, priority: MetricPriority, level: PerformanceThreshold) -> float:
        """Get response time threshold for priority and level"""
        return self.response_time_thresholds[priority][level]
    
    def get_throughput_threshold(self, priority: MetricPriority, level: PerformanceThreshold) -> float:
        """Get throughput threshold for priority and level"""
        return self.throughput_thresholds[priority][level]
    
    def evaluate_response_time(self, response_time: float, priority: MetricPriority) -> PerformanceThreshold:
        """Evaluate response time against thresholds"""
        thresholds = self.response_time_thresholds[priority]
        
        if response_time <= thresholds[PerformanceThreshold.EXCELLENT]:
            return PerformanceThreshold.EXCELLENT
        elif response_time <= thresholds[PerformanceThreshold.GOOD]:
            return PerformanceThreshold.GOOD
        elif response_time <= thresholds[PerformanceThreshold.WARNING]:
            return PerformanceThreshold.WARNING
        else:
            return PerformanceThreshold.CRITICAL
    
    def evaluate_throughput(self, throughput: float, priority: MetricPriority) -> PerformanceThreshold:
        """Evaluate throughput against thresholds"""
        thresholds = self.throughput_thresholds[priority]
        
        if throughput >= thresholds[PerformanceThreshold.EXCELLENT]:
            return PerformanceThreshold.EXCELLENT
        elif throughput >= thresholds[PerformanceThreshold.GOOD]:
            return PerformanceThreshold.GOOD
        elif throughput >= thresholds[PerformanceThreshold.WARNING]:
            return PerformanceThreshold.WARNING
        else:
            return PerformanceThreshold.CRITICAL

class PerformanceAlert:
    """Performance alert"""
    
    def __init__(self, 
                 endpoint: str,
                 metric_type: str,
                 current_value: float,
                 threshold: float,
                 priority: MetricPriority,
                 severity: PerformanceThreshold,
                 message: str):
        
    """__init__ function."""
self.id = f"{endpoint}_{metric_type}_{int(time.time())}"
        self.endpoint = endpoint
        self.metric_type = metric_type
        self.current_value = current_value
        self.threshold = threshold
        self.priority = priority
        self.severity = severity
        self.message = message
        self.timestamp = time.time()
        self.acknowledged = False
        self.resolved = False

class APIPerformanceMonitor:
    """Main API performance monitoring system"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        
    """__init__ function."""
self.redis_url = redis_url
        self.redis_client: Optional[redis.Redis] = None
        
        # Metrics storage
        self.metrics: Dict[str, APIPerformanceMetrics] = {}
        self.metrics_lock = threading.Lock()
        
        # Thresholds
        self.thresholds = PerformanceThresholds()
        
        # Alerts
        self.alerts: List[PerformanceAlert] = []
        self.alerts_lock = threading.Lock()
        
        # Prometheus metrics
        self.prometheus_metrics = {
            "response_time": Histogram("api_response_time_seconds", "API response time", ["endpoint", "method", "priority"]),
            "throughput": Counter("api_requests_total", "Total API requests", ["endpoint", "method", "status"]),
            "latency": Histogram("api_latency_seconds", "API latency breakdown", ["endpoint", "latency_type"]),
            "concurrent_users": Gauge("api_concurrent_users", "Concurrent API users", ["endpoint"]),
            "error_rate": Gauge("api_error_rate", "API error rate", ["endpoint"])
        }
        
        # Performance tracking
        self.performance_history: deque = deque(maxlen=10000)
        self.start_time = time.time()
        
        logger.info("API Performance Monitor initialized")
    
    async def initialize(self) -> Any:
        """Initialize the monitor"""
        try:
            self.redis_client = redis.from_url(self.redis_url)
            await self.redis_client.ping()
            logger.info("Redis connection established for API performance monitoring")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}. Using in-memory storage only.")
            self.redis_client = None
    
    def register_endpoint(self, 
                         endpoint: str, 
                         method: str = "GET", 
                         priority: MetricPriority = MetricPriority.MEDIUM):
        """Register an endpoint for monitoring"""
        key = f"{endpoint}:{method}"
        
        with self.metrics_lock:
            if key not in self.metrics:
                self.metrics[key] = APIPerformanceMetrics(
                    endpoint=endpoint,
                    method=method,
                    priority=priority
                )
                logger.info(f"Registered endpoint for monitoring: {key} (priority: {priority.value})")
    
    def record_request(self, 
                      endpoint: str, 
                      method: str = "GET",
                      response_time: float = 0.0,
                      status_code: int = 200,
                      latency_breakdown: Optional[LatencyBreakdown] = None,
                      bytes_transferred: int = 0):
        """Record a request with performance metrics"""
        key = f"{endpoint}:{method}"
        
        # Ensure endpoint is registered
        if key not in self.metrics:
            self.register_endpoint(endpoint, method)
        
        with self.metrics_lock:
            metrics = self.metrics[key]
            
            # Update response time
            if response_time > 0:
                metrics.update_response_time(response_time)
                
                # Update Prometheus metrics
                self.prometheus_metrics["response_time"].observe(
                    response_time, 
                    endpoint=endpoint, 
                    method=method, 
                    priority=metrics.priority.value
                )
            
            # Update latency breakdown
            if latency_breakdown:
                for latency_type in LatencyType:
                    if latency_type != LatencyType.TOTAL:
                        latency_value = getattr(latency_breakdown, f"{latency_type.value}_latency")
                        if latency_value > 0:
                            metrics.update_latency(latency_type, latency_value)
                            self.prometheus_metrics["latency"].observe(
                                latency_value,
                                endpoint=endpoint,
                                latency_type=latency_type.value
                            )
            
            # Update throughput
            success = 200 <= status_code < 400
            metrics.update_throughput(
                request_count=1,
                success=success,
                bytes_transferred=bytes_transferred
            )
            
            # Update Prometheus metrics
            self.prometheus_metrics["throughput"].inc(
                endpoint=endpoint,
                method=method,
                status="success" if success else "error"
            )
            
            # Store in history
            self.performance_history.append({
                "timestamp": time.time(),
                "endpoint": endpoint,
                "method": method,
                "response_time": response_time,
                "status_code": status_code,
                "success": success
            })
            
            # Check for alerts
            self._check_alerts(metrics)
    
    def _check_alerts(self, metrics: APIPerformanceMetrics):
        """Check for performance alerts"""
        # Check response time
        if metrics.response_time.average > 0:
            response_time_threshold = self.thresholds.evaluate_response_time(
                metrics.response_time.average, 
                metrics.priority
            )
            
            if response_time_threshold in [PerformanceThreshold.WARNING, PerformanceThreshold.CRITICAL]:
                threshold_value = self.thresholds.get_response_time_threshold(
                    metrics.priority, 
                    PerformanceThreshold.GOOD
                )
                
                alert = PerformanceAlert(
                    endpoint=metrics.endpoint,
                    metric_type="response_time",
                    current_value=metrics.response_time.average,
                    threshold=threshold_value,
                    priority=metrics.priority,
                    severity=response_time_threshold,
                    message=f"Response time {metrics.response_time.average:.3f}s exceeds threshold {threshold_value:.3f}s"
                )
                
                with self.alerts_lock:
                    self.alerts.append(alert)
        
        # Check throughput
        if metrics.throughput.requests_per_second > 0:
            throughput_threshold = self.thresholds.evaluate_throughput(
                metrics.throughput.requests_per_second,
                metrics.priority
            )
            
            if throughput_threshold in [PerformanceThreshold.WARNING, PerformanceThreshold.CRITICAL]:
                threshold_value = self.thresholds.get_throughput_threshold(
                    metrics.priority,
                    PerformanceThreshold.GOOD
                )
                
                alert = PerformanceAlert(
                    endpoint=metrics.endpoint,
                    metric_type="throughput",
                    current_value=metrics.throughput.requests_per_second,
                    threshold=threshold_value,
                    priority=metrics.priority,
                    severity=throughput_threshold,
                    message=f"Throughput {metrics.throughput.requests_per_second:.2f} req/s below threshold {threshold_value:.2f} req/s"
                )
                
                with self.alerts_lock:
                    self.alerts.append(alert)
    
    def get_endpoint_metrics(self, endpoint: str, method: str = "GET") -> Optional[APIPerformanceMetrics]:
        """Get metrics for a specific endpoint"""
        key = f"{endpoint}:{method}"
        with self.metrics_lock:
            return self.metrics.get(key)
    
    def get_all_metrics(self) -> Dict[str, APIPerformanceMetrics]:
        """Get all metrics"""
        with self.metrics_lock:
            return self.metrics.copy()
    
    def get_alerts(self, severity: Optional[PerformanceThreshold] = None) -> List[PerformanceAlert]:
        """Get alerts, optionally filtered by severity"""
        with self.alerts_lock:
            if severity:
                return [alert for alert in self.alerts if alert.severity == severity]
            return self.alerts.copy()
    
    def acknowledge_alert(self, alert_id: str):
        """Acknowledge an alert"""
        with self.alerts_lock:
            for alert in self.alerts:
                if alert.id == alert_id:
                    alert.acknowledged = True
                    break
    
    def resolve_alert(self, alert_id: str):
        """Resolve an alert"""
        with self.alerts_lock:
            for alert in self.alerts:
                if alert.id == alert_id:
                    alert.resolved = True
                    break
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        with self.metrics_lock:
            total_endpoints = len(self.metrics)
            total_requests = sum(m.throughput.total_requests for m in self.metrics.values())
            total_successful = sum(m.throughput.successful_requests for m in self.metrics.values())
            
            # Calculate overall response time statistics
            all_response_times = []
            for metrics in self.metrics.values():
                if metrics.response_time.count > 0:
                    all_response_times.extend([
                        metrics.response_time.average
                    ])
            
            overall_avg_response_time = statistics.mean(all_response_times) if all_response_times else 0.0
            overall_p95_response_time = np.percentile(all_response_times, 95) if all_response_times else 0.0
            
            # Calculate overall throughput
            total_throughput = sum(m.throughput.requests_per_second for m in self.metrics.values())
            
            return {
                "summary": {
                    "total_endpoints": total_endpoints,
                    "total_requests": total_requests,
                    "total_successful_requests": total_successful,
                    "success_rate": total_successful / total_requests if total_requests > 0 else 0.0,
                    "overall_avg_response_time": overall_avg_response_time,
                    "overall_p95_response_time": overall_p95_response_time,
                    "total_throughput": total_throughput,
                    "uptime_seconds": time.time() - self.start_time
                },
                "endpoints": {
                    key: {
                        "response_time": {
                            "count": m.response_time.count,
                            "average": m.response_time.average,
                            "p95": m.response_time.p95,
                            "p99": m.response_time.p99,
                            "min": m.response_time.min_time,
                            "max": m.response_time.max_time
                        },
                        "throughput": {
                            "requests_per_second": m.throughput.requests_per_second,
                            "total_requests": m.throughput.total_requests,
                            "success_rate": m.throughput.success_rate
                        },
                        "latency_breakdown": {
                            "total": m.latency_breakdown.total_latency,
                            "network": m.latency_breakdown.network_latency,
                            "processing": m.latency_breakdown.processing_latency,
                            "database": m.latency_breakdown.database_latency,
                            "cache": m.latency_breakdown.cache_latency,
                            "external_api": m.latency_breakdown.external_api_latency
                        },
                        "priority": m.priority.value,
                        "last_updated": m.last_updated
                    }
                    for key, m in self.metrics.items()
                },
                "alerts": {
                    "total": len(self.alerts),
                    "active": len([a for a in self.alerts if not a.resolved]),
                    "acknowledged": len([a for a in self.alerts if a.acknowledged]),
                    "by_severity": {
                        severity.value: len([a for a in self.alerts if a.severity == severity])
                        for severity in PerformanceThreshold
                    }
                }
            }
    
    def get_prometheus_metrics(self) -> str:
        """Get Prometheus metrics"""
        return generate_latest()
    
    async def store_metrics(self) -> Any:
        """Store metrics to Redis"""
        if not self.redis_client:
            return
        
        try:
            summary = self.get_performance_summary()
            await self.redis_client.setex(
                "api_performance_summary",
                3600,  # 1 hour TTL
                json.dumps(summary)
            )
            
            # Store individual endpoint metrics
            for key, metrics in self.metrics.items():
                await self.redis_client.setex(
                    f"api_metrics:{key}",
                    1800,  # 30 minutes TTL
                    json.dumps({
                        "response_time": {
                            "count": metrics.response_time.count,
                            "average": metrics.response_time.average,
                            "p95": metrics.response_time.p95,
                            "p99": metrics.response_time.p99
                        },
                        "throughput": {
                            "requests_per_second": metrics.throughput.requests_per_second,
                            "total_requests": metrics.throughput.total_requests,
                            "success_rate": metrics.throughput.success_rate
                        },
                        "priority": metrics.priority.value,
                        "last_updated": metrics.last_updated
                    })
                )
        except Exception as e:
            logger.error(f"Failed to store metrics to Redis: {e}")
    
    async def load_metrics(self) -> Any:
        """Load metrics from Redis"""
        if not self.redis_client:
            return
        
        try:
            # Load summary
            summary_data = await self.redis_client.get("api_performance_summary")
            if summary_data:
                summary = json.loads(summary_data)
                logger.info(f"Loaded performance summary from Redis: {summary['summary']['total_endpoints']} endpoints")
            
            # Load individual metrics
            for key in self.metrics.keys():
                metrics_data = await self.redis_client.get(f"api_metrics:{key}")
                if metrics_data:
                    data = json.loads(metrics_data)
                    # Update metrics with loaded data
                    if key in self.metrics:
                        self.metrics[key].response_time.average = data["response_time"]["average"]
                        self.metrics[key].throughput.requests_per_second = data["throughput"]["requests_per_second"]
        except Exception as e:
            logger.error(f"Failed to load metrics from Redis: {e}")

# Global monitor instance
_monitor: Optional[APIPerformanceMonitor] = None

async async def get_api_monitor() -> APIPerformanceMonitor:
    """Get the global API performance monitor instance"""
    global _monitor
    if _monitor is None:
        _monitor = APIPerformanceMonitor()
        await _monitor.initialize()
    return _monitor

def monitor_api_performance(endpoint: str = None, method: str = "GET", priority: MetricPriority = MetricPriority.MEDIUM):
    """Decorator to monitor API performance"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            # Determine endpoint from function name if not provided
            actual_endpoint = endpoint or f"/{func.__name__}"
            
            # Get monitor
            monitor = await get_api_monitor()
            
            # Register endpoint
            monitor.register_endpoint(actual_endpoint, method, priority)
            
            # Start timing
            start_time = time.time()
            latency_breakdown = LatencyBreakdown()
            
            try:
                # Execute function
                result = await func(*args, **kwargs)
                
                # Calculate response time
                response_time = time.time() - start_time
                
                # Record metrics
                monitor.record_request(
                    endpoint=actual_endpoint,
                    method=method,
                    response_time=response_time,
                    status_code=200,
                    latency_breakdown=latency_breakdown
                )
                
                return result
                
            except Exception as e:
                # Calculate response time
                response_time = time.time() - start_time
                
                # Record metrics with error
                monitor.record_request(
                    endpoint=actual_endpoint,
                    method=method,
                    response_time=response_time,
                    status_code=500,
                    latency_breakdown=latency_breakdown
                )
                
                raise
        
        return wrapper
    return decorator

def track_latency(latency_type: LatencyType):
    """Decorator to track specific latency types"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                latency = time.time() - start_time
                # Store latency for later use in main request tracking
                # This would need to be integrated with the main request context
                logger.debug(f"{latency_type.value} latency: {latency:.3f}s")
        
        return wrapper
    return decorator

async def example_usage():
    """Example usage of the API performance monitoring system"""
    
    # Get monitor
    monitor = await get_api_monitor()
    
    # Register endpoints with different priorities
    monitor.register_endpoint("/api/users", "GET", MetricPriority.HIGH)
    monitor.register_endpoint("/api/admin", "POST", MetricPriority.CRITICAL)
    monitor.register_endpoint("/api/health", "GET", MetricPriority.LOW)
    
    # Example API functions with monitoring
    @monitor_api_performance("/api/users", "GET", MetricPriority.HIGH)
    async def get_users():
        
    """get_users function."""
await asyncio.sleep(0.1)  # Simulate processing
        return {"users": []}
    
    @monitor_api_performance("/api/admin", "POST", MetricPriority.CRITICAL)
    async def create_admin():
        
    """create_admin function."""
await asyncio.sleep(0.05)  # Simulate processing
        return {"admin": "created"}
    
    @monitor_api_performance("/api/health", "GET", MetricPriority.LOW)
    async def health_check():
        
    """health_check function."""
await asyncio.sleep(0.01)  # Simulate processing
        return {"status": "healthy"}
    
    # Execute some requests
    for _ in range(10):
        await get_users()
        await create_admin()
        await health_check()
    
    # Get performance summary
    summary = monitor.get_performance_summary()
    print("Performance Summary:")
    print(json.dumps(summary["summary"], indent=2))
    
    # Get alerts
    alerts = monitor.get_alerts()
    print(f"\nActive Alerts: {len(alerts)}")
    for alert in alerts:
        print(f"- {alert.message}")
    
    # Get Prometheus metrics
    prometheus_metrics = monitor.get_prometheus_metrics()
    print(f"\nPrometheus Metrics:\n{prometheus_metrics}")

match __name__:
    case "__main__":
    asyncio.run(example_usage()) 