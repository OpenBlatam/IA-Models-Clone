"""
Ultra-Performance Monitoring Middleware
======================================

High-performance middleware for monitoring API performance, metrics collection,
and real-time performance analysis.

Key Features:
- Request/response timing
- Memory usage monitoring
- CPU usage tracking
- Cache performance metrics
- Error rate monitoring
- Throughput measurement
- Real-time alerts
- Performance dashboards

Author: TruthGPT Development Team
Version: 1.0.0 - Ultra-Performance
License: MIT
"""

import time
import asyncio
import psutil
import logging
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
from contextlib import asynccontextmanager
import threading
from functools import wraps
import json

from fastapi import FastAPI, Request, Response
from fastapi.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Performance metrics container."""
    request_count: int = 0
    total_response_time: float = 0.0
    min_response_time: float = float('inf')
    max_response_time: float = 0.0
    error_count: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    memory_usage: float = 0.0
    cpu_usage: float = 0.0
    active_connections: int = 0
    throughput: float = 0.0
    last_updated: float = field(default_factory=time.time)

@dataclass
class RequestMetrics:
    """Individual request metrics."""
    method: str
    path: str
    start_time: float
    end_time: Optional[float] = None
    response_time: Optional[float] = None
    status_code: Optional[int] = None
    memory_before: float = 0.0
    memory_after: float = 0.0
    cache_hit: bool = False
    error: Optional[str] = None

class PerformanceMonitor:
    """Core performance monitoring system."""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.metrics = PerformanceMetrics()
        self.request_history: deque = deque(maxlen=max_history)
        self.endpoint_metrics: Dict[str, PerformanceMetrics] = defaultdict(PerformanceMetrics)
        self.error_counts: Dict[str, int] = defaultdict(int)
        self._lock = threading.Lock()
        self._start_time = time.time()
        
        # Performance thresholds
        self.thresholds = {
            "max_response_time": 5.0,  # 5 seconds
            "max_memory_usage": 80.0,  # 80%
            "max_cpu_usage": 90.0,    # 90%
            "max_error_rate": 0.05    # 5%
        }
        
        # Start background monitoring
        self._monitoring_task = None
        self._start_monitoring()
    
    def _start_monitoring(self):
        """Start background monitoring task."""
        if self._monitoring_task is None or self._monitoring_task.done():
            self._monitoring_task = asyncio.create_task(self._monitor_system())
    
    async def _monitor_system(self):
        """Background system monitoring."""
        while True:
            try:
                await asyncio.sleep(1.0)  # Monitor every second
                
                # Update system metrics
                with self._lock:
                    self.metrics.memory_usage = psutil.virtual_memory().percent
                    self.metrics.cpu_usage = psutil.cpu_percent()
                    self.metrics.active_connections = len(psutil.net_connections())
                    self.metrics.last_updated = time.time()
                    
                    # Calculate throughput (requests per second)
                    if self.metrics.request_count > 0:
                        uptime = time.time() - self._start_time
                        self.metrics.throughput = self.metrics.request_count / uptime
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"System monitoring error: {e}")
    
    def record_request(self, request_metrics: RequestMetrics):
        """Record request metrics."""
        with self._lock:
            # Update global metrics
            self.metrics.request_count += 1
            
            if request_metrics.response_time is not None:
                self.metrics.total_response_time += request_metrics.response_time
                self.metrics.min_response_time = min(
                    self.metrics.min_response_time, 
                    request_metrics.response_time
                )
                self.metrics.max_response_time = max(
                    self.metrics.max_response_time, 
                    request_metrics.response_time
                )
            
            if request_metrics.status_code and request_metrics.status_code >= 400:
                self.metrics.error_count += 1
                self.error_counts[f"{request_metrics.status_code}"] += 1
            
            if request_metrics.cache_hit:
                self.metrics.cache_hits += 1
            else:
                self.metrics.cache_misses += 1
            
            # Update endpoint-specific metrics
            endpoint_key = f"{request_metrics.method} {request_metrics.path}"
            endpoint_metrics = self.endpoint_metrics[endpoint_key]
            
            endpoint_metrics.request_count += 1
            if request_metrics.response_time is not None:
                endpoint_metrics.total_response_time += request_metrics.response_time
                endpoint_metrics.min_response_time = min(
                    endpoint_metrics.min_response_time,
                    request_metrics.response_time
                )
                endpoint_metrics.max_response_time = max(
                    endpoint_metrics.max_response_time,
                    request_metrics.response_time
                )
            
            if request_metrics.status_code and request_metrics.status_code >= 400:
                endpoint_metrics.error_count += 1
            
            # Add to history
            self.request_history.append(request_metrics)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        with self._lock:
            avg_response_time = (
                self.metrics.total_response_time / max(self.metrics.request_count, 1)
            )
            
            error_rate = (
                self.metrics.error_count / max(self.metrics.request_count, 1)
            )
            
            cache_hit_ratio = (
                self.metrics.cache_hits / max(self.metrics.cache_hits + self.metrics.cache_misses, 1)
            )
            
            return {
                "global": {
                    "request_count": self.metrics.request_count,
                    "avg_response_time": avg_response_time,
                    "min_response_time": self.metrics.min_response_time if self.metrics.min_response_time != float('inf') else 0,
                    "max_response_time": self.metrics.max_response_time,
                    "error_count": self.metrics.error_count,
                    "error_rate": error_rate,
                    "cache_hit_ratio": cache_hit_ratio,
                    "memory_usage": self.metrics.memory_usage,
                    "cpu_usage": self.metrics.cpu_usage,
                    "active_connections": self.metrics.active_connections,
                    "throughput": self.metrics.throughput,
                    "uptime": time.time() - self._start_time
                },
                "endpoints": {
                    endpoint: {
                        "request_count": metrics.request_count,
                        "avg_response_time": metrics.total_response_time / max(metrics.request_count, 1),
                        "min_response_time": metrics.min_response_time if metrics.min_response_time != float('inf') else 0,
                        "max_response_time": metrics.max_response_time,
                        "error_count": metrics.error_count,
                        "error_rate": metrics.error_count / max(metrics.request_count, 1)
                    }
                    for endpoint, metrics in self.endpoint_metrics.items()
                },
                "error_counts": dict(self.error_counts),
                "thresholds": self.thresholds
            }
    
    def check_alerts(self) -> List[Dict[str, Any]]:
        """Check for performance alerts."""
        alerts = []
        metrics = self.get_metrics()
        
        global_metrics = metrics["global"]
        
        # Response time alert
        if global_metrics["avg_response_time"] > self.thresholds["max_response_time"]:
            alerts.append({
                "type": "high_response_time",
                "severity": "warning",
                "message": f"Average response time {global_metrics['avg_response_time']:.2f}s exceeds threshold {self.thresholds['max_response_time']}s",
                "value": global_metrics["avg_response_time"],
                "threshold": self.thresholds["max_response_time"]
            })
        
        # Memory usage alert
        if global_metrics["memory_usage"] > self.thresholds["max_memory_usage"]:
            alerts.append({
                "type": "high_memory_usage",
                "severity": "critical",
                "message": f"Memory usage {global_metrics['memory_usage']:.1f}% exceeds threshold {self.thresholds['max_memory_usage']}%",
                "value": global_metrics["memory_usage"],
                "threshold": self.thresholds["max_memory_usage"]
            })
        
        # CPU usage alert
        if global_metrics["cpu_usage"] > self.thresholds["max_cpu_usage"]:
            alerts.append({
                "type": "high_cpu_usage",
                "severity": "warning",
                "message": f"CPU usage {global_metrics['cpu_usage']:.1f}% exceeds threshold {self.thresholds['max_cpu_usage']}%",
                "value": global_metrics["cpu_usage"],
                "threshold": self.thresholds["max_cpu_usage"]
            })
        
        # Error rate alert
        if global_metrics["error_rate"] > self.thresholds["max_error_rate"]:
            alerts.append({
                "type": "high_error_rate",
                "severity": "critical",
                "message": f"Error rate {global_metrics['error_rate']:.2%} exceeds threshold {self.thresholds['max_error_rate']:.2%}",
                "value": global_metrics["error_rate"],
                "threshold": self.thresholds["max_error_rate"]
            })
        
        return alerts
    
    def reset_metrics(self):
        """Reset all metrics."""
        with self._lock:
            self.metrics = PerformanceMetrics()
            self.endpoint_metrics.clear()
            self.error_counts.clear()
            self.request_history.clear()
            self._start_time = time.time()

# Global performance monitor
_performance_monitor = PerformanceMonitor()

class PerformanceMiddleware(BaseHTTPMiddleware):
    """Middleware for performance monitoring."""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.monitor = _performance_monitor
    
    async def dispatch(self, request: Request, call_next):
        # Record request start
        start_time = time.time()
        memory_before = psutil.virtual_memory().percent
        
        # Create request metrics
        request_metrics = RequestMetrics(
            method=request.method,
            path=request.url.path,
            start_time=start_time,
            memory_before=memory_before
        )
        
        try:
            # Process request
            response = await call_next(request)
            
            # Record response
            end_time = time.time()
            response_time = end_time - start_time
            memory_after = psutil.virtual_memory().percent
            
            request_metrics.end_time = end_time
            request_metrics.response_time = response_time
            request_metrics.status_code = response.status_code
            request_metrics.memory_after = memory_after
            
            # Add performance headers
            response.headers["X-Response-Time"] = str(response_time)
            response.headers["X-Memory-Usage"] = str(memory_after)
            response.headers["X-Request-ID"] = str(int(start_time * 1000000))
            
            # Record metrics
            self.monitor.record_request(request_metrics)
            
            return response
            
        except Exception as e:
            # Record error
            end_time = time.time()
            response_time = end_time - start_time
            
            request_metrics.end_time = end_time
            request_metrics.response_time = response_time
            request_metrics.status_code = 500
            request_metrics.error = str(e)
            
            # Record metrics
            self.monitor.record_request(request_metrics)
            
            # Return error response
            return JSONResponse(
                status_code=500,
                content={"error": "Internal server error", "request_id": str(int(start_time * 1000000))}
            )

class CachePerformanceMiddleware(BaseHTTPMiddleware):
    """Middleware for cache performance monitoring."""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.cache_stats = {
            "hits": 0,
            "misses": 0,
            "total_requests": 0
        }
    
    async def dispatch(self, request: Request, call_next):
        # Check if request is cache-related
        if "cache" in request.url.path.lower():
            self.cache_stats["total_requests"] += 1
            
            # Check for cache hit indicators in headers
            cache_hit = request.headers.get("X-Cache-Hit", "false").lower() == "true"
            
            if cache_hit:
                self.cache_stats["hits"] += 1
            else:
                self.cache_stats["misses"] += 1
        
        response = await call_next(request)
        
        # Add cache performance headers
        if self.cache_stats["total_requests"] > 0:
            hit_ratio = self.cache_stats["hits"] / self.cache_stats["total_requests"]
            response.headers["X-Cache-Hit-Ratio"] = str(hit_ratio)
            response.headers["X-Cache-Total-Requests"] = str(self.cache_stats["total_requests"])
        
        return response

class MemoryMonitoringMiddleware(BaseHTTPMiddleware):
    """Middleware for memory usage monitoring."""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.memory_history = deque(maxlen=100)
        self.peak_memory = 0.0
    
    async def dispatch(self, request: Request, call_next):
        # Record memory before request
        memory_before = psutil.virtual_memory().percent
        self.memory_history.append(memory_before)
        self.peak_memory = max(self.peak_memory, memory_before)
        
        response = await call_next(request)
        
        # Record memory after request
        memory_after = psutil.virtual_memory().percent
        memory_delta = memory_after - memory_before
        
        # Add memory headers
        response.headers["X-Memory-Before"] = str(memory_before)
        response.headers["X-Memory-After"] = str(memory_after)
        response.headers["X-Memory-Delta"] = str(memory_delta)
        response.headers["X-Peak-Memory"] = str(self.peak_memory)
        
        return response

# Performance monitoring decorators
def monitor_performance(func_name: str = None):
    """Decorator for monitoring function performance."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            memory_before = psutil.virtual_memory().percent
            
            try:
                result = await func(*args, **kwargs)
                
                end_time = time.time()
                execution_time = end_time - start_time
                memory_after = psutil.virtual_memory().percent
                
                logger.info(f"{func_name or func.__name__} executed in {execution_time:.4f}s, memory: {memory_before:.1f}% -> {memory_after:.1f}%")
                
                return result
                
            except Exception as e:
                end_time = time.time()
                execution_time = end_time - start_time
                logger.error(f"{func_name or func.__name__} failed after {execution_time:.4f}s: {e}")
                raise
        
        return wrapper
    return decorator

def monitor_memory_usage(func: Callable) -> Callable:
    """Decorator for monitoring memory usage."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        memory_before = psutil.virtual_memory().percent
        memory_before_mb = psutil.virtual_memory().used / 1024 / 1024
        
        result = await func(*args, **kwargs)
        
        memory_after = psutil.virtual_memory().percent
        memory_after_mb = psutil.virtual_memory().used / 1024 / 1024
        memory_delta = memory_after_mb - memory_before_mb
        
        logger.info(f"{func.__name__} memory usage: {memory_before:.1f}% -> {memory_after:.1f}% (delta: {memory_delta:.1f}MB)")
        
        return result
    
    return wrapper

# Performance utilities
async def get_performance_summary() -> Dict[str, Any]:
    """Get comprehensive performance summary."""
    metrics = _performance_monitor.get_metrics()
    alerts = _performance_monitor.check_alerts()
    
    return {
        "metrics": metrics,
        "alerts": alerts,
        "timestamp": time.time(),
        "status": "healthy" if not alerts else "degraded"
    }

async def get_endpoint_performance(endpoint: str) -> Dict[str, Any]:
    """Get performance metrics for specific endpoint."""
    metrics = _performance_monitor.get_metrics()
    
    if endpoint in metrics["endpoints"]:
        return {
            "endpoint": endpoint,
            "metrics": metrics["endpoints"][endpoint],
            "timestamp": time.time()
        }
    else:
        return {
            "endpoint": endpoint,
            "error": "Endpoint not found",
            "timestamp": time.time()
        }

def setup_performance_middleware(app: FastAPI):
    """Setup all performance monitoring middleware."""
    app.add_middleware(PerformanceMiddleware)
    app.add_middleware(CachePerformanceMiddleware)
    app.add_middleware(MemoryMonitoringMiddleware)
    
    logger.info("Performance monitoring middleware setup complete")

# Performance dashboard data
async def get_performance_dashboard_data() -> Dict[str, Any]:
    """Get data for performance dashboard."""
    metrics = _performance_monitor.get_metrics()
    alerts = _performance_monitor.check_alerts()
    
    # Calculate additional metrics
    uptime = time.time() - _performance_monitor._start_time
    requests_per_minute = (metrics["global"]["request_count"] / uptime) * 60 if uptime > 0 else 0
    
    # Get recent request history
    recent_requests = list(_performance_monitor.request_history)[-50:]  # Last 50 requests
    
    return {
        "overview": {
            "uptime": uptime,
            "total_requests": metrics["global"]["request_count"],
            "requests_per_minute": requests_per_minute,
            "avg_response_time": metrics["global"]["avg_response_time"],
            "error_rate": metrics["global"]["error_rate"],
            "cache_hit_ratio": metrics["global"]["cache_hit_ratio"],
            "memory_usage": metrics["global"]["memory_usage"],
            "cpu_usage": metrics["global"]["cpu_usage"]
        },
        "alerts": alerts,
        "top_endpoints": sorted(
            metrics["endpoints"].items(),
            key=lambda x: x[1]["request_count"],
            reverse=True
        )[:10],
        "recent_requests": [
            {
                "method": req.method,
                "path": req.path,
                "response_time": req.response_time,
                "status_code": req.status_code,
                "timestamp": req.start_time
            }
            for req in recent_requests
        ],
        "error_breakdown": metrics["error_counts"],
        "timestamp": time.time()
    }

# Cleanup function
async def cleanup_performance_monitoring():
    """Cleanup performance monitoring resources."""
    if _performance_monitor._monitoring_task and not _performance_monitor._monitoring_task.done():
        _performance_monitor._monitoring_task.cancel()
        try:
            await _performance_monitor._monitoring_task
        except asyncio.CancelledError:
            pass
    
    logger.info("Performance monitoring cleanup complete")
