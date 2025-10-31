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
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import logging
from contextlib import asynccontextmanager
from functools import wraps
import json
from typing import Any, List, Dict, Optional
#!/usr/bin/env python3
"""
API Performance Monitor

Comprehensive performance monitoring system for tracking API metrics:
- Response time and latency
- Throughput and request rates
- Database operation timing
- External API call performance
- Async operation monitoring
- Real-time metrics collection
"""


# Performance monitoring logger
logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics data structure."""
    request_id: str
    endpoint: str
    method: str
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    status_code: Optional[int] = None
    error: Optional[str] = None
    db_operations: List[Dict[str, Any]] = field(default_factory=list)
    external_calls: List[Dict[str, Any]] = field(default_factory=list)
    async_operations: List[Dict[str, Any]] = field(default_factory=list)
    memory_usage: Optional[float] = None
    cpu_usage: Optional[float] = None


class APIPerformanceMonitor:
    """
    Comprehensive API performance monitoring system.
    
    Tracks response time, latency, throughput, database operations,
    external API calls, and async operation performance.
    """
    
    def __init__(self, max_history: int = 10000):
        """Initialize the performance monitor."""
        self.max_history = max_history
        self.metrics_history: deque = deque(maxlen=max_history)
        self.current_requests: Dict[str, PerformanceMetrics] = {}
        self.endpoint_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_duration': 0.0,
            'min_duration': float('inf'),
            'max_duration': 0.0,
            'durations': deque(maxlen=1000),
            'error_counts': defaultdict(int),
            'db_operation_times': deque(maxlen=1000),
            'external_call_times': deque(maxlen=1000),
            'async_operation_times': deque(maxlen=1000)
        })
        
        # Real-time metrics
        self.request_rate = 0.0
        self.error_rate = 0.0
        self.avg_response_time = 0.0
        self.p95_response_time = 0.0
        self.p99_response_time = 0.0
        
        # Background task for metrics calculation
        self._metrics_task: Optional[asyncio.Task] = None
        self._running = False
    
    async def start_monitoring(self) -> None:
        """Start the performance monitoring system."""
        if self._running:
            return
        
        self._running = True
        self._metrics_task = asyncio.create_task(self._calculate_metrics_loop())
        logger.info("API Performance Monitor started")
    
    async def stop_monitoring(self) -> None:
        """Stop the performance monitoring system."""
        self._running = False
        if self._metrics_task:
            self._metrics_task.cancel()
            try:
                await self._metrics_task
            except asyncio.CancelledError:
                pass
        logger.info("API Performance Monitor stopped")
    
    async def start_request(self, request_id: str, endpoint: str, method: str) -> None:
        """Start tracking a new request."""
        metrics = PerformanceMetrics(
            request_id=request_id,
            endpoint=endpoint,
            method=method,
            start_time=time.time()
        )
        self.current_requests[request_id] = metrics
    
    async def end_request(self, request_id: str, status_code: int, error: Optional[str] = None) -> None:
        """End tracking a request and calculate metrics."""
        if request_id not in self.current_requests:
            return
        
        metrics = self.current_requests[request_id]
        metrics.end_time = time.time()
        metrics.duration = metrics.end_time - metrics.start_time
        metrics.status_code = status_code
        metrics.error = error
        
        # Add to history
        self.metrics_history.append(metrics)
        
        # Update endpoint statistics
        self._update_endpoint_stats(metrics)
        
        # Remove from current requests
        del self.current_requests[request_id]
    
    def record_db_operation(self, request_id: str, operation: str, duration: float, success: bool) -> None:
        """Record database operation metrics."""
        if request_id not in self.current_requests:
            return
        
        db_op = {
            'operation': operation,
            'duration': duration,
            'success': success,
            'timestamp': time.time()
        }
        self.current_requests[request_id].db_operations.append(db_op)
        
        # Update endpoint stats
        endpoint = self.current_requests[request_id].endpoint
        self.endpoint_stats[endpoint]['db_operation_times'].append(duration)
    
    def record_external_call(self, request_id: str, service: str, duration: float, success: bool) -> None:
        """Record external API call metrics."""
        if request_id not in self.current_requests:
            return
        
        external_call = {
            'service': service,
            'duration': duration,
            'success': success,
            'timestamp': time.time()
        }
        self.current_requests[request_id].external_calls.append(external_call)
        
        # Update endpoint stats
        endpoint = self.current_requests[request_id].endpoint
        self.endpoint_stats[endpoint]['external_call_times'].append(duration)
    
    def record_async_operation(self, request_id: str, operation: str, duration: float, success: bool) -> None:
        """Record async operation metrics."""
        if request_id not in self.current_requests:
            return
        
        async_op = {
            'operation': operation,
            'duration': duration,
            'success': success,
            'timestamp': time.time()
        }
        self.current_requests[request_id].async_operations.append(async_op)
        
        # Update endpoint stats
        endpoint = self.current_requests[request_id].endpoint
        self.endpoint_stats[endpoint]['async_operation_times'].append(duration)
    
    def _update_endpoint_stats(self, metrics: PerformanceMetrics) -> None:
        """Update endpoint statistics."""
        endpoint = metrics.endpoint
        stats = self.endpoint_stats[endpoint]
        
        stats['total_requests'] += 1
        stats['total_duration'] += metrics.duration or 0.0
        
        if metrics.duration:
            stats['min_duration'] = min(stats['min_duration'], metrics.duration)
            stats['max_duration'] = max(stats['max_duration'], metrics.duration)
            stats['durations'].append(metrics.duration)
        
        if metrics.status_code and 200 <= metrics.status_code < 400:
            stats['successful_requests'] += 1
        else:
            stats['failed_requests'] += 1
            if metrics.error:
                stats['error_counts'][metrics.error] += 1
    
    async def _calculate_metrics_loop(self) -> None:
        """Background task to calculate real-time metrics."""
        while self._running:
            try:
                await self._calculate_real_time_metrics()
                await asyncio.sleep(5)  # Update every 5 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in metrics calculation loop: {e}")
                await asyncio.sleep(5)
    
    async def _calculate_real_time_metrics(self) -> None:
        """Calculate real-time performance metrics."""
        now = time.time()
        window_start = now - 60  # Last 60 seconds
        
        # Filter recent metrics
        recent_metrics = [
            m for m in self.metrics_history
            if m.end_time and m.end_time >= window_start
        ]
        
        if not recent_metrics:
            return
        
        # Calculate request rate
        self.request_rate = len(recent_metrics) / 60.0
        
        # Calculate error rate
        error_count = sum(1 for m in recent_metrics if m.status_code and m.status_code >= 400)
        self.error_rate = error_count / len(recent_metrics) if recent_metrics else 0.0
        
        # Calculate response time percentiles
        durations = [m.duration for m in recent_metrics if m.duration]
        if durations:
            self.avg_response_time = statistics.mean(durations)
            sorted_durations = sorted(durations)
            p95_index = int(len(sorted_durations) * 0.95)
            p99_index = int(len(sorted_durations) * 0.99)
            
            self.p95_response_time = sorted_durations[p95_index] if p95_index < len(sorted_durations) else sorted_durations[-1]
            self.p99_response_time = sorted_durations[p99_index] if p99_index < len(sorted_durations) else sorted_durations[-1]
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        return {
            'real_time': {
                'request_rate': self.request_rate,
                'error_rate': self.error_rate,
                'avg_response_time': self.avg_response_time,
                'p95_response_time': self.p95_response_time,
                'p99_response_time': self.p99_response_time,
                'active_requests': len(self.current_requests)
            },
            'endpoints': {
                endpoint: {
                    'total_requests': stats['total_requests'],
                    'successful_requests': stats['successful_requests'],
                    'failed_requests': stats['failed_requests'],
                    'avg_duration': statistics.mean(stats['durations']) if stats['durations'] else 0.0,
                    'min_duration': stats['min_duration'] if stats['min_duration'] != float('inf') else 0.0,
                    'max_duration': stats['max_duration'],
                    'p95_duration': self._calculate_percentile(stats['durations'], 95),
                    'p99_duration': self._calculate_percentile(stats['durations'], 99),
                    'avg_db_operation_time': statistics.mean(stats['db_operation_times']) if stats['db_operation_times'] else 0.0,
                    'avg_external_call_time': statistics.mean(stats['external_call_times']) if stats['external_call_times'] else 0.0,
                    'avg_async_operation_time': statistics.mean(stats['async_operation_times']) if stats['async_operation_times'] else 0.0,
                    'error_counts': dict(stats['error_counts'])
                }
                for endpoint, stats in self.endpoint_stats.items()
            },
            'system': {
                'total_metrics_recorded': len(self.metrics_history),
                'monitoring_active': self._running
            }
        }
    
    def _calculate_percentile(self, values: deque, percentile: int) -> float:
        """Calculate percentile from a deque of values."""
        if not values:
            return 0.0
        
        sorted_values = sorted(values)
        index = int(len(sorted_values) * percentile / 100)
        return sorted_values[index] if index < len(sorted_values) else sorted_values[-1]
    
    def get_endpoint_metrics(self, endpoint: str) -> Optional[Dict[str, Any]]:
        """Get metrics for a specific endpoint."""
        if endpoint not in self.endpoint_stats:
            return None
        
        stats = self.endpoint_stats[endpoint]
        return {
            'endpoint': endpoint,
            'total_requests': stats['total_requests'],
            'successful_requests': stats['successful_requests'],
            'failed_requests': stats['failed_requests'],
            'success_rate': stats['successful_requests'] / stats['total_requests'] if stats['total_requests'] > 0 else 0.0,
            'avg_duration': statistics.mean(stats['durations']) if stats['durations'] else 0.0,
            'min_duration': stats['min_duration'] if stats['min_duration'] != float('inf') else 0.0,
            'max_duration': stats['max_duration'],
            'p95_duration': self._calculate_percentile(stats['durations'], 95),
            'p99_duration': self._calculate_percentile(stats['durations'], 99),
            'avg_db_operation_time': statistics.mean(stats['db_operation_times']) if stats['db_operation_times'] else 0.0,
            'avg_external_call_time': statistics.mean(stats['external_call_times']) if stats['external_call_times'] else 0.0,
            'avg_async_operation_time': statistics.mean(stats['async_operation_times']) if stats['async_operation_times'] else 0.0,
            'error_counts': dict(stats['error_counts'])
        }


# Global performance monitor instance
performance_monitor = APIPerformanceMonitor()


# Decorators for easy performance monitoring
def monitor_performance(endpoint: str):
    """Decorator to monitor function performance."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            request_id = f"{endpoint}_{int(time.time() * 1000)}"
            performance_monitor.start_request(request_id, endpoint, "POST")
            
            try:
                result = await func(*args, **kwargs)
                performance_monitor.end_request(request_id, 200)
                return result
            except Exception as e:
                performance_monitor.end_request(request_id, 500, str(e))
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            request_id = f"{endpoint}_{int(time.time() * 1000)}"
            performance_monitor.start_request(request_id, endpoint, "POST")
            
            try:
                result = func(*args, **kwargs)
                performance_monitor.end_request(request_id, 200)
                return result
            except Exception as e:
                performance_monitor.end_request(request_id, 500, str(e))
                raise
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator


@asynccontextmanager
async def monitor_db_operation(request_id: str, operation: str):
    """Context manager for monitoring database operations."""
    start_time = time.time()
    try:
        yield
        duration = time.time() - start_time
        performance_monitor.record_db_operation(request_id, operation, duration, True)
    except Exception as e:
        duration = time.time() - start_time
        performance_monitor.record_db_operation(request_id, operation, duration, False)
        raise


@asynccontextmanager
async def monitor_external_call(request_id: str, service: str):
    """Context manager for monitoring external API calls."""
    start_time = time.time()
    try:
        yield
        duration = time.time() - start_time
        performance_monitor.record_external_call(request_id, service, duration, True)
    except Exception as e:
        duration = time.time() - start_time
        performance_monitor.record_external_call(request_id, service, duration, False)
        raise


@asynccontextmanager
async def monitor_async_operation(request_id: str, operation: str):
    """Context manager for monitoring async operations."""
    start_time = time.time()
    try:
        yield
        duration = time.time() - start_time
        performance_monitor.record_async_operation(request_id, operation, duration, True)
    except Exception as e:
        duration = time.time() - start_time
        performance_monitor.record_async_operation(request_id, operation, duration, False)
        raise


# Utility functions for performance monitoring
async def get_performance_metrics() -> Dict[str, Any]:
    """Get current performance metrics."""
    return performance_monitor.get_current_metrics()


async def get_endpoint_performance(endpoint: str) -> Optional[Dict[str, Any]]:
    """Get performance metrics for a specific endpoint."""
    return performance_monitor.get_endpoint_metrics(endpoint)


async def start_performance_monitoring() -> None:
    """Start the performance monitoring system."""
    await performance_monitor.start_monitoring()


async def stop_performance_monitoring() -> None:
    """Stop the performance monitoring system."""
    await performance_monitor.stop_monitoring() 