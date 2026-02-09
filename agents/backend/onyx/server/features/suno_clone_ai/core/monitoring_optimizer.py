"""
Monitoring and Observability Optimizations

Optimizations for:
- Performance monitoring
- Metrics collection
- Logging optimization
- Tracing
- Health checks
"""

import logging
import time
import asyncio
from typing import Optional, Dict, Any, Callable
from functools import wraps
from contextlib import contextmanager
import psutil
import os

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """Performance monitoring with minimal overhead."""
    
    def __init__(self):
        """Initialize performance monitor."""
        self.metrics: Dict[str, list] = {}
    
    @contextmanager
    def measure(self, metric_name: str):
        """
        Measure execution time.
        
        Args:
            metric_name: Name of metric
        """
        start_time = time.time()
        start_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        
        try:
            yield
        finally:
            elapsed = time.time() - start_time
            end_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
            memory_delta = end_memory - start_memory
            
            if metric_name not in self.metrics:
                self.metrics[metric_name] = []
            
            self.metrics[metric_name].append({
                'time': elapsed,
                'memory_delta': memory_delta
            })
    
    def get_stats(self, metric_name: str) -> Optional[Dict[str, float]]:
        """
        Get statistics for metric.
        
        Args:
            metric_name: Name of metric
            
        Returns:
            Statistics dictionary
        """
        if metric_name not in self.metrics or not self.metrics[metric_name]:
            return None
        
        times = [m['time'] for m in self.metrics[metric_name]]
        memories = [m['memory_delta'] for m in self.metrics[metric_name]]
        
        return {
            'count': len(times),
            'avg_time': sum(times) / len(times),
            'min_time': min(times),
            'max_time': max(times),
            'total_time': sum(times),
            'avg_memory': sum(memories) / len(memories),
            'max_memory': max(memories)
        }
    
    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all metrics."""
        return {
            name: self.get_stats(name)
            for name in self.metrics.keys()
        }


def monitor_performance(metric_name: Optional[str] = None):
    """
    Decorator to monitor function performance.
    
    Args:
        metric_name: Name of metric (defaults to function name)
    """
    def decorator(func: Callable):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            name = metric_name or func.__name__
            monitor = PerformanceMonitor()
            with monitor.measure(name):
                return await func(*args, **kwargs)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            name = metric_name or func.__name__
            monitor = PerformanceMonitor()
            with monitor.measure(name):
                return func(*args, **kwargs)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


class SystemMetrics:
    """System resource metrics."""
    
    @staticmethod
    def get_cpu_usage() -> float:
        """Get CPU usage percentage."""
        return psutil.cpu_percent(interval=0.1)
    
    @staticmethod
    def get_memory_usage() -> Dict[str, float]:
        """Get memory usage."""
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,
            'vms_mb': memory_info.vms / 1024 / 1024,
            'percent': process.memory_percent()
        }
    
    @staticmethod
    def get_disk_usage(path: str = '/') -> Dict[str, Any]:
        """Get disk usage."""
        usage = psutil.disk_usage(path)
        
        return {
            'total_gb': usage.total / 1024 / 1024 / 1024,
            'used_gb': usage.used / 1024 / 1024 / 1024,
            'free_gb': usage.free / 1024 / 1024 / 1024,
            'percent': usage.percent
        }
    
    @staticmethod
    def get_all_metrics() -> Dict[str, Any]:
        """Get all system metrics."""
        return {
            'cpu': SystemMetrics.get_cpu_usage(),
            'memory': SystemMetrics.get_memory_usage(),
            'disk': SystemMetrics.get_disk_usage()
        }


class HealthChecker:
    """Health check optimization."""
    
    def __init__(self):
        """Initialize health checker."""
        self.checks: Dict[str, Callable] = {}
    
    def register_check(self, name: str, check_func: Callable) -> None:
        """
        Register health check.
        
        Args:
            name: Check name
            check_func: Check function
        """
        self.checks[name] = check_func
    
    async def check_all(self) -> Dict[str, Dict[str, Any]]:
        """
        Run all health checks.
        
        Returns:
            Dictionary of check results
        """
        results = {}
        
        for name, check_func in self.checks.items():
            try:
                if asyncio.iscoroutinefunction(check_func):
                    result = await check_func()
                else:
                    result = check_func()
                
                results[name] = {
                    'status': 'healthy' if result else 'unhealthy',
                    'result': result
                }
            except Exception as e:
                results[name] = {
                    'status': 'error',
                    'error': str(e)
                }
        
        return results

