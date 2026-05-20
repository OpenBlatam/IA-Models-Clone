"""
Performance Optimizer for BUL System
====================================

Utilities for optimizing system performance and monitoring.
"""

import asyncio
import time
import logging
from typing import Dict, Any, Callable, Optional
from functools import wraps
from datetime import datetime
import statistics

logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """Monitor and track performance metrics"""
    
    def __init__(self):
        self.metrics: Dict[str, list] = {}
        self.call_counts: Dict[str, int] = {}
    
    def record_metric(self, name: str, value: float):
        """Record a performance metric"""
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(value)
        
        # Keep only last 1000 measurements
        if len(self.metrics[name]) > 1000:
            self.metrics[name] = self.metrics[name][-1000:]
    
    def record_call(self, name: str):
        """Record a function call"""
        self.call_counts[name] = self.call_counts.get(name, 0) + 1
    
    def get_stats(self, name: str) -> Dict[str, float]:
        """Get statistics for a metric"""
        if name not in self.metrics or not self.metrics[name]:
            return {}
        
        values = self.metrics[name]
        return {
            'count': len(values),
            'mean': statistics.mean(values),
            'median': statistics.median(values),
            'min': min(values),
            'max': max(values),
            'std': statistics.stdev(values) if len(values) > 1 else 0
        }
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get all performance statistics"""
        stats = {}
        for name in self.metrics:
            stats[name] = {
                'performance': self.get_stats(name),
                'call_count': self.call_counts.get(name, 0)
            }
        return stats

# Global performance monitor
_performance_monitor = PerformanceMonitor()

def get_performance_monitor() -> PerformanceMonitor:
    """Get the global performance monitor"""
    return _performance_monitor

def monitor_performance(metric_name: Optional[str] = None):
    """
    Decorator to monitor function performance
    
    Args:
        metric_name: Name for the metric (defaults to function name)
    """
    def decorator(func: Callable):
        name = metric_name or f"{func.__module__}.{func.__name__}"
        
        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                start_time = time.time()
                monitor = get_performance_monitor()
                monitor.record_call(name)
                
                try:
                    result = await func(*args, **kwargs)
                    return result
                finally:
                    duration = time.time() - start_time
                    monitor.record_metric(name, duration)
                    logger.debug(f"{name} took {duration:.3f}s")
            
            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                start_time = time.time()
                monitor = get_performance_monitor()
                monitor.record_call(name)
                
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    duration = time.time() - start_time
                    monitor.record_metric(name, duration)
                    logger.debug(f"{name} took {duration:.3f}s")
            
            return sync_wrapper
    
    return decorator

class ConnectionPool:
    """Simple connection pool for HTTP clients"""
    
    def __init__(self, max_connections: int = 10):
        self.max_connections = max_connections
        self.connections = []
        self.in_use = set()
        self.lock = asyncio.Lock()
    
    async def get_connection(self):
        """Get a connection from the pool"""
        async with self.lock:
            if self.connections:
                conn = self.connections.pop()
                self.in_use.add(conn)
                return conn
            elif len(self.in_use) < self.max_connections:
                # Create new connection
                conn = await self._create_connection()
                self.in_use.add(conn)
                return conn
            else:
                # Wait for a connection to be available
                while not self.connections:
                    await asyncio.sleep(0.01)
                conn = self.connections.pop()
                self.in_use.add(conn)
                return conn
    
    async def return_connection(self, conn):
        """Return a connection to the pool"""
        async with self.lock:
            if conn in self.in_use:
                self.in_use.remove(conn)
                self.connections.append(conn)
    
    async def _create_connection(self):
        """Create a new connection (to be implemented by subclasses)"""
        raise NotImplementedError

class OptimizedHTTPClient:
    """Optimized HTTP client with connection pooling and retries"""
    
    def __init__(self, max_retries: int = 3, timeout: float = 30.0):
        self.max_retries = max_retries
        self.timeout = timeout
        self.session = None
    
    async def __aenter__(self):
        import httpx
        self.session = httpx.AsyncClient(
            timeout=httpx.Timeout(self.timeout),
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=100)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.aclose()
    
    async def request_with_retry(self, method: str, url: str, **kwargs):
        """Make HTTP request with automatic retries"""
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                response = await self.session.request(method, url, **kwargs)
                return response
            except Exception as e:
                last_exception = e
                if attempt < self.max_retries:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.warning(f"Request failed (attempt {attempt + 1}), retrying in {wait_time}s: {e}")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"Request failed after {self.max_retries + 1} attempts: {e}")
        
        raise last_exception

def optimize_memory_usage():
    """Optimize memory usage by cleaning up unused objects"""
    import gc
    gc.collect()
    logger.info("Memory optimization completed")

def get_system_info() -> Dict[str, Any]:
    """Get system information for optimization"""
    try:
        import psutil
        return {
            'cpu_count': psutil.cpu_count(),
            'memory_total': psutil.virtual_memory().total,
            'memory_available': psutil.virtual_memory().available,
            'disk_usage': psutil.disk_usage('/').percent
        }
    except ImportError:
        return {'error': 'psutil not available'}

# Performance optimization utilities
async def batch_process(items: list, processor: Callable, batch_size: int = 10):
    """Process items in batches for better performance"""
    results = []
    
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        batch_results = await asyncio.gather(*[processor(item) for item in batch])
        results.extend(batch_results)
        
        # Small delay between batches to prevent overwhelming the system
        await asyncio.sleep(0.01)
    
    return results

def optimize_json_serialization(data: Any) -> str:
    """Optimize JSON serialization"""
    import json
    return json.dumps(data, separators=(',', ':'), ensure_ascii=False)

def create_performance_report() -> Dict[str, Any]:
    """Create a comprehensive performance report"""
    monitor = get_performance_monitor()
    stats = monitor.get_all_stats()
    
    return {
        'timestamp': datetime.now().isoformat(),
        'system_info': get_system_info(),
        'performance_metrics': stats,
        'recommendations': _generate_recommendations(stats)
    }

def _generate_recommendations(stats: Dict[str, Any]) -> list:
    """Generate performance optimization recommendations"""
    recommendations = []
    
    for name, data in stats.items():
        perf = data.get('performance', {})
        if perf.get('mean', 0) > 5.0:  # Functions taking more than 5 seconds
            recommendations.append(f"Consider optimizing {name} - average time: {perf['mean']:.2f}s")
        
        if perf.get('std', 0) > perf.get('mean', 0) * 0.5:  # High variance
            recommendations.append(f"High variance in {name} performance - consider caching")
    
    return recommendations




