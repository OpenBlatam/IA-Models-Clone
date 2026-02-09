"""Performance optimization utilities."""

from functools import lru_cache, wraps
from typing import Callable, Any, Optional
import time
import asyncio
from collections import deque
from contextlib import asynccontextmanager

from utils.logger import get_logger
from utils.metrics import metrics_collector

logger = get_logger(__name__)


class PerformanceMonitor:
    """Monitor performance of operations."""
    
    def __init__(self, window_size: int = 100):
        """
        Initialize performance monitor.
        
        Args:
            window_size: Size of sliding window for metrics
        """
        self.window_size = window_size
        self.metrics: deque = deque(maxlen=window_size)
        self.measurements: dict[str, list[float]] = {}
    
    def record(self, operation: str, duration: float) -> None:
        """
        Record operation duration.
        
        Args:
            operation: Operation name
            duration: Duration in seconds
        """
        self.metrics.append({
            "operation": operation,
            "duration": duration,
            "timestamp": time.time()
        })
    
    def start(self, operation_name: str) -> float:
        """Start timing an operation."""
        return time.time()
    
    def end(self, operation_name: str, start_time: float) -> float:
        """End timing and record duration."""
        duration = time.time() - start_time
        if operation_name not in self.measurements:
            self.measurements[operation_name] = []
        self.measurements[operation_name].append(duration)
        self.record(operation_name, duration)
        metrics_collector.record_timing(f"performance.{operation_name}", duration)
        return duration
    
    async def measure_async(
        self,
        operation_name: str,
        coro: Callable
    ) -> Any:
        """Measure async operation."""
        start = time.time()
        try:
            result = await coro
            self.end(operation_name, start)
            return result
        except Exception as e:
            self.end(operation_name, start)
            logger.error(f"Error in {operation_name}: {e}")
            raise
    
    def get_stats(self, operation: Optional[str] = None) -> dict:
        """
        Get performance statistics.
        
        Args:
            operation: Optional operation name to filter
            
        Returns:
            Statistics dictionary
        """
        if operation:
            if operation in self.measurements:
                durations = self.measurements[operation]
            else:
                filtered = [m for m in self.metrics if m["operation"] == operation]
                durations = [m["duration"] for m in filtered] if filtered else []
        else:
            filtered = list(self.metrics)
            durations = [m["duration"] for m in filtered]
        
        if not durations:
            return {
                "count": 0,
                "avg": 0,
                "min": 0,
                "max": 0,
                "p95": 0,
                "p99": 0
            }
        
        durations_sorted = sorted(durations)
        
        return {
            "count": len(durations),
            "avg": sum(durations) / len(durations),
            "min": durations_sorted[0],
            "max": durations_sorted[-1],
            "p95": durations_sorted[int(len(durations_sorted) * 0.95)] if len(durations_sorted) > 1 else durations_sorted[0],
            "p99": durations_sorted[int(len(durations_sorted) * 0.99)] if len(durations_sorted) > 1 else durations_sorted[0],
        }


def optimize_memory(func: Callable) -> Callable:
    """
    Decorator to optimize memory usage.
    
    Args:
        func: Function to optimize
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        import gc
        gc.collect()  # Clean up before execution
        
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            return result
        finally:
            gc.collect()  # Clean up after execution
    
    return wrapper


class ConnectionPool:
    """Simple connection pool."""
    
    def __init__(self, factory: Callable, max_size: int = 10):
        """
        Initialize connection pool.
        
        Args:
            factory: Function to create connections
            max_size: Maximum pool size
        """
        self.factory = factory
        self.max_size = max_size
        self.pool: deque = deque()
        self.active = 0
        self._lock = asyncio.Lock()
    
    async def acquire(self):
        """Acquire connection from pool."""
        async with self._lock:
            if self.pool:
                return self.pool.popleft()
            
            if self.active < self.max_size:
                self.active += 1
                return self.factory()
            
            # Wait for available connection
            while not self.pool:
                await asyncio.sleep(0.1)
            
            return self.pool.popleft()
    
    async def release(self, conn):
        """Release connection back to pool."""
        async with self._lock:
            self.pool.append(conn)


def batch_process(items: list, batch_size: int = 10):
    """
    Generator to process items in batches.
    
    Args:
        items: List of items
        batch_size: Size of each batch
        
    Yields:
        Batches of items
    """
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


# Global performance monitor
performance_monitor = PerformanceMonitor()


def measure_performance(operation_name: str):
    """Decorator to measure function performance."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start = time.time()
            try:
                result = await func(*args, **kwargs)
                performance_monitor.end(operation_name, start)
                return result
            except Exception as e:
                performance_monitor.end(operation_name, start)
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start = time.time()
            try:
                result = func(*args, **kwargs)
                performance_monitor.end(operation_name, start)
                return result
            except Exception as e:
                performance_monitor.end(operation_name, start)
                raise
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


@asynccontextmanager
async def performance_context(operation_name: str):
    """Context manager for performance measurement."""
    start = time.time()
    try:
        yield
    finally:
        performance_monitor.end(operation_name, start)

