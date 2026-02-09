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
import functools
import weakref
from typing import Any, Callable, Dict, Optional, Union, List
from collections import OrderedDict
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
    import psutil
    import redis
from typing import Any, List, Dict, Optional
"""
AI Video System - Performance Optimization

Production-ready performance optimization utilities including caching,
connection pooling, performance monitoring, and optimization strategies.
"""


try:
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics data class."""
    operation: str
    duration: float
    timestamp: datetime = field(default_factory=datetime.now)
    success: bool = True
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class PerformanceMonitor:
    """
    Performance monitoring and metrics collection.
    
    Provides:
    - Operation timing
    - Resource usage monitoring
    - Performance alerts
    - Metrics aggregation
    """
    
    def __init__(self, max_metrics: int = 1000):
        
    """__init__ function."""
self.max_metrics = max_metrics
        self.metrics: List[PerformanceMetrics] = []
        self.lock = threading.Lock()
        self.alert_thresholds = {
            'operation_time': 30.0,  # seconds
            'memory_usage': 0.8,     # 80%
            'cpu_usage': 0.9,        # 90%
            'disk_usage': 0.9        # 90%
        }
    
    def record_operation(
        self,
        operation: str,
        duration: float,
        success: bool = True,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record operation performance metrics."""
        metric = PerformanceMetrics(
            operation=operation,
            duration=duration,
            success=success,
            error=error,
            metadata=metadata or {}
        )
        
        with self.lock:
            self.metrics.append(metric)
            
            # Keep only recent metrics
            if len(self.metrics) > self.max_metrics:
                self.metrics.pop(0)
    
    def get_operation_stats(self, operation: str, window_minutes: int = 60) -> Dict[str, Any]:
        """Get statistics for a specific operation."""
        cutoff_time = datetime.now() - timedelta(minutes=window_minutes)
        
        with self.lock:
            recent_metrics = [
                m for m in self.metrics
                if m.operation == operation and m.timestamp > cutoff_time
            ]
        
        if not recent_metrics:
            return {
                'count': 0,
                'avg_duration': 0.0,
                'min_duration': 0.0,
                'max_duration': 0.0,
                'success_rate': 0.0
            }
        
        durations = [m.duration for m in recent_metrics]
        success_count = sum(1 for m in recent_metrics if m.success)
        
        return {
            'count': len(recent_metrics),
            'avg_duration': sum(durations) / len(durations),
            'min_duration': min(durations),
            'max_duration': max(durations),
            'success_rate': success_count / len(recent_metrics)
        }
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get current system resource statistics."""
        if not PSUTIL_AVAILABLE:
            return {'error': 'psutil not available'}
        
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            
            return {
                'memory': {
                    'rss': memory_info.rss,
                    'vms': memory_info.vms,
                    'percent': process.memory_percent()
                },
                'cpu': {
                    'percent': process.cpu_percent(),
                    'system_percent': psutil.cpu_percent(interval=1)
                },
                'disk': {
                    'usage': psutil.disk_usage('/').percent / 100
                },
                'threads': process.num_threads(),
                'open_files': len(process.open_files()),
                'connections': len(process.connections())
            }
        except Exception as e:
            logger.error(f"Failed to get system stats: {e}")
            return {'error': str(e)}
    
    def check_alerts(self) -> List[Dict[str, Any]]:
        """Check for performance alerts."""
        alerts = []
        system_stats = self.get_system_stats()
        
        if 'error' in system_stats:
            return alerts
        
        # Memory usage alert
        if system_stats['memory']['percent'] > self.alert_thresholds['memory_usage'] * 100:
            alerts.append({
                'type': 'memory_usage',
                'severity': 'warning',
                'message': f"High memory usage: {system_stats['memory']['percent']:.1f}%",
                'value': system_stats['memory']['percent'],
                'threshold': self.alert_thresholds['memory_usage'] * 100
            })
        
        # CPU usage alert
        if system_stats['cpu']['system_percent'] > self.alert_thresholds['cpu_usage'] * 100:
            alerts.append({
                'type': 'cpu_usage',
                'severity': 'warning',
                'message': f"High CPU usage: {system_stats['cpu']['system_percent']:.1f}%",
                'value': system_stats['cpu']['system_percent'],
                'threshold': self.alert_thresholds['cpu_usage'] * 100
            })
        
        # Disk usage alert
        if system_stats['disk']['usage'] > self.alert_thresholds['disk_usage']:
            alerts.append({
                'type': 'disk_usage',
                'severity': 'warning',
                'message': f"High disk usage: {system_stats['disk']['usage']*100:.1f}%",
                'value': system_stats['disk']['usage'],
                'threshold': self.alert_thresholds['disk_usage']
            })
        
        return alerts


class LRUCache:
    """
    Thread-safe LRU (Least Recently Used) cache implementation.
    
    Features:
    - Configurable size limit
    - TTL (Time To Live) support
    - Thread-safe operations
    - Memory-efficient
    """
    
    def __init__(self, max_size: int = 100, ttl_seconds: Optional[int] = None):
        
    """__init__ function."""
self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: OrderedDict = OrderedDict()
        self.timestamps: Dict[str, float] = {}
        self.lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self.lock:
            if key not in self.cache:
                return None
            
            # Check TTL
            if self.ttl_seconds and time.time() - self.timestamps[key] > self.ttl_seconds:
                self._remove(key)
                return None
            
            # Move to end (most recently used)
            value = self.cache.pop(key)
            self.cache[key] = value
            return value
    
    def set(self, key: str, value: Any) -> None:
        """Set value in cache."""
        with self.lock:
            # Remove if exists
            if key in self.cache:
                self.cache.pop(key)
            
            # Add new value
            self.cache[key] = value
            self.timestamps[key] = time.time()
            
            # Remove oldest if cache is full
            if len(self.cache) > self.max_size:
                oldest_key = next(iter(self.cache))
                self._remove(oldest_key)
    
    def _remove(self, key: str) -> None:
        """Remove key from cache."""
        if key in self.cache:
            self.cache.pop(key)
        if key in self.timestamps:
            self.timestamps.pop(key)
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self.lock:
            self.cache.clear()
            self.timestamps.clear()
    
    def size(self) -> int:
        """Get current cache size."""
        with self.lock:
            return len(self.cache)
    
    def keys(self) -> List[str]:
        """Get all cache keys."""
        with self.lock:
            return list(self.cache.keys())


class ConnectionPool:
    """
    Generic connection pool for managing reusable connections.
    
    Features:
    - Connection pooling
    - Health checks
    - Automatic reconnection
    - Connection limits
    """
    
    def __init__(
        self,
        max_connections: int = 10,
        max_idle_time: int = 300,
        health_check_interval: int = 60
    ):
        
    """__init__ function."""
self.max_connections = max_connections
        self.max_idle_time = max_idle_time
        self.health_check_interval = health_check_interval
        
        self.connections: List[Dict[str, Any]] = []
        self.available_connections: List[Dict[str, Any]] = []
        self.lock = threading.RLock()
        self.last_health_check = time.time()
    
    async def get_connection(self, factory_func: Callable) -> Optional[Dict[str, Any]]:
        """Get a connection from the pool."""
        with self.lock:
            # Check for available connections
            if self.available_connections:
                conn_info = self.available_connections.pop()
                
                # Check if connection is still valid
                if await self._is_connection_healthy(conn_info['connection']):
                    return conn_info['connection']
                else:
                    # Remove unhealthy connection
                    self.connections.remove(conn_info)
            
            # Create new connection if pool not full
            if len(self.connections) < self.max_connections:
                try:
                    connection = await factory_func()
                    conn_info = {
                        'connection': connection,
                        'created_at': time.time(),
                        'last_used': time.time()
                    }
                    self.connections.append(conn_info)
                    return connection
                except Exception as e:
                    logger.error(f"Failed to create connection: {e}")
                    raise
            
            # Wait for available connection
            raise Exception("Connection pool exhausted")
    
    async def return_connection(self, connection: Any) -> None:
        """Return a connection to the pool."""
        with self.lock:
            # Find connection in pool
            for conn_info in self.connections:
                if conn_info['connection'] == connection:
                    conn_info['last_used'] = time.time()
                    self.available_connections.append(conn_info)
                    return
            
            # Connection not found in pool, close it
            await self._close_connection(connection)
    
    async def _is_connection_healthy(self, connection: Any) -> bool:
        """Check if connection is healthy."""
        try:
            # This is a generic implementation
            # Specific connection types should override this method
            return True
        except Exception:
            return False
    
    async def _close_connection(self, connection: Any) -> None:
        """Close a connection."""
        try:
            if hasattr(connection, 'close'):
                await connection.close()
            elif hasattr(connection, 'close'):
                connection.close()
        except Exception as e:
            logger.warning(f"Error closing connection: {e}")
    
    async def cleanup_idle_connections(self) -> int:
        """Clean up idle connections."""
        with self.lock:
            current_time = time.time()
            removed_count = 0
            
            # Remove idle connections
            for conn_info in self.connections[:]:
                if current_time - conn_info['last_used'] > self.max_idle_time:
                    await self._close_connection(conn_info['connection'])
                    self.connections.remove(conn_info)
                    if conn_info in self.available_connections:
                        self.available_connections.remove(conn_info)
                    removed_count += 1
            
            return removed_count
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on all connections."""
        with self.lock:
            current_time = time.time()
            
            # Only check periodically
            if current_time - self.last_health_check < self.health_check_interval:
                return {'status': 'skipped', 'reason': 'too_frequent'}
            
            self.last_health_check = current_time
            
            healthy_count = 0
            total_count = len(self.connections)
            
            for conn_info in self.connections:
                if await self._is_connection_healthy(conn_info['connection']):
                    healthy_count += 1
                else:
                    # Remove unhealthy connection
                    await self._close_connection(conn_info['connection'])
                    self.connections.remove(conn_info)
                    if conn_info in self.available_connections:
                        self.available_connections.remove(conn_info)
            
            return {
                'status': 'completed',
                'total_connections': total_count,
                'healthy_connections': healthy_count,
                'available_connections': len(self.available_connections)
            }


class AsyncRateLimiter:
    """
    Async rate limiter with sliding window implementation.
    
    Features:
    - Sliding window algorithm
    - Configurable limits
    - Distributed rate limiting support
    - Token bucket algorithm
    """
    
    def __init__(
        self,
        max_requests: int,
        window_seconds: int,
        distributed: bool = False,
        redis_client: Optional[Any] = None
    ):
        
    """__init__ function."""
self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.distributed = distributed
        self.redis_client = redis_client
        
        if not distributed:
            self.requests: List[float] = []
            self.lock = asyncio.Lock()
    
    async def acquire(self, key: str = "default") -> bool:
        """Acquire permission to make a request."""
        current_time = time.time()
        
        if self.distributed and self.redis_client:
            return await self._distributed_acquire(key, current_time)
        else:
            return await self._local_acquire(current_time)
    
    async def _local_acquire(self, current_time: float) -> bool:
        """Local rate limiting implementation."""
        async with self.lock:
            # Remove old requests outside the window
            cutoff_time = current_time - self.window_seconds
            self.requests = [req_time for req_time in self.requests if req_time > cutoff_time]
            
            # Check if we can make a new request
            if len(self.requests) < self.max_requests:
                self.requests.append(current_time)
                return True
            
            return False
    
    async def _distributed_acquire(self, key: str, current_time: float) -> bool:
        """Distributed rate limiting using Redis."""
        if not self.redis_client:
            return await self._local_acquire(current_time)
        
        try:
            # Use Redis sorted set for sliding window
            window_key = f"rate_limit:{key}"
            cutoff_time = current_time - self.window_seconds
            
            # Remove old entries
            await self.redis_client.zremrangebyscore(window_key, 0, cutoff_time)
            
            # Count current requests
            current_count = await self.redis_client.zcard(window_key)
            
            if current_count < self.max_requests:
                # Add new request
                await self.redis_client.zadd(window_key, {str(current_time): current_time})
                await self.redis_client.expire(window_key, self.window_seconds)
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Redis rate limiting error: {e}")
            return await self._local_acquire(current_time)
    
    async async def get_remaining_requests(self, key: str = "default") -> int:
        """Get remaining requests allowed."""
        current_time = time.time()
        
        if self.distributed and self.redis_client:
            try:
                window_key = f"rate_limit:{key}"
                cutoff_time = current_time - self.window_seconds
                await self.redis_client.zremrangebyscore(window_key, 0, cutoff_time)
                current_count = await self.redis_client.zcard(window_key)
                return max(0, self.max_requests - current_count)
            except Exception as e:
                logger.error(f"Redis error getting remaining requests: {e}")
                return 0
        else:
            async with self.lock:
                cutoff_time = current_time - self.window_seconds
                self.requests = [req_time for req_time in self.requests if req_time > cutoff_time]
                return max(0, self.max_requests - len(self.requests))


# Performance decorators and utilities
def measure_performance(operation_name: str):
    """Decorator to measure function performance."""
    def decorator(func) -> Any:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                monitor.record_operation(operation_name, duration, success=True)
                return result
            except Exception as e:
                duration = time.time() - start_time
                monitor.record_operation(operation_name, duration, success=False, error=str(e))
                raise
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                monitor.record_operation(operation_name, duration, success=True)
                return result
            except Exception as e:
                duration = time.time() - start_time
                monitor.record_operation(operation_name, duration, success=False, error=str(e))
                raise
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def cache_result(ttl_seconds: Optional[int] = None, max_size: int = 100):
    """Decorator to cache function results."""
    cache = LRUCache(max_size=max_size, ttl_seconds=ttl_seconds)
    
    def decorator(func) -> Any:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            # Create cache key
            cache_key = f"{func.__name__}:{hash(str(args) + str(sorted(kwargs.items())))}"
            
            # Try to get from cache
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Cache result
            cache.set(cache_key, result)
            return result
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            # Create cache key
            cache_key = f"{func.__name__}:{hash(str(args) + str(sorted(kwargs.items())))}"
            
            # Try to get from cache
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Cache result
            cache.set(cache_key, result)
            return result
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def rate_limit(max_requests: int, window_seconds: int, key_func: Optional[Callable] = None):
    """Decorator to apply rate limiting."""
    limiter = AsyncRateLimiter(max_requests, window_seconds)
    
    def decorator(func) -> Any:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            # Get rate limit key
            if key_func:
                key = key_func(*args, **kwargs)
            else:
                key = "default"
            
            # Check rate limit
            if not await limiter.acquire(key):
                raise Exception(f"Rate limit exceeded: {max_requests} requests per {window_seconds} seconds")
            
            return await func(*args, **kwargs)
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            # Get rate limit key
            if key_func:
                key = key_func(*args, **kwargs)
            else:
                key = "default"
            
            # For sync functions, we need to run the limiter in an event loop
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            # Check rate limit
            if not loop.run_until_complete(limiter.acquire(key)):
                raise Exception(f"Rate limit exceeded: {max_requests} requests per {window_seconds} seconds")
            
            return func(*args, **kwargs)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


# Global performance monitor instance
monitor = PerformanceMonitor()

# Global caches
caches: Dict[str, LRUCache] = {}

def get_cache(name: str, max_size: int = 100, ttl_seconds: Optional[int] = None) -> LRUCache:
    """Get or create a named cache."""
    if name not in caches:
        caches[name] = LRUCache(max_size=max_size, ttl_seconds=ttl_seconds)
    return caches[name]


def clear_all_caches() -> None:
    """Clear all caches."""
    for cache in caches.values():
        cache.clear()


async def cleanup_performance_resources() -> None:
    """Cleanup performance-related resources."""
    # Clear caches
    clear_all_caches()
    
    # Clear performance metrics
    monitor.metrics.clear()
    
    logger.info("Performance resources cleaned up") 