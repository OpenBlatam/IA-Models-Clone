"""
Advanced async pool management for optimal resource utilization.
"""

import asyncio
import time
from typing import Dict, Any, List, Optional, Callable, Union, TypeVar, Generic
from dataclasses import dataclass, field
from collections import deque
import threading
import weakref
from contextlib import asynccontextmanager

from ..core.logging import get_logger
from ..core.config import get_settings

logger = get_logger(__name__)
T = TypeVar('T')


@dataclass
class PoolStats:
    """Statistics for a pool."""
    total_created: int = 0
    total_acquired: int = 0
    total_released: int = 0
    current_size: int = 0
    max_size: int = 0
    wait_time: float = 0.0
    last_activity: float = field(default_factory=time.time)


@dataclass
class PoolConfig:
    """Configuration for a pool."""
    min_size: int = 1
    max_size: int = 10
    max_idle_time: float = 300.0  # 5 minutes
    cleanup_interval: float = 60.0  # 1 minute
    acquire_timeout: float = 30.0
    health_check_interval: float = 300.0  # 5 minutes


class AsyncPool(Generic[T]):
    """Advanced async pool with health checks and auto-scaling."""
    
    def __init__(
        self,
        factory: Callable[[], T],
        config: PoolConfig,
        health_check: Optional[Callable[[T], bool]] = None,
        cleanup: Optional[Callable[[T], None]] = None
    ):
        self.factory = factory
        self.config = config
        self.health_check = health_check
        self.cleanup = cleanup
        
        self._pool: deque = deque()
        self._in_use: set = set()
        self._stats = PoolStats(max_size=config.max_size)
        self._lock = asyncio.Lock()
        self._cleanup_task: Optional[asyncio.Task] = None
        self._health_check_task: Optional[asyncio.Task] = None
        self._running = False
        
        # Initialize pool
        self._initialize_pool()
    
    def _initialize_pool(self):
        """Initialize the pool with minimum objects."""
        for _ in range(self.config.min_size):
            obj = self.factory()
            self._pool.append(obj)
            self._stats.total_created += 1
            self._stats.current_size += 1
    
    async def start(self):
        """Start pool management tasks."""
        if self._running:
            return
        
        self._running = True
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        logger.info(f"AsyncPool started with {self.config.min_size} initial objects")
    
    async def stop(self):
        """Stop pool management tasks."""
        self._running = False
        
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        
        # Cleanup all objects
        async with self._lock:
            while self._pool:
                obj = self._pool.popleft()
                if self.cleanup:
                    self.cleanup(obj)
        
        logger.info("AsyncPool stopped")
    
    async def acquire(self) -> T:
        """Acquire an object from the pool."""
        start_time = time.time()
        
        async with self._lock:
            # Try to get from pool
            if self._pool:
                obj = self._pool.popleft()
                self._in_use.add(obj)
                self._stats.total_acquired += 1
                self._stats.last_activity = time.time()
                return obj
            
            # Create new object if under limit
            if self._stats.current_size < self.config.max_size:
                obj = self.factory()
                self._in_use.add(obj)
                self._stats.total_created += 1
                self._stats.current_size += 1
                self._stats.total_acquired += 1
                self._stats.last_activity = time.time()
                return obj
        
        # Wait for available object
        wait_start = time.time()
        while time.time() - wait_start < self.config.acquire_timeout:
            async with self._lock:
                if self._pool:
                    obj = self._pool.popleft()
                    self._in_use.add(obj)
                    self._stats.total_acquired += 1
                    self._stats.last_activity = time.time()
                    self._stats.wait_time += time.time() - start_time
                    return obj
            
            await asyncio.sleep(0.01)
        
        raise TimeoutError(f"Failed to acquire object within {self.config.acquire_timeout} seconds")
    
    async def release(self, obj: T):
        """Release an object back to the pool."""
        async with self._lock:
            if obj in self._in_use:
                self._in_use.remove(obj)
                
                # Check if object is healthy
                if self.health_check and not self.health_check(obj):
                    if self.cleanup:
                        self.cleanup(obj)
                    self._stats.current_size -= 1
                    return
                
                # Add back to pool if under limit
                if len(self._pool) < self.config.max_size:
                    self._pool.append(obj)
                    self._stats.total_released += 1
                    self._stats.last_activity = time.time()
                else:
                    # Pool is full, cleanup object
                    if self.cleanup:
                        self.cleanup(obj)
                    self._stats.current_size -= 1
    
    @asynccontextmanager
    async def get(self):
        """Context manager for acquiring and releasing objects."""
        obj = await self.acquire()
        try:
            yield obj
        finally:
            await self.release(obj)
    
    async def _cleanup_loop(self):
        """Cleanup idle objects."""
        while self._running:
            try:
                await asyncio.sleep(self.config.cleanup_interval)
                await self._cleanup_idle_objects()
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
    
    async def _cleanup_idle_objects(self):
        """Cleanup idle objects."""
        current_time = time.time()
        idle_threshold = current_time - self.config.max_idle_time
        
        async with self._lock:
            # Remove idle objects
            objects_to_remove = []
            for obj in list(self._pool):
                # Simple heuristic: if pool has been idle, remove excess objects
                if (self._stats.last_activity < idle_threshold and 
                    len(self._pool) > self.config.min_size):
                    objects_to_remove.append(obj)
            
            for obj in objects_to_remove:
                self._pool.remove(obj)
                if self.cleanup:
                    self.cleanup(obj)
                self._stats.current_size -= 1
            
            if objects_to_remove:
                logger.info(f"Cleaned up {len(objects_to_remove)} idle objects")
    
    async def _health_check_loop(self):
        """Health check loop for objects in pool."""
        while self._running:
            try:
                await asyncio.sleep(self.config.health_check_interval)
                await self._perform_health_checks()
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
    
    async def _perform_health_checks(self):
        """Perform health checks on pool objects."""
        if not self.health_check:
            return
        
        async with self._lock:
            unhealthy_objects = []
            for obj in list(self._pool):
                if not self.health_check(obj):
                    unhealthy_objects.append(obj)
            
            for obj in unhealthy_objects:
                self._pool.remove(obj)
                if self.cleanup:
                    self.cleanup(obj)
                self._stats.current_size -= 1
            
            if unhealthy_objects:
                logger.info(f"Removed {len(unhealthy_objects)} unhealthy objects")
    
    def get_stats(self) -> PoolStats:
        """Get pool statistics."""
        return PoolStats(
            total_created=self._stats.total_created,
            total_acquired=self._stats.total_acquired,
            total_released=self._stats.total_released,
            current_size=len(self._pool),
            max_size=self.config.max_size,
            wait_time=self._stats.wait_time,
            last_activity=self._stats.last_activity
        )
    
    def get_utilization(self) -> float:
        """Get pool utilization percentage."""
        if self.config.max_size == 0:
            return 0.0
        return (len(self._in_use) / self.config.max_size) * 100


class PoolManager:
    """Manager for multiple async pools."""
    
    def __init__(self):
        self.pools: Dict[str, AsyncPool] = {}
        self._lock = asyncio.Lock()
    
    async def create_pool(
        self,
        name: str,
        factory: Callable[[], T],
        config: PoolConfig,
        health_check: Optional[Callable[[T], bool]] = None,
        cleanup: Optional[Callable[[T], None]] = None
    ) -> AsyncPool[T]:
        """Create a new pool."""
        async with self._lock:
            if name in self.pools:
                raise ValueError(f"Pool {name} already exists")
            
            pool = AsyncPool(factory, config, health_check, cleanup)
            await pool.start()
            self.pools[name] = pool
            
            logger.info(f"Created pool {name} with config: {config}")
            return pool
    
    async def get_pool(self, name: str) -> Optional[AsyncPool]:
        """Get a pool by name."""
        async with self._lock:
            return self.pools.get(name)
    
    async def remove_pool(self, name: str):
        """Remove a pool."""
        async with self._lock:
            if name in self.pools:
                pool = self.pools[name]
                await pool.stop()
                del self.pools[name]
                logger.info(f"Removed pool {name}")
    
    async def get_all_stats(self) -> Dict[str, PoolStats]:
        """Get statistics for all pools."""
        async with self._lock:
            return {name: pool.get_stats() for name, pool in self.pools.items()}
    
    async def shutdown_all(self):
        """Shutdown all pools."""
        async with self._lock:
            for name, pool in self.pools.items():
                try:
                    await pool.stop()
                except Exception as e:
                    logger.error(f"Error stopping pool {name}: {e}")
            
            self.pools.clear()
            logger.info("All pools shutdown")


class OptimizedExecutor:
    """Optimized executor with dynamic scaling."""
    
    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers or min(32, (psutil.cpu_count() or 1) + 4)
        self._executor = None
        self._lock = asyncio.Lock()
        self._current_workers = 0
        self._task_queue = deque()
        self._active_tasks = set()
    
    async def submit(self, func: Callable, *args, **kwargs) -> asyncio.Future:
        """Submit a task to the executor."""
        if not self._executor:
            await self._ensure_executor()
        
        loop = asyncio.get_event_loop()
        future = loop.run_in_executor(self._executor, func, *args, **kwargs)
        self._active_tasks.add(future)
        
        # Cleanup completed tasks
        future.add_done_callback(self._active_tasks.discard)
        
        return future
    
    async def _ensure_executor(self):
        """Ensure executor is created."""
        async with self._lock:
            if not self._executor:
                self._executor = ThreadPoolExecutor(
                    max_workers=self.max_workers,
                    thread_name_prefix="optimized_executor"
                )
                self._current_workers = self.max_workers
    
    async def shutdown(self, wait: bool = True):
        """Shutdown the executor."""
        async with self._lock:
            if self._executor:
                self._executor.shutdown(wait=wait)
                self._executor = None
                self._current_workers = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get executor statistics."""
        return {
            "max_workers": self.max_workers,
            "current_workers": self._current_workers,
            "active_tasks": len(self._active_tasks),
            "queued_tasks": len(self._task_queue)
        }


# Global instances
_pool_manager: Optional[PoolManager] = None
_optimized_executor: Optional[OptimizedExecutor] = None


def get_pool_manager() -> PoolManager:
    """Get global pool manager instance."""
    global _pool_manager
    if _pool_manager is None:
        _pool_manager = PoolManager()
    return _pool_manager


def get_optimized_executor() -> OptimizedExecutor:
    """Get global optimized executor instance."""
    global _optimized_executor
    if _optimized_executor is None:
        _optimized_executor = OptimizedExecutor()
    return _optimized_executor


# Utility functions
async def create_connection_pool(
    name: str,
    factory: Callable,
    max_connections: int = 10,
    min_connections: int = 1
) -> AsyncPool:
    """Create a connection pool."""
    config = PoolConfig(
        min_size=min_connections,
        max_size=max_connections,
        max_idle_time=300.0,
        cleanup_interval=60.0
    )
    
    pool_manager = get_pool_manager()
    return await pool_manager.create_pool(name, factory, config)


async def create_worker_pool(
    name: str,
    factory: Callable,
    max_workers: int = 10,
    min_workers: int = 1
) -> AsyncPool:
    """Create a worker pool."""
    config = PoolConfig(
        min_size=min_workers,
        max_size=max_workers,
        max_idle_time=600.0,
        cleanup_interval=120.0
    )
    
    pool_manager = get_pool_manager()
    return await pool_manager.create_pool(name, factory, config)


async def shutdown_all_pools():
    """Shutdown all pools."""
    pool_manager = get_pool_manager()
    await pool_manager.shutdown_all()
    
    executor = get_optimized_executor()
    await executor.shutdown()


# Decorators for pool usage
def use_pool(pool_name: str):
    """Decorator to use a specific pool for function execution."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            pool_manager = get_pool_manager()
            pool = await pool_manager.get_pool(pool_name)
            
            if not pool:
                raise ValueError(f"Pool {pool_name} not found")
            
            async with pool.get() as obj:
                return await func(obj, *args, **kwargs)
        
        return async_wrapper
    
    return decorator


def use_executor(func: Callable) -> Callable:
    """Decorator to use optimized executor for function execution."""
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        executor = get_optimized_executor()
        return await executor.submit(func, *args, **kwargs)
    
    return async_wrapper


