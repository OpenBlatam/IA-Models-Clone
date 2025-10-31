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
import weakref
import threading
from typing import Dict, Any, List, Optional, Callable, TypeVar, Generic, Union, Set
from dataclasses import dataclass, field
from enum import Enum
from contextlib import asynccontextmanager
import logging
from functools import wraps
import gc
import psutil
import os
    import numba
    from numba import jit, njit
from typing import Any, List, Dict, Optional
"""
Lazy Loader for Instagram Captions API v14.0

Advanced lazy loading strategies:
- Resource loading on demand
- Memory-efficient loading
- Background preloading
- Dependency management
- Resource pooling
- Performance monitoring
"""


# Performance libraries
try:
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def jit(*args, **kwargs) -> Any:
        def decorator(func) -> Any:
            return func
        return decorator
    njit = jit

logger = logging.getLogger(__name__)

# Type variables for generic operations
T = TypeVar('T')
K = TypeVar('K')
V = TypeVar('V')


class LoadState(Enum):
    """Resource loading states"""
    UNLOADED = "unloaded"
    LOADING = "loading"
    LOADED = "loaded"
    ERROR = "error"
    UNLOADING = "unloading"


class LoadPriority(Enum):
    """Loading priorities"""
    CRITICAL = "critical"      # Load immediately
    HIGH = "high"              # Load in background
    NORMAL = "normal"          # Load when needed
    LOW = "low"                # Load during idle time
    BACKGROUND = "background"  # Load in background thread


@dataclass
class LoadConfig:
    """Configuration for lazy loading"""
    # Memory management
    max_memory_mb: int = 512
    memory_threshold: float = 0.8  # Unload when 80% memory used
    
    # Loading strategies
    enable_preloading: bool = True
    preload_threshold: int = 5  # Preload after N accesses
    background_loading: bool = True
    max_background_tasks: int = 10
    
    # Caching
    enable_caching: bool = True
    cache_size: int = 1000
    cache_ttl: int = 3600
    
    # Performance
    enable_monitoring: bool = True
    load_timeout: float = 30.0
    retry_attempts: int = 3
    
    # Resource management
    enable_pooling: bool = True
    pool_size: int = 50
    cleanup_interval: int = 300  # 5 minutes


@dataclass
class ResourceInfo:
    """Information about a lazy-loaded resource"""
    key: str
    state: LoadState = LoadState.UNLOADED
    priority: LoadPriority = LoadPriority.NORMAL
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    load_time: float = 0.0
    memory_usage: int = 0
    dependencies: Set[str] = field(default_factory=set)
    error_count: int = 0
    last_error: Optional[str] = None


class LazyLoader(Generic[T]):
    """Advanced lazy loader with memory management and optimization"""
    
    def __init__(self, config: LoadConfig):
        
    """__init__ function."""
self.config = config
        self.resources: Dict[str, T] = {}
        self.resource_info: Dict[str, ResourceInfo] = {}
        self.loading_tasks: Dict[str, asyncio.Task] = {}
        self.resource_pool: Dict[str, List[T]] = {}
        
        # Thread safety
        self._lock = threading.RLock()
        self._async_lock = asyncio.Lock()
        
        # Background tasks
        self._background_tasks: Set[asyncio.Task] = set()
        self._cleanup_task = None
        
        # Statistics
        self.stats = {
            "total_loads": 0,
            "cache_hits": 0,
            "background_loads": 0,
            "memory_unloads": 0,
            "errors": 0,
            "avg_load_time": 0.0
        }
        
        # Start background tasks
        self._start_background_tasks()
    
    def _start_background_tasks(self) -> Any:
        """Start background maintenance tasks"""
        if self.config.enable_monitoring:
            asyncio.create_task(self._monitor_memory())
        if self.config.enable_pooling:
            asyncio.create_task(self._cleanup_pool())
    
    async def get(
        self, 
        key: str, 
        loader_func: Callable[[], T],
        priority: LoadPriority = LoadPriority.NORMAL,
        dependencies: Optional[List[str]] = None
    ) -> T:
        """Get resource with lazy loading"""
        
        # Check if already loaded
        if key in self.resources:
            self._update_access(key)
            self.stats["cache_hits"] += 1
            return self.resources[key]
        
        # Check resource pool
        if self.config.enable_pooling and key in self.resource_pool:
            if self.resource_pool[key]:
                resource = self.resource_pool[key].pop()
                self.resources[key] = resource
                self._update_access(key)
                self.stats["cache_hits"] += 1
                return resource
        
        # Create resource info
        resource_info = ResourceInfo(
            key=key,
            priority=priority,
            dependencies=set(dependencies or [])
        )
        self.resource_info[key] = resource_info
        
        # Load based on priority
        if priority == LoadPriority.CRITICAL:
            return await self._load_immediately(key, loader_func)
        elif priority == LoadPriority.HIGH:
            return await self._load_background(key, loader_func)
        else:
            return await self._load_on_demand(key, loader_func)
    
    async def _load_immediately(self, key: str, loader_func: Callable[[], T]) -> T:
        """Load resource immediately"""
        async with self._async_lock:
            if key in self.resources:
                return self.resources[key]
            
            self.resource_info[key].state = LoadState.LOADING
            
            try:
                start_time = time.time()
                
                # Load dependencies first
                await self._load_dependencies(key)
                
                # Load the resource
                if asyncio.iscoroutinefunction(loader_func):
                    resource = await loader_func()
                else:
                    # Run sync function in thread pool
                    loop = asyncio.get_event_loop()
                    resource = await loop.run_in_executor(None, loader_func)
                
                # Store resource
                self.resources[key] = resource
                self.resource_info[key].state = LoadState.LOADED
                self.resource_info[key].load_time = time.time() - start_time
                self.resource_info[key].memory_usage = self._estimate_memory_usage(resource)
                
                # Update stats
                self.stats["total_loads"] += 1
                self.stats["avg_load_time"] = (
                    (self.stats["avg_load_time"] * (self.stats["total_loads"] - 1) + 
                     self.resource_info[key].load_time) / self.stats["total_loads"]
                )
                
                return resource
                
            except Exception as e:
                self.resource_info[key].state = LoadState.ERROR
                self.resource_info[key].error_count += 1
                self.resource_info[key].last_error = str(e)
                self.stats["errors"] += 1
                logger.error(f"Failed to load resource {key}: {e}")
                raise
    
    async def _load_background(self, key: str, loader_func: Callable[[], T]) -> T:
        """Load resource in background"""
        # Start background loading
        if key not in self.loading_tasks:
            self.loading_tasks[key] = asyncio.create_task(
                self._load_immediately(key, loader_func)
            )
            self.stats["background_loads"] += 1
        
        # Wait for loading to complete
        try:
            return await self.loading_tasks[key]
        finally:
            if key in self.loading_tasks:
                del self.loading_tasks[key]
    
    async def _load_on_demand(self, key: str, loader_func: Callable[[], T]) -> T:
        """Load resource on demand with optimization"""
        # Check if we should preload
        if (self.config.enable_preloading and 
            self.resource_info[key].access_count >= self.config.preload_threshold):
            # Start background preloading
            asyncio.create_task(self._load_immediately(key, loader_func))
        
        # Load immediately for now
        return await self._load_immediately(key, loader_func)
    
    async def _load_dependencies(self, key: str) -> None:
        """Load dependencies for a resource"""
        dependencies = self.resource_info[key].dependencies
        
        for dep_key in dependencies:
            if dep_key not in self.resources:
                # Load dependency recursively
                # This is a simplified version - in practice, you'd have dependency loaders
                logger.debug(f"Loading dependency {dep_key} for {key}")
    
    def _update_access(self, key: str) -> None:
        """Update access information for a resource"""
        if key in self.resource_info:
            self.resource_info[key].last_accessed = time.time()
            self.resource_info[key].access_count += 1
    
    def _estimate_memory_usage(self, resource: T) -> int:
        """Estimate memory usage of a resource"""
        try:
            # Simple estimation - can be improved based on resource type
            return len(str(resource)) * 2  # Rough estimate
        except Exception:
            return 1024  # Default estimate
    
    async def _monitor_memory(self) -> None:
        """Monitor memory usage and unload if necessary"""
        while True:
            await asyncio.sleep(30)  # Check every 30 seconds
            
            try:
                # Get memory usage
                process = psutil.Process()
                memory_percent = process.memory_percent()
                
                if memory_percent > self.config.memory_threshold * 100:
                    await self._unload_least_used()
                    
            except Exception as e:
                logger.error(f"Memory monitoring error: {e}")
    
    async def _unload_least_used(self) -> None:
        """Unload least used resources to free memory"""
        async with self._async_lock:
            if not self.resources:
                return
            
            # Calculate usage scores
            current_time = time.time()
            scores = {}
            
            for key, info in self.resource_info.items():
                if key in self.resources:
                    age = current_time - info.last_accessed
                    frequency = info.access_count
                    memory = info.memory_usage
                    
                    # Score based on access frequency, recency, and memory usage
                    scores[key] = (frequency / (age + 1)) / (memory + 1)
            
            # Unload lowest scoring resources
            sorted_keys = sorted(scores.keys(), key=lambda k: scores[k])
            resources_to_unload = len(self.resources) // 4  # Unload 25%
            
            for key in sorted_keys[:resources_to_unload]:
                await self._unload_resource(key)
                self.stats["memory_unloads"] += 1
    
    async def _unload_resource(self, key: str) -> None:
        """Unload a specific resource"""
        if key not in self.resources:
            return
        
        self.resource_info[key].state = LoadState.UNLOADING
        
        # Move to pool if pooling is enabled
        if self.config.enable_pooling:
            if key not in self.resource_pool:
                self.resource_pool[key] = []
            
            if len(self.resource_pool[key]) < self.config.pool_size:
                self.resource_pool[key].append(self.resources[key])
        
        # Remove from active resources
        del self.resources[key]
        self.resource_info[key].state = LoadState.UNLOADED
        
        # Force garbage collection
        gc.collect()
    
    async def _cleanup_pool(self) -> None:
        """Cleanup resource pool periodically"""
        while True:
            await asyncio.sleep(self.config.cleanup_interval)
            
            try:
                async with self._async_lock:
                    for key, pool in self.resource_pool.items():
                        # Keep only recent items
                        if len(pool) > self.config.pool_size // 2:
                            # Remove oldest items
                            items_to_remove = len(pool) - self.config.pool_size // 2
                            self.resource_pool[key] = pool[items_to_remove:]
                    
                    # Force garbage collection
                    gc.collect()
                    
            except Exception as e:
                logger.error(f"Pool cleanup error: {e}")
    
    async def preload(self, keys: List[str], loader_funcs: Dict[str, Callable[[], T]]) -> None:
        """Preload multiple resources"""
        tasks = []
        
        for key in keys:
            if key in loader_funcs:
                task = asyncio.create_task(
                    self._load_immediately(key, loader_funcs[key])
                )
                tasks.append(task)
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def unload(self, key: str) -> None:
        """Manually unload a resource"""
        await self._unload_resource(key)
    
    async def clear(self) -> None:
        """Clear all resources"""
        async with self._async_lock:
            self.resources.clear()
            self.resource_pool.clear()
            self.loading_tasks.clear()
            
            # Cancel background tasks
            for task in self._background_tasks:
                task.cancel()
            self._background_tasks.clear()
            
            # Force garbage collection
            gc.collect()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get loader statistics"""
        total_memory = sum(
            info.memory_usage for info in self.resource_info.values()
            if info.state == LoadState.LOADED
        )
        
        return {
            "total_resources": len(self.resources),
            "total_info": len(self.resource_info),
            "pooled_resources": sum(len(pool) for pool in self.resource_pool.values()),
            "loading_tasks": len(self.loading_tasks),
            "background_tasks": len(self._background_tasks),
            "total_memory_mb": total_memory / (1024 * 1024),
            "stats": self.stats
        }


class ResourcePool:
    """Resource pool for efficient resource management"""
    
    def __init__(self, max_size: int = 100):
        
    """__init__ function."""
self.max_size = max_size
        self.pool: Dict[str, List[Any]] = {}
        self._lock = asyncio.Lock()
    
    async def get(self, key: str, creator_func: Callable[[], Any]) -> Optional[Dict[str, Any]]:
        """Get resource from pool or create new one"""
        async with self._lock:
            if key in self.pool and self.pool[key]:
                return self.pool[key].pop()
            else:
                return creator_func()
    
    async def put(self, key: str, resource: Any) -> None:
        """Return resource to pool"""
        async with self._lock:
            if key not in self.pool:
                self.pool[key] = []
            
            if len(self.pool[key]) < self.max_size:
                self.pool[key].append(resource)
    
    async def clear(self) -> None:
        """Clear the pool"""
        async with self._lock:
            self.pool.clear()


# Decorators for lazy loading
def lazy_load(priority: LoadPriority = LoadPriority.NORMAL):
    """Lazy load decorator"""
    def decorator(func) -> Any:
        loader = LazyLoader(LoadConfig())
        
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            # Generate key from function name and arguments
            key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"
            
            # Define loader function
            def loader_func():
                
    """loader_func function."""
if asyncio.iscoroutinefunction(func):
                    return asyncio.create_task(func(*args, **kwargs))
                else:
                    return func(*args, **kwargs)
            
            return await loader.get(key, loader_func, priority)
        
        return wrapper
    return decorator


def background_load(priority: LoadPriority = LoadPriority.BACKGROUND):
    """Background load decorator"""
    def decorator(func) -> Any:
        loader = LazyLoader(LoadConfig(background_loading=True))
        
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"
            
            def loader_func():
                
    """loader_func function."""
if asyncio.iscoroutinefunction(func):
                    return asyncio.create_task(func(*args, **kwargs))
                else:
                    return func(*args, **kwargs)
            
            return await loader.get(key, loader_func, priority)
        
        return wrapper
    return decorator


# Global instances
lazy_loader = LazyLoader(LoadConfig())
resource_pool = ResourcePool()


# Utility functions
async def preload_resources(resources: Dict[str, Callable[[], T]]) -> None:
    """Preload multiple resources"""
    await lazy_loader.preload(list(resources.keys()), resources)


async def get_memory_usage() -> Dict[str, float]:
    """Get current memory usage"""
    process = psutil.Process()
    memory_info = process.memory_info()
    
    return {
        "rss_mb": memory_info.rss / (1024 * 1024),
        "vms_mb": memory_info.vms / (1024 * 1024),
        "percent": process.memory_percent()
    }


async def optimize_memory() -> None:
    """Optimize memory usage"""
    # Force garbage collection
    gc.collect()
    
    # Unload least used resources
    await lazy_loader._unload_least_used()


# Context managers
@asynccontextmanager
async def resource_context(resource_key: str, loader_func: Callable[[], T]):
    """Context manager for resource management"""
    resource = await lazy_loader.get(resource_key, loader_func)
    try:
        yield resource
    finally:
        # Resource is automatically managed by lazy loader
        pass


@asynccontextmanager
async def pool_context(resource_key: str, creator_func: Callable[[], T]):
    """Context manager for resource pooling"""
    resource = await resource_pool.get(resource_key, creator_func)
    try:
        yield resource
    finally:
        await resource_pool.put(resource_key, resource) 