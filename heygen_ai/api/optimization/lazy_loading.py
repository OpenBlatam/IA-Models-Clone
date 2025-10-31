from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import asyncio
import weakref
from typing import (
from datetime import datetime, timezone, timedelta
from contextlib import asynccontextmanager
import structlog
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
from typing import Any, List, Dict, Optional
import logging
#!/usr/bin/env python3
"""
Lazy Loading Strategies for HeyGen AI API
Efficient resource management with lazy initialization and background loading.
"""

    Dict, List, Any, Optional, Union, Callable, Awaitable,
    TypeVar, Generic, Tuple, Set, Protocol
)

logger = structlog.get_logger()

T = TypeVar('T')
R = TypeVar('R')

# =============================================================================
# Lazy Loading Types
# =============================================================================

class LoadingState(Enum):
    """Loading states for lazy loaded resources."""
    UNLOADED = "unloaded"
    LOADING = "loading"
    LOADED = "loaded"
    ERROR = "error"

@dataclass
class LoadingMetadata:
    """Metadata for lazy loading operations."""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    loaded_at: Optional[datetime] = None
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    load_duration_ms: Optional[float] = None
    error_count: int = 0
    last_error: Optional[str] = None
    size_bytes: int = 0

# =============================================================================
# Base Lazy Loader
# =============================================================================

class BaseLazyLoader(ABC, Generic[T]):
    """Base class for lazy loading implementations."""
    
    def __init__(
        self,
        loader_func: Callable[[], Awaitable[T]],
        auto_reload: bool = False,
        reload_interval: Optional[float] = None,
        max_retries: int = 3
    ):
        
    """__init__ function."""
self.loader_func = loader_func
        self.auto_reload = auto_reload
        self.reload_interval = reload_interval
        self.max_retries = max_retries
        
        self._state = LoadingState.UNLOADED
        self._value: Optional[T] = None
        self._metadata = LoadingMetadata()
        self._loading_task: Optional[asyncio.Task] = None
        self._loading_lock = asyncio.Lock()
        self._reload_task: Optional[asyncio.Task] = None
    
    @property
    def state(self) -> LoadingState:
        """Get current loading state."""
        return self._state
    
    @property
    def metadata(self) -> LoadingMetadata:
        """Get loading metadata."""
        return self._metadata
    
    @property
    def is_loaded(self) -> bool:
        """Check if resource is loaded."""
        return self._state == LoadingState.LOADED
    
    @property
    def is_loading(self) -> bool:
        """Check if resource is currently loading."""
        return self._state == LoadingState.LOADING
    
    async def load(self) -> T:
        """Load the resource."""
        # Check if already loaded
        if self._state == LoadingState.LOADED:
            self._update_access_metadata()
            return self._value
        
        # Check if currently loading
        if self._state == LoadingState.LOADING and self._loading_task:
            return await self._loading_task
        
        # Start loading
        async with self._loading_lock:
            # Double-check after acquiring lock
            if self._state == LoadingState.LOADED:
                self._update_access_metadata()
                return self._value
            
            if self._state == LoadingState.LOADING and self._loading_task:
                return await self._loading_task
            
            # Start loading task
            self._state = LoadingState.LOADING
            self._loading_task = asyncio.create_task(self._load_with_retry())
            
            # Start auto-reload if enabled
            if self.auto_reload and self.reload_interval:
                self._start_auto_reload()
        
        return await self._loading_task
    
    async def _load_with_retry(self) -> T:
        """Load resource with retry logic."""
        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                start_time = datetime.now(timezone.utc)
                
                # Load resource
                value = await self.loader_func()
                
                # Update state and metadata
                self._value = value
                self._state = LoadingState.LOADED
                self._metadata.loaded_at = datetime.now(timezone.utc)
                self._metadata.load_duration_ms = (
                    self._metadata.loaded_at - start_time
                ).total_seconds() * 1000
                self._metadata.size_bytes = self._calculate_size(value)
                
                logger.info(
                    "Resource loaded successfully",
                    duration_ms=self._metadata.load_duration_ms,
                    size_bytes=self._metadata.size_bytes
                )
                
                return value
                
            except Exception as e:
                last_exception = e
                self._metadata.error_count += 1
                self._metadata.last_error = str(e)
                
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    logger.warning(
                        "Resource loading failed, retrying",
                        attempt=attempt + 1,
                        error=str(e)
                    )
        
        # All retries failed
        self._state = LoadingState.ERROR
        logger.error("Resource loading failed after all retries", error=str(last_exception))
        raise last_exception
    
    def _update_access_metadata(self) -> Any:
        """Update access metadata."""
        self._metadata.access_count += 1
        self._metadata.last_accessed = datetime.now(timezone.utc)
    
    def _calculate_size(self, value: T) -> int:
        """Calculate size of loaded value."""
        try:
            return len(str(value))
        except Exception:
            return 0
    
    def _start_auto_reload(self) -> Any:
        """Start auto-reload task."""
        if self._reload_task is None:
            self._reload_task = asyncio.create_task(self._auto_reload_loop())
    
    async def _auto_reload_loop(self) -> Any:
        """Auto-reload loop."""
        while self.auto_reload and self.reload_interval:
            try:
                await asyncio.sleep(self.reload_interval)
                
                # Reload if loaded
                if self._state == LoadingState.LOADED:
                    await self.reload()
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Auto-reload error", error=str(e))
    
    async def reload(self) -> T:
        """Force reload of the resource."""
        # Reset state
        self._state = LoadingState.UNLOADED
        self._value = None
        self._loading_task = None
        
        # Load again
        return await self.load()
    
    async def unload(self) -> Any:
        """Unload the resource."""
        self._state = LoadingState.UNLOADED
        self._value = None
        
        # Cancel loading task
        if self._loading_task:
            self._loading_task.cancel()
            self._loading_task = None
        
        # Cancel reload task
        if self._reload_task:
            self._reload_task.cancel()
            self._reload_task = None
    
    def get_value(self) -> Optional[T]:
        """Get current value without loading."""
        return self._value

# =============================================================================
# Lazy Loading Manager
# =============================================================================

class LazyLoadingManager:
    """Manager for multiple lazy loaded resources."""
    
    def __init__(self) -> Any:
        self._loaders: Dict[str, BaseLazyLoader] = {}
        self._metadata: Dict[str, LoadingMetadata] = {}
        self._cleanup_task: Optional[asyncio.Task] = None
    
    async def start(self) -> Any:
        """Start the lazy loading manager."""
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
    
    async def stop(self) -> Any:
        """Stop the lazy loading manager."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Unload all resources
        for loader in self._loaders.values():
            await loader.unload()
        
        self._loaders.clear()
        self._metadata.clear()
    
    async def _cleanup_loop(self) -> Any:
        """Background cleanup loop."""
        while True:
            try:
                await asyncio.sleep(300)  # Cleanup every 5 minutes
                await self._cleanup_unused_resources()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Lazy loading cleanup error", error=str(e))
    
    async def _cleanup_unused_resources(self) -> Any:
        """Cleanup unused resources."""
        current_time = datetime.now(timezone.utc)
        resources_to_remove = []
        
        for key, loader in self._loaders.items():
            metadata = loader.metadata
            
            # Remove resources not accessed in the last hour
            if (metadata.last_accessed and 
                current_time - metadata.last_accessed > timedelta(hours=1)):
                resources_to_remove.append(key)
        
        for key in resources_to_remove:
            await self.remove_resource(key)
            logger.info("Removed unused resource", key=key)
    
    def register_resource(
        self,
        key: str,
        loader_func: Callable[[], Awaitable[T]],
        auto_reload: bool = False,
        reload_interval: Optional[float] = None,
        max_retries: int = 3
    ) -> BaseLazyLoader[T]:
        """Register a new lazy loaded resource."""
        loader = BaseLazyLoader(
            loader_func=loader_func,
            auto_reload=auto_reload,
            reload_interval=reload_interval,
            max_retries=max_retries
        )
        
        self._loaders[key] = loader
        return loader
    
    async def get_resource(self, key: str) -> Optional[Dict[str, Any]]:
        """Get a resource, loading it if necessary."""
        if key not in self._loaders:
            raise KeyError(f"Resource '{key}' not registered")
        
        loader = self._loaders[key]
        return await loader.load()
    
    async def preload_resource(self, key: str) -> asyncio.Task:
        """Preload a resource in the background."""
        if key not in self._loaders:
            raise KeyError(f"Resource '{key}' not registered")
        
        loader = self._loaders[key]
        return asyncio.create_task(loader.load())
    
    async def reload_resource(self, key: str) -> Any:
        """Reload a specific resource."""
        if key not in self._loaders:
            raise KeyError(f"Resource '{key}' not registered")
        
        loader = self._loaders[key]
        return await loader.reload()
    
    async def remove_resource(self, key: str):
        """Remove a resource."""
        if key in self._loaders:
            await self._loaders[key].unload()
            del self._loaders[key]
    
    def get_resource_state(self, key: str) -> Optional[LoadingState]:
        """Get the state of a resource."""
        if key in self._loaders:
            return self._loaders[key].state
        return None
    
    def get_all_states(self) -> Dict[str, LoadingState]:
        """Get states of all resources."""
        return {key: loader.state for key, loader in self._loaders.items()}
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get loading statistics."""
        total_resources = len(self._loaders)
        loaded_resources = sum(1 for loader in self._loaders.values() if loader.is_loaded)
        loading_resources = sum(1 for loader in self._loaders.values() if loader.is_loading)
        error_resources = sum(1 for loader in self._loaders.values() if loader.state == LoadingState.ERROR)
        
        total_access_count = sum(loader.metadata.access_count for loader in self._loaders.values())
        total_size_bytes = sum(loader.metadata.size_bytes for loader in self._loaders.values())
        
        return {
            "total_resources": total_resources,
            "loaded_resources": loaded_resources,
            "loading_resources": loading_resources,
            "error_resources": error_resources,
            "total_access_count": total_access_count,
            "total_size_bytes": total_size_bytes,
            "average_access_count": total_access_count / total_resources if total_resources > 0 else 0,
        }

# =============================================================================
# Proxy Pattern Implementation
# =============================================================================

class LazyProxy:
    """Proxy for lazy loading with transparent access."""
    
    def __init__(
        self,
        loader_func: Callable[[], Awaitable[T]],
        auto_reload: bool = False,
        reload_interval: Optional[float] = None
    ):
        
    """__init__ function."""
self._loader = BaseLazyLoader(
            loader_func=loader_func,
            auto_reload=auto_reload,
            reload_interval=reload_interval
        )
        self._value: Optional[T] = None
        self._loaded = False
    
    async def _ensure_loaded(self) -> Any:
        """Ensure the value is loaded."""
        if not self._loaded:
            self._value = await self._loader.load()
            self._loaded = True
    
    def __getattr__(self, name: str):
        """Proxy attribute access."""
        if name.startswith('_'):
            return super().__getattr__(name)
        
        # Create async wrapper for attribute access
        async def async_getattr():
            
    """async_getattr function."""
await self._ensure_loaded()
            return getattr(self._value, name)
        
        return async_getattr()
    
    def __getitem__(self, key) -> Optional[Dict[str, Any]]:
        """Proxy item access."""
        async def async_getitem():
            
    """async_getitem function."""
await self._ensure_loaded()
            return self._value[key]
        
        return async_getitem()
    
    def __len__(self) -> Any:
        """Proxy length access."""
        if self._loaded:
            return len(self._value)
        return 0
    
    def __str__(self) -> Any:
        """Proxy string representation."""
        if self._loaded:
            return str(self._value)
        return f"<LazyProxy: {self._loader.state.value}>"
    
    def __repr__(self) -> Any:
        """Proxy representation."""
        if self._loaded:
            return repr(self._value)
        return f"<LazyProxy: {self._loader.state.value}>"

# =============================================================================
# Background Loading
# =============================================================================

class BackgroundLoader:
    """Background loading with priority and queuing."""
    
    def __init__(self, max_concurrent: int = 5):
        
    """__init__ function."""
self.max_concurrent = max_concurrent
        self._queue: asyncio.Queue = asyncio.Queue()
        self._workers: List[asyncio.Task] = []
        self._running = False
        self._stats = {
            "queued": 0,
            "completed": 0,
            "failed": 0,
            "in_progress": 0
        }
    
    async def start(self) -> Any:
        """Start background loading workers."""
        if self._running:
            return
        
        self._running = True
        
        # Start worker tasks
        for _ in range(self.max_concurrent):
            worker = asyncio.create_task(self._worker())
            self._workers.append(worker)
        
        logger.info("Background loader started", workers=self.max_concurrent)
    
    async def stop(self) -> Any:
        """Stop background loading workers."""
        if not self._running:
            return
        
        self._running = False
        
        # Cancel all workers
        for worker in self._workers:
            worker.cancel()
        
        # Wait for workers to finish
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()
        
        logger.info("Background loader stopped")
    
    async def _worker(self) -> Any:
        """Background worker task."""
        while self._running:
            try:
                # Get task from queue
                task = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                
                self._stats["in_progress"] += 1
                
                try:
                    # Execute task
                    await task()
                    self._stats["completed"] += 1
                except Exception as e:
                    self._stats["failed"] += 1
                    logger.error("Background task failed", error=str(e))
                finally:
                    self._stats["in_progress"] -= 1
                    self._queue.task_done()
                    
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Worker error", error=str(e))
    
    async def queue_task(self, task: Callable[[], Awaitable[Any]], priority: int = 0):
        """Queue a task for background execution."""
        await self._queue.put((priority, task))
        self._stats["queued"] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get background loading statistics."""
        return {
            **self._stats,
            "queue_size": self._queue.qsize(),
            "workers": len(self._workers),
            "running": self._running
        }

# =============================================================================
# Resource Pool with Lazy Loading
# =============================================================================

class LazyResourcePool:
    """Resource pool with lazy loading capabilities."""
    
    def __init__(
        self,
        factory_func: Callable[[], Awaitable[T]],
        max_size: int = 10,
        min_size: int = 2
    ):
        
    """__init__ function."""
self.factory_func = factory_func
        self.max_size = max_size
        self.min_size = min_size
        
        self._resources: List[T] = []
        self._available: asyncio.Queue = asyncio.Queue()
        self._in_use: Set[T] = set()
        self._initialization_task: Optional[asyncio.Task] = None
    
    async def start(self) -> Any:
        """Start the resource pool."""
        # Initialize minimum resources
        self._initialization_task = asyncio.create_task(self._initialize_pool())
    
    async def stop(self) -> Any:
        """Stop the resource pool."""
        if self._initialization_task:
            self._initialization_task.cancel()
            try:
                await self._initialization_task
            except asyncio.CancelledError:
                pass
        
        # Clear all resources
        self._resources.clear()
        self._in_use.clear()
        
        # Clear queue
        while not self._available.empty():
            try:
                self._available.get_nowait()
            except asyncio.QueueEmpty:
                break
    
    async def _initialize_pool(self) -> Any:
        """Initialize the resource pool with minimum resources."""
        for _ in range(self.min_size):
            try:
                resource = await self.factory_func()
                self._resources.append(resource)
                await self._available.put(resource)
            except Exception as e:
                logger.error("Failed to initialize resource", error=str(e))
    
    @asynccontextmanager
    async def get_resource(self) -> Optional[Dict[str, Any]]:
        """Get a resource from the pool."""
        # Try to get from available queue
        try:
            resource = self._available.get_nowait()
        except asyncio.QueueEmpty:
            # Create new resource if under max size
            if len(self._resources) < self.max_size:
                resource = await self.factory_func()
                self._resources.append(resource)
            else:
                # Wait for available resource
                resource = await self._available.get()
        
        self._in_use.add(resource)
        
        try:
            yield resource
        finally:
            self._in_use.discard(resource)
            await self._available.put(resource)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get resource pool statistics."""
        return {
            "total_resources": len(self._resources),
            "available_resources": self._available.qsize(),
            "in_use_resources": len(self._in_use),
            "max_size": self.max_size,
            "min_size": self.min_size
        }

# =============================================================================
# Usage Examples
# =============================================================================

async def example_lazy_loading():
    """Example of lazy loading usage."""
    
    # Create lazy loading manager
    manager = LazyLoadingManager()
    await manager.start()
    
    try:
        # Register resources
        async def load_user_data():
            
    """load_user_data function."""
await asyncio.sleep(0.1)  # Simulate loading
            return {"id": 1, "name": "John Doe", "email": "john@example.com"}
        
        async def load_video_data():
            
    """load_video_data function."""
await asyncio.sleep(0.2)  # Simulate loading
            return {"id": 1, "title": "Sample Video", "duration": 120}
        
        manager.register_resource("user_data", load_user_data)
        manager.register_resource("video_data", load_video_data, auto_reload=True, reload_interval=300)
        
        # Get resources (will be loaded on first access)
        user_data = await manager.get_resource("user_data")
        video_data = await manager.get_resource("video_data")
        
        print(f"User data: {user_data}")
        print(f"Video data: {video_data}")
        
        # Get statistics
        stats = manager.get_statistics()
        print(f"Statistics: {stats}")
        
    finally:
        await manager.stop()

async def example_background_loading():
    """Example of background loading usage."""
    
    # Create background loader
    loader = BackgroundLoader(max_concurrent=3)
    await loader.start()
    
    try:
        # Queue tasks
        async def task1():
            
    """task1 function."""
await asyncio.sleep(1)
            print("Task 1 completed")
        
        async def task2():
            
    """task2 function."""
await asyncio.sleep(2)
            print("Task 2 completed")
        
        await loader.queue_task(task1)
        await loader.queue_task(task2)
        
        # Wait for completion
        await asyncio.sleep(3)
        
        # Get statistics
        stats = loader.get_stats()
        print(f"Background loader stats: {stats}")
        
    finally:
        await loader.stop()

# =============================================================================
# Export
# =============================================================================

__all__ = [
    "LoadingState",
    "LoadingMetadata",
    "BaseLazyLoader",
    "LazyLoadingManager",
    "LazyProxy",
    "BackgroundLoader",
    "LazyResourcePool",
    "example_lazy_loading",
    "example_background_loading",
] 