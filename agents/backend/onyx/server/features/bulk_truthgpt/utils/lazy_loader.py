"""
Lazy Loading System
==================

Advanced lazy loading system for optimizing resource usage.
"""

import asyncio
import logging
import time
import weakref
from typing import Dict, List, Optional, Any, Union, Callable, Tuple, Generic, TypeVar
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from functools import wraps
import inspect

logger = logging.getLogger(__name__)

T = TypeVar('T')

class LoadingStrategy(str, Enum):
    """Loading strategies."""
    ON_DEMAND = "on_demand"      # Load when first accessed
    PRELOAD = "preload"          # Load in background
    CACHED = "cached"            # Load and cache
    LAZY = "lazy"               # Load only when needed

class LoadingPriority(str, Enum):
    """Loading priorities."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class LoadingConfig:
    """Loading configuration."""
    max_cache_size: int = 1000
    cache_ttl: int = 3600
    preload_count: int = 10
    enable_compression: bool = True
    enable_monitoring: bool = True
    default_strategy: LoadingStrategy = LoadingStrategy.ON_DEMAND
    default_priority: LoadingPriority = LoadingPriority.NORMAL

@dataclass
class LoadingStats:
    """Loading statistics."""
    total_loads: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    preloads: int = 0
    evictions: int = 0
    load_time: float = 0.0
    average_load_time: float = 0.0

class LazyLoader(Generic[T]):
    """
    Generic lazy loader for any type.
    
    Features:
    - On-demand loading
    - Caching
    - Preloading
    - Priority-based loading
    - Memory optimization
    """
    
    def __init__(self, 
                 loader_func: Callable[[], T],
                 key: Optional[str] = None,
                 strategy: LoadingStrategy = LoadingStrategy.ON_DEMAND,
                 priority: LoadingPriority = LoadingPriority.NORMAL,
                 ttl: Optional[int] = None):
        
        self.loader_func = loader_func
        self.key = key or f"lazy_{id(self)}"
        self.strategy = strategy
        self.priority = priority
        self.ttl = ttl
        self._value = None
        self._loaded = False
        self._loading = False
        self._created_at = None
        self._last_accessed = None
        self._access_count = 0
        
    async def get(self) -> T:
        """Get the value, loading if necessary."""
        try:
            # Check if already loaded and not expired
            if self._loaded and self._value is not None:
                if self.ttl is None or (datetime.utcnow() - self._created_at).seconds < self.ttl:
                    self._last_accessed = datetime.utcnow()
                    self._access_count += 1
                    return self._value
            
            # Load if not loaded or expired
            if not self._loading:
                await self._load()
            
            return self._value
            
        except Exception as e:
            logger.error(f"Failed to get lazy value for {self.key}: {str(e)}")
            raise
    
    async def _load(self):
        """Load the value."""
        try:
            self._loading = True
            start_time = time.time()
            
            # Load the value
            if asyncio.iscoroutinefunction(self.loader_func):
                self._value = await self.loader_func()
            else:
                self._value = self.loader_func()
            
            self._loaded = True
            self._created_at = datetime.utcnow()
            self._last_accessed = datetime.utcnow()
            self._access_count = 1
            
            load_time = time.time() - start_time
            logger.debug(f"Loaded {self.key} in {load_time:.3f}s")
            
        except Exception as e:
            logger.error(f"Failed to load {self.key}: {str(e)}")
            raise
        finally:
            self._loading = False
    
    def is_loaded(self) -> bool:
        """Check if value is loaded."""
        return self._loaded and self._value is not None
    
    def is_expired(self) -> bool:
        """Check if value is expired."""
        if not self._loaded or self.ttl is None:
            return False
        
        return (datetime.utcnow() - self._created_at).seconds >= self.ttl
    
    def get_stats(self) -> Dict[str, Any]:
        """Get loading statistics."""
        return {
            'key': self.key,
            'loaded': self._loaded,
            'loading': self._loading,
            'expired': self.is_expired(),
            'access_count': self._access_count,
            'created_at': self._created_at.isoformat() if self._created_at else None,
            'last_accessed': self._last_accessed.isoformat() if self._last_accessed else None,
            'strategy': self.strategy.value,
            'priority': self.priority.value,
            'ttl': self.ttl
        }

class LazyCache:
    """
    Lazy loading cache with intelligent management.
    
    Features:
    - Automatic eviction
    - Priority-based loading
    - Memory optimization
    - Performance monitoring
    """
    
    def __init__(self, config: Optional[LoadingConfig] = None):
        self.config = config or LoadingConfig()
        self.loaders: Dict[str, LazyLoader] = {}
        self.stats = LoadingStats()
        self.lock = threading.RLock()
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        
    async def initialize(self):
        """Initialize lazy cache."""
        logger.info("Initializing Lazy Cache...")
        
        try:
            # Start monitoring
            if self.config.enable_monitoring:
                asyncio.create_task(self._monitor_cache())
            
            logger.info("Lazy Cache initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Lazy Cache: {str(e)}")
            raise
    
    async def _monitor_cache(self):
        """Monitor cache performance."""
        while True:
            try:
                await asyncio.sleep(30)  # Monitor every 30 seconds
                
                # Log cache stats
                with self.lock:
                    logger.info(f"Lazy cache stats: {self.get_stats()}")
                
            except Exception as e:
                logger.error(f"Error monitoring cache: {str(e)}")
    
    def create_loader(self, 
                     key: str,
                     loader_func: Callable[[], T],
                     strategy: LoadingStrategy = LoadingStrategy.ON_DEMAND,
                     priority: LoadingPriority = LoadingPriority.NORMAL,
                     ttl: Optional[int] = None) -> LazyLoader[T]:
        """Create a lazy loader."""
        with self.lock:
            loader = LazyLoader(
                loader_func=loader_func,
                key=key,
                strategy=strategy,
                priority=priority,
                ttl=ttl
            )
            
            self.loaders[key] = loader
            
            # Preload if strategy is PRELOAD
            if strategy == LoadingStrategy.PRELOAD:
                asyncio.create_task(self._preload(loader))
            
            return loader
    
    async def _preload(self, loader: LazyLoader):
        """Preload a lazy loader."""
        try:
            await loader.get()
            self.stats.preloads += 1
            logger.debug(f"Preloaded {loader.key}")
            
        except Exception as e:
            logger.error(f"Failed to preload {loader.key}: {str(e)}")
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self.lock:
            if key not in self.loaders:
                self.stats.cache_misses += 1
                return None
            
            loader = self.loaders[key]
            
            try:
                value = await loader.get()
                self.stats.cache_hits += 1
                return value
                
            except Exception as e:
                logger.error(f"Failed to get {key}: {str(e)}")
                self.stats.cache_misses += 1
                return None
    
    def remove(self, key: str) -> bool:
        """Remove loader from cache."""
        with self.lock:
            if key in self.loaders:
                del self.loaders[key]
                self.stats.evictions += 1
                return True
            return False
    
    def clear(self):
        """Clear all loaders."""
        with self.lock:
            self.loaders.clear()
            self.stats.evictions += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            return {
                'total_loaders': len(self.loaders),
                'total_loads': self.stats.total_loads,
                'cache_hits': self.stats.cache_hits,
                'cache_misses': self.stats.cache_misses,
                'preloads': self.stats.preloads,
                'evictions': self.stats.evictions,
                'load_time': self.stats.load_time,
                'average_load_time': self.stats.average_load_time,
                'hit_rate': self.stats.cache_hits / max(self.stats.cache_hits + self.stats.cache_misses, 1)
            }
    
    async def cleanup(self):
        """Cleanup lazy cache."""
        try:
            with self.lock:
                self.loaders.clear()
            
            self.thread_pool.shutdown(wait=True)
            logger.info("Lazy Cache cleaned up successfully")
            
        except Exception as e:
            logger.error(f"Failed to cleanup Lazy Cache: {str(e)}")

class LazyProperty:
    """
    Lazy property decorator.
    
    Makes properties load on first access and cache the result.
    """
    
    def __init__(self, func: Callable[[Any], T], ttl: Optional[int] = None):
        self.func = func
        self.ttl = ttl
        self.name = func.__name__
        self._cache = {}
        
    def __get__(self, instance, owner):
        if instance is None:
            return self
        
        # Check cache
        if instance in self._cache:
            cached_value, timestamp = self._cache[instance]
            
            # Check TTL
            if self.ttl is None or (time.time() - timestamp) < self.ttl:
                return cached_value
        
        # Load value
        value = self.func(instance)
        
        # Cache value
        self._cache[instance] = (value, time.time())
        
        return value
    
    def __set__(self, instance, value):
        # Update cache
        self._cache[instance] = (value, time.time())
    
    def __delete__(self, instance):
        # Remove from cache
        if instance in self._cache:
            del self._cache[instance]

class LazyMethod:
    """
    Lazy method decorator.
    
    Makes methods load on first call and cache the result.
    """
    
    def __init__(self, func: Callable, ttl: Optional[int] = None):
        self.func = func
        self.ttl = ttl
        self._cache = {}
        
    def __call__(self, *args, **kwargs):
        # Create cache key
        cache_key = (args, tuple(sorted(kwargs.items())))
        
        # Check cache
        if cache_key in self._cache:
            cached_value, timestamp = self._cache[cache_key]
            
            # Check TTL
            if self.ttl is None or (time.time() - timestamp) < self.ttl:
                return cached_value
        
        # Execute function
        value = self.func(*args, **kwargs)
        
        # Cache value
        self._cache[cache_key] = (value, time.time())
        
        return value

# Global lazy cache
lazy_cache = LazyCache()

# Decorators for lazy loading
def lazy_property(ttl: Optional[int] = None):
    """Decorator for lazy properties."""
    def decorator(func):
        return LazyProperty(func, ttl)
    return decorator

def lazy_method(ttl: Optional[int] = None):
    """Decorator for lazy methods."""
    def decorator(func):
        return LazyMethod(func, ttl)
    return decorator

def lazy_load(key: str, 
              strategy: LoadingStrategy = LoadingStrategy.ON_DEMAND,
              priority: LoadingPriority = LoadingPriority.NORMAL,
              ttl: Optional[int] = None):
    """Decorator for lazy loading functions."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Create loader
            loader = lazy_cache.create_loader(
                key=key,
                loader_func=lambda: func(*args, **kwargs),
                strategy=strategy,
                priority=priority,
                ttl=ttl
            )
            
            # Get value
            return await loader.get()
        
        return wrapper
    return decorator

def preload(key: str, ttl: Optional[int] = None):
    """Decorator for preloading functions."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Create loader with preload strategy
            loader = lazy_cache.create_loader(
                key=key,
                loader_func=lambda: func(*args, **kwargs),
                strategy=LoadingStrategy.PRELOAD,
                priority=LoadingPriority.HIGH,
                ttl=ttl
            )
            
            # Get value
            return await loader.get()
        
        return wrapper
    return decorator

def cached_load(key: str, ttl: Optional[int] = None):
    """Decorator for cached loading functions."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Create loader with cached strategy
            loader = lazy_cache.create_loader(
                key=key,
                loader_func=lambda: func(*args, **kwargs),
                strategy=LoadingStrategy.CACHED,
                priority=LoadingPriority.NORMAL,
                ttl=ttl
            )
            
            # Get value
            return await loader.get()
        
        return wrapper
    return decorator











