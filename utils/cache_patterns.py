from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import asyncio
import time
import logging
from typing import Any, Optional, Dict, List, Callable, Awaitable, Union, Tuple
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import hashlib
import orjson
from pydantic import BaseModel, Field
import structlog
from .advanced_caching_system import (
from typing import Any, List, Dict, Optional
"""
🎯 Cache Patterns and Strategies
================================

Specialized cache patterns and strategies for different data types:
- Cache-Aside Pattern
- Write-Through Pattern
- Write-Behind Pattern
- Cache-As-SoR Pattern
- Cache-Only Pattern
- Cache-With-Expiry Pattern
- Cache-With-Invalidation Pattern
- Cache-With-Prefetch Pattern
"""



    AdvancedCachingSystem, CacheConfig, CacheType, CacheLevel, 
    cache_result, static_cache, dynamic_cache
)

logger = structlog.get_logger(__name__)

class CachePattern(Enum):
    """Cache patterns"""
    CACHE_ASIDE = "cache_aside"
    WRITE_THROUGH = "write_through"
    WRITE_BEHIND = "write_behind"
    CACHE_AS_SOR = "cache_as_sor"
    CACHE_ONLY = "cache_only"
    CACHE_WITH_EXPIRY = "cache_with_expiry"
    CACHE_WITH_INVALIDATION = "cache_with_invalidation"
    CACHE_WITH_PREFETCH = "cache_with_prefetch"

@dataclass
class CachePatternConfig:
    """Configuration for cache patterns"""
    pattern: CachePattern
    ttl: int = 3600
    cache_type: CacheType = CacheType.DYNAMIC
    levels: List[CacheLevel] = None
    invalidation_keys: List[str] = None
    prefetch_threshold: int = 5
    write_buffer_size: int = 100
    write_buffer_timeout: float = 30.0

class CacheAsidePattern:
    """
    Cache-Aside Pattern: Application manages cache explicitly.
    Data is loaded into cache on demand and invalidated when updated.
    """
    
    def __init__(self, cache_system: AdvancedCachingSystem, config: CachePatternConfig):
        
    """__init__ function."""
self.cache_system = cache_system
        self.config = config
        self.access_counts = defaultdict(int)
    
    async def get(self, key: str, loader_func: Callable, *args, **kwargs) -> Optional[Dict[str, Any]]:
        """Get data with cache-aside pattern."""
        # Try cache first
        cached_value = await self.cache_system.get(key, self.config.cache_type)
        if cached_value is not None:
            self.access_counts[key] += 1
            return cached_value
        
        # Load from source if not in cache
        value = await loader_func(*args, **kwargs)
        if value is not None:
            await self.cache_system.set(
                key, value, self.config.cache_type, self.config.ttl, self.config.levels
            )
        
        return value
    
    async def set(self, key: str, value: Any, invalidate_cache: bool = True) -> None:
        """Set data and optionally invalidate cache."""
        # Update source
        # await self.update_source(key, value)  # Implement based on your data source
        
        # Invalidate cache
        if invalidate_cache:
            await self.cache_system.delete(key)
    
    async def invalidate(self, key: str) -> None:
        """Invalidate cache entry."""
        await self.cache_system.delete(key)
    
    def get_access_stats(self) -> Dict[str, int]:
        """Get access statistics."""
        return dict(self.access_counts)

class WriteThroughPattern:
    """
    Write-Through Pattern: Data is written to cache and source simultaneously.
    Ensures consistency but may impact write performance.
    """
    
    def __init__(self, cache_system: AdvancedCachingSystem, config: CachePatternConfig):
        
    """__init__ function."""
self.cache_system = cache_system
        self.config = config
    
    async def set(self, key: str, value: Any, writer_func: Callable, *args, **kwargs) -> None:
        """Set data with write-through pattern."""
        # Write to source first
        await writer_func(key, value, *args, **kwargs)
        
        # Write to cache
        await self.cache_system.set(
            key, value, self.config.cache_type, self.config.ttl, self.config.levels
        )
    
    async def get(self, key: str, loader_func: Callable, *args, **kwargs) -> Optional[Dict[str, Any]]:
        """Get data (same as cache-aside for reads)."""
        cached_value = await self.cache_system.get(key, self.config.cache_type)
        if cached_value is not None:
            return cached_value
        
        # Load from source if not in cache
        value = await loader_func(*args, **kwargs)
        if value is not None:
            await self.cache_system.set(
                key, value, self.config.cache_type, self.config.ttl, self.config.levels
            )
        
        return value

class WriteBehindPattern:
    """
    Write-Behind Pattern: Data is written to cache immediately and to source asynchronously.
    Improves write performance but may lose data if cache fails.
    """
    
    def __init__(self, cache_system: AdvancedCachingSystem, config: CachePatternConfig):
        
    """__init__ function."""
self.cache_system = cache_system
        self.config = config
        self.write_buffer = deque()
        self.write_task = None
        self._lock = asyncio.Lock()
    
    async def set(self, key: str, value: Any, writer_func: Callable, *args, **kwargs) -> None:
        """Set data with write-behind pattern."""
        # Write to cache immediately
        await self.cache_system.set(
            key, value, self.config.cache_type, self.config.ttl, self.config.levels
        )
        
        # Add to write buffer
        async with self._lock:
            self.write_buffer.append({
                'key': key,
                'value': value,
                'writer_func': writer_func,
                'args': args,
                'kwargs': kwargs,
                'timestamp': time.time()
            })
            
            # Start write task if not running
            if self.write_task is None or self.write_task.done():
                self.write_task = asyncio.create_task(self._process_write_buffer())
    
    async def _process_write_buffer(self) -> None:
        """Process write buffer asynchronously."""
        while True:
            try:
                # Wait for buffer to fill or timeout
                await asyncio.sleep(self.config.write_buffer_timeout)
                
                async with self._lock:
                    if not self.write_buffer:
                        break
                    
                    # Process all items in buffer
                    items_to_process = list(self.write_buffer)
                    self.write_buffer.clear()
                
                # Write items to source
                for item in items_to_process:
                    try:
                        await item['writer_func'](
                            item['key'], item['value'], 
                            *item['args'], **item['kwargs']
                        )
                    except Exception as e:
                        logger.error(f"Write-behind error for key {item['key']}: {e}")
                
            except Exception as e:
                logger.error(f"Write-behind processing error: {e}")
                break
    
    async def get(self, key: str, loader_func: Callable, *args, **kwargs) -> Optional[Dict[str, Any]]:
        """Get data (same as cache-aside for reads)."""
        cached_value = await self.cache_system.get(key, self.config.cache_type)
        if cached_value is not None:
            return cached_value
        
        # Load from source if not in cache
        value = await loader_func(*args, **kwargs)
        if value is not None:
            await self.cache_system.set(
                key, value, self.config.cache_type, self.config.ttl, self.config.levels
            )
        
        return value

class CacheAsSoRPattern:
    """
    Cache-As-Source-of-Truth Pattern: Cache is the primary data store.
    Source is only used for persistence and recovery.
    """
    
    def __init__(self, cache_system: AdvancedCachingSystem, config: CachePatternConfig):
        
    """__init__ function."""
self.cache_system = cache_system
        self.config = config
        self.persistent_keys = set()
    
    async def set(self, key: str, value: Any, persistent: bool = True) -> None:
        """Set data with cache-as-SoR pattern."""
        # Always write to cache
        await self.cache_system.set(
            key, value, self.config.cache_type, self.config.ttl, self.config.levels
        )
        
        # Mark as persistent if needed
        if persistent:
            self.persistent_keys.add(key)
    
    async def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get data from cache (primary source)."""
        return await self.cache_system.get(key, self.config.cache_type)
    
    async def persist_to_source(self, writer_func: Callable) -> None:
        """Persist persistent data to source."""
        for key in self.persistent_keys:
            value = await self.cache_system.get(key, self.config.cache_type)
            if value is not None:
                try:
                    await writer_func(key, value)
                except Exception as e:
                    logger.error(f"Persistence error for key {key}: {e}")
    
    async def load_from_source(self, loader_func: Callable) -> None:
        """Load data from source to cache."""
        try:
            data = await loader_func()
            for key, value in data.items():
                await self.cache_system.set(
                    key, value, self.config.cache_type, self.config.ttl, self.config.levels
                )
        except Exception as e:
            logger.error(f"Load from source error: {e}")

class CacheWithInvalidationPattern:
    """
    Cache-With-Invalidation Pattern: Automatic cache invalidation based on dependencies.
    """
    
    def __init__(self, cache_system: AdvancedCachingSystem, config: CachePatternConfig):
        
    """__init__ function."""
self.cache_system = cache_system
        self.config = config
        self.dependency_map = defaultdict(set)
        self.reverse_dependency_map = defaultdict(set)
    
    async def set_with_dependencies(self, key: str, value: Any, dependencies: List[str]) -> None:
        """Set data with dependencies for invalidation."""
        # Set data in cache
        await self.cache_system.set(
            key, value, self.config.cache_type, self.config.ttl, self.config.levels
        )
        
        # Update dependency maps
        self.dependency_map[key] = set(dependencies)
        for dep in dependencies:
            self.reverse_dependency_map[dep].add(key)
    
    async def invalidate_dependencies(self, dependency: str) -> None:
        """Invalidate all cache entries that depend on the given dependency."""
        affected_keys = self.reverse_dependency_map.get(dependency, set())
        
        for key in affected_keys:
            await self.cache_system.delete(key)
            # Remove from dependency maps
            self.dependency_map.pop(key, None)
        
        # Clear reverse dependency
        self.reverse_dependency_map.pop(dependency, None)
    
    async def get(self, key: str, loader_func: Callable, dependencies: List[str] = None, *args, **kwargs) -> Optional[Dict[str, Any]]:
        """Get data with dependency tracking."""
        cached_value = await self.cache_system.get(key, self.config.cache_type)
        if cached_value is not None:
            return cached_value
        
        # Load from source
        value = await loader_func(*args, **kwargs)
        if value is not None:
            await self.set_with_dependencies(key, value, dependencies or [])
        
        return value

class CacheWithPrefetchPattern:
    """
    Cache-With-Prefetch Pattern: Preload data based on access patterns.
    """
    
    def __init__(self, cache_system: AdvancedCachingSystem, config: CachePatternConfig):
        
    """__init__ function."""
self.cache_system = cache_system
        self.config = config
        self.access_patterns = defaultdict(int)
        self.prefetch_rules = {}
        self.prefetch_task = None
    
    async async def register_prefetch_rule(self, pattern: str, prefetch_func: Callable, 
                                   threshold: int = None) -> None:
        """Register a prefetch rule."""
        self.prefetch_rules[pattern] = {
            'func': prefetch_func,
            'threshold': threshold or self.config.prefetch_threshold
        }
    
    async def access(self, key: str, loader_func: Callable, *args, **kwargs) -> Any:
        """Access data and trigger prefetch if needed."""
        # Get data
        value = await self.cache_system.get(key, self.config.cache_type)
        if value is None:
            value = await loader_func(*args, **kwargs)
            if value is not None:
                await self.cache_system.set(
                    key, value, self.config.cache_type, self.config.ttl, self.config.levels
                )
        
        # Update access patterns
        self.access_patterns[key] += 1
        
        # Check for prefetch opportunities
        await self._check_prefetch(key)
        
        return value
    
    async async def _check_prefetch(self, key: str) -> None:
        """Check if prefetch should be triggered."""
        for pattern, rule in self.prefetch_rules.items():
            if pattern in key and self.access_patterns[key] >= rule['threshold']:
                # Trigger prefetch
                if self.prefetch_task is None or self.prefetch_task.done():
                    self.prefetch_task = asyncio.create_task(
                        self._execute_prefetch(pattern, rule['func'])
                    )
    
    async async def _execute_prefetch(self, pattern: str, prefetch_func: Callable) -> None:
        """Execute prefetch operation."""
        try:
            prefetch_data = await prefetch_func()
            for key, value in prefetch_data.items():
                await self.cache_system.set(
                    key, value, self.config.cache_type, self.config.ttl, self.config.levels
                )
            logger.info(f"Prefetch completed for pattern: {pattern}")
        except Exception as e:
            logger.error(f"Prefetch error for pattern {pattern}: {e}")

# Cache pattern decorators
def cache_aside(ttl: int = 3600, cache_type: CacheType = CacheType.DYNAMIC):
    """Decorator for cache-aside pattern."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(cache_system: AdvancedCachingSystem, key: str, *args, **kwargs):
            
    """wrapper function."""
pattern = CacheAsidePattern(
                cache_system, 
                CachePatternConfig(CachePattern.CACHE_ASIDE, ttl, cache_type)
            )
            return await pattern.get(key, func, *args, **kwargs)
        return wrapper
    return decorator

def write_through(ttl: int = 3600, cache_type: CacheType = CacheType.DYNAMIC):
    """Decorator for write-through pattern."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(cache_system: AdvancedCachingSystem, key: str, value: Any, *args, **kwargs):
            
    """wrapper function."""
pattern = WriteThroughPattern(
                cache_system, 
                CachePatternConfig(CachePattern.WRITE_THROUGH, ttl, cache_type)
            )
            return await pattern.set(key, value, func, *args, **kwargs)
        return wrapper
    return decorator

def cache_with_invalidation(ttl: int = 3600, cache_type: CacheType = CacheType.DYNAMIC):
    """Decorator for cache-with-invalidation pattern."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(cache_system: AdvancedCachingSystem, key: str, dependencies: List[str] = None, *args, **kwargs):
            
    """wrapper function."""
pattern = CacheWithInvalidationPattern(
                cache_system, 
                CachePatternConfig(CachePattern.CACHE_WITH_INVALIDATION, ttl, cache_type)
            )
            return await pattern.get(key, func, dependencies, *args, **kwargs)
        return wrapper
    return decorator

# Example usage
async def example_cache_patterns():
    """Example usage of cache patterns."""
    
    # Initialize cache system
    config = CacheConfig()
    cache_system = AdvancedCachingSystem(config)
    await cache_system.initialize()
    
    try:
        # Cache-Aside Pattern
        cache_aside_pattern = CacheAsidePattern(
            cache_system, 
            CachePatternConfig(CachePattern.CACHE_ASIDE, ttl=3600)
        )
        
        async def load_user_data(user_id: int):
            
    """load_user_data function."""
# Simulate database load
            await asyncio.sleep(0.1)
            return {"id": user_id, "name": f"User {user_id}"}
        
        # Get user data with cache-aside
        user_data = await cache_aside_pattern.get("user:123", load_user_data, 123)
        logger.info(f"User data: {user_data}")
        
        # Write-Through Pattern
        write_through_pattern = WriteThroughPattern(
            cache_system, 
            CachePatternConfig(CachePattern.WRITE_THROUGH, ttl=3600)
        )
        
        async def save_user_data(key: str, value: dict):
            
    """save_user_data function."""
# Simulate database save
            await asyncio.sleep(0.1)
            logger.info(f"Saved to database: {key} = {value}")
        
        # Save user data with write-through
        await write_through_pattern.set("user:123", {"id": 123, "name": "John"}, save_user_data)
        
        # Cache-With-Invalidation Pattern
        invalidation_pattern = CacheWithInvalidationPattern(
            cache_system, 
            CachePatternConfig(CachePattern.CACHE_WITH_INVALIDATION, ttl=3600)
        )
        
        # Set data with dependencies
        await invalidation_pattern.set_with_dependencies(
            "user_profile:123", 
            {"id": 123, "name": "John", "email": "john@example.com"},
            ["user:123", "profile:123"]
        )
        
        # Invalidate dependencies
        await invalidation_pattern.invalidate_dependencies("user:123")
        
        # Cache-With-Prefetch Pattern
        prefetch_pattern = CacheWithPrefetchPattern(
            cache_system, 
            CachePatternConfig(CachePattern.CACHE_WITH_PREFETCH, ttl=3600)
        )
        
        async def prefetch_user_data():
            
    """prefetch_user_data function."""
# Simulate prefetching related data
            return {
                "user:124": {"id": 124, "name": "Jane"},
                "user:125": {"id": 125, "name": "Bob"}
            }
        
        # Register prefetch rule
        await prefetch_pattern.register_prefetch_rule("user:", prefetch_user_data, threshold=3)
        
        # Access data multiple times to trigger prefetch
        for i in range(5):
            await prefetch_pattern.access("user:123", load_user_data, 123)
        
        # Get comprehensive statistics
        stats = cache_system.get_comprehensive_stats()
        logger.info(f"Cache statistics: {stats}")
        
    except Exception as e:
        logger.error(f"Cache pattern error: {e}")
    
    finally:
        await cache_system.shutdown()

match __name__:
    case "__main__":
    asyncio.run(example_cache_patterns()) 