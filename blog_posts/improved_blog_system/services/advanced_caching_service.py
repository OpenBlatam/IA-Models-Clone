"""
Advanced Caching Service for multi-layer caching and cache optimization
"""

import asyncio
import json
import pickle
import hashlib
from typing import Dict, Any, List, Optional, Union, Callable
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, desc
import aioredis
from functools import wraps
import time

from ..models.database import CacheEntry, CacheStats
from ..core.exceptions import DatabaseError, ValidationError


class AdvancedCachingService:
    """Service for advanced caching operations."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
        self.redis_client = None
        self.memory_cache = {}
        self.cache_stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "deletes": 0,
            "expires": 0
        }
        self.cache_layers = ["memory", "redis", "database"]
        self.default_ttl = 3600  # 1 hour
        
    async def initialize_redis(self, redis_url: str = "redis://localhost:6379"):
        """Initialize Redis connection."""
        try:
            self.redis_client = await aioredis.from_url(redis_url)
            await self.redis_client.ping()
        except Exception as e:
            print(f"Warning: Could not connect to Redis: {e}")
            self.redis_client = None
    
    async def get(
        self,
        key: str,
        layer: Optional[str] = None,
        deserialize: bool = True
    ) -> Optional[Any]:
        """Get value from cache."""
        try:
            if layer:
                return await self._get_from_layer(key, layer, deserialize)
            
            # Try all layers in order
            for cache_layer in self.cache_layers:
                value = await self._get_from_layer(key, cache_layer, deserialize)
                if value is not None:
                    self.cache_stats["hits"] += 1
                    # Promote to higher layers if found in lower layer
                    if cache_layer != "memory":
                        await self._set_in_layer(key, value, "memory", ttl=300)
                    return value
            
            self.cache_stats["misses"] += 1
            return None
            
        except Exception as e:
            self.cache_stats["misses"] += 1
            return None
    
    async def _get_from_layer(self, key: str, layer: str, deserialize: bool = True) -> Optional[Any]:
        """Get value from specific cache layer."""
        try:
            if layer == "memory":
                return self.memory_cache.get(key)
            
            elif layer == "redis" and self.redis_client:
                value = await self.redis_client.get(key)
                if value:
                    if deserialize:
                        try:
                            return json.loads(value)
                        except json.JSONDecodeError:
                            return pickle.loads(value)
                    return value
            
            elif layer == "database":
                cache_entry_query = select(CacheEntry).where(
                    and_(
                        CacheEntry.key == key,
                        CacheEntry.expires_at > datetime.utcnow()
                    )
                )
                cache_entry_result = await self.session.execute(cache_entry_query)
                cache_entry = cache_entry_result.scalar_one_or_none()
                
                if cache_entry:
                    if deserialize:
                        try:
                            return json.loads(cache_entry.value)
                        except json.JSONDecodeError:
                            return pickle.loads(cache_entry.value)
                    return cache_entry.value
            
            return None
            
        except Exception as e:
            return None
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        layers: Optional[List[str]] = None,
        serialize: bool = True
    ) -> bool:
        """Set value in cache."""
        try:
            if ttl is None:
                ttl = self.default_ttl
            
            if layers is None:
                layers = self.cache_layers
            
            success = True
            for layer in layers:
                layer_success = await self._set_in_layer(key, value, layer, ttl, serialize)
                success = success and layer_success
            
            if success:
                self.cache_stats["sets"] += 1
            
            return success
            
        except Exception as e:
            return False
    
    async def _set_in_layer(
        self,
        key: str,
        value: Any,
        layer: str,
        ttl: int,
        serialize: bool = True
    ) -> bool:
        """Set value in specific cache layer."""
        try:
            if layer == "memory":
                self.memory_cache[key] = value
                # Set expiration for memory cache
                asyncio.create_task(self._expire_memory_key(key, ttl))
                return True
            
            elif layer == "redis" and self.redis_client:
                if serialize:
                    try:
                        serialized_value = json.dumps(value)
                    except (TypeError, ValueError):
                        serialized_value = pickle.dumps(value)
                else:
                    serialized_value = value
                
                await self.redis_client.setex(key, ttl, serialized_value)
                return True
            
            elif layer == "database":
                if serialize:
                    try:
                        serialized_value = json.dumps(value)
                    except (TypeError, ValueError):
                        serialized_value = pickle.dumps(value)
                else:
                    serialized_value = value
                
                expires_at = datetime.utcnow() + timedelta(seconds=ttl)
                
                # Check if entry exists
                existing_query = select(CacheEntry).where(CacheEntry.key == key)
                existing_result = await self.session.execute(existing_query)
                existing_entry = existing_result.scalar_one_or_none()
                
                if existing_entry:
                    existing_entry.value = serialized_value
                    existing_entry.expires_at = expires_at
                    existing_entry.updated_at = datetime.utcnow()
                else:
                    cache_entry = CacheEntry(
                        key=key,
                        value=serialized_value,
                        expires_at=expires_at,
                        created_at=datetime.utcnow()
                    )
                    self.session.add(cache_entry)
                
                await self.session.commit()
                return True
            
            return False
            
        except Exception as e:
            return False
    
    async def _expire_memory_key(self, key: str, ttl: int):
        """Expire memory cache key after TTL."""
        await asyncio.sleep(ttl)
        if key in self.memory_cache:
            del self.memory_cache[key]
            self.cache_stats["expires"] += 1
    
    async def delete(self, key: str, layers: Optional[List[str]] = None) -> bool:
        """Delete value from cache."""
        try:
            if layers is None:
                layers = self.cache_layers
            
            success = True
            for layer in layers:
                layer_success = await self._delete_from_layer(key, layer)
                success = success and layer_success
            
            if success:
                self.cache_stats["deletes"] += 1
            
            return success
            
        except Exception as e:
            return False
    
    async def _delete_from_layer(self, key: str, layer: str) -> bool:
        """Delete value from specific cache layer."""
        try:
            if layer == "memory":
                if key in self.memory_cache:
                    del self.memory_cache[key]
                return True
            
            elif layer == "redis" and self.redis_client:
                await self.redis_client.delete(key)
                return True
            
            elif layer == "database":
                delete_query = CacheEntry.__table__.delete().where(CacheEntry.key == key)
                await self.session.execute(delete_query)
                await self.session.commit()
                return True
            
            return False
            
        except Exception as e:
            return False
    
    async def exists(self, key: str, layer: Optional[str] = None) -> bool:
        """Check if key exists in cache."""
        try:
            if layer:
                return await self._exists_in_layer(key, layer)
            
            # Check all layers
            for cache_layer in self.cache_layers:
                if await self._exists_in_layer(key, cache_layer):
                    return True
            
            return False
            
        except Exception as e:
            return False
    
    async def _exists_in_layer(self, key: str, layer: str) -> bool:
        """Check if key exists in specific cache layer."""
        try:
            if layer == "memory":
                return key in self.memory_cache
            
            elif layer == "redis" and self.redis_client:
                return await self.redis_client.exists(key) > 0
            
            elif layer == "database":
                exists_query = select(func.count(CacheEntry.id)).where(
                    and_(
                        CacheEntry.key == key,
                        CacheEntry.expires_at > datetime.utcnow()
                    )
                )
                exists_result = await self.session.execute(exists_query)
                return exists_result.scalar() > 0
            
            return False
            
        except Exception as e:
            return False
    
    async def clear(self, pattern: Optional[str] = None, layer: Optional[str] = None) -> int:
        """Clear cache entries."""
        try:
            cleared_count = 0
            
            if layer:
                cleared_count = await self._clear_layer(pattern, layer)
            else:
                for cache_layer in self.cache_layers:
                    cleared_count += await self._clear_layer(pattern, cache_layer)
            
            return cleared_count
            
        except Exception as e:
            return 0
    
    async def _clear_layer(self, pattern: Optional[str], layer: str) -> int:
        """Clear cache entries from specific layer."""
        try:
            if layer == "memory":
                if pattern:
                    keys_to_delete = [key for key in self.memory_cache.keys() if pattern in key]
                    for key in keys_to_delete:
                        del self.memory_cache[key]
                    return len(keys_to_delete)
                else:
                    count = len(self.memory_cache)
                    self.memory_cache.clear()
                    return count
            
            elif layer == "redis" and self.redis_client:
                if pattern:
                    keys = await self.redis_client.keys(pattern)
                    if keys:
                        await self.redis_client.delete(*keys)
                    return len(keys)
                else:
                    await self.redis_client.flushdb()
                    return -1  # Unknown count
            
            elif layer == "database":
                if pattern:
                    delete_query = CacheEntry.__table__.delete().where(
                        CacheEntry.key.like(f"%{pattern}%")
                    )
                else:
                    delete_query = CacheEntry.__table__.delete()
                
                result = await self.session.execute(delete_query)
                await self.session.commit()
                return result.rowcount
            
            return 0
            
        except Exception as e:
            return 0
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        try:
            # Get memory cache stats
            memory_stats = {
                "size": len(self.memory_cache),
                "keys": list(self.memory_cache.keys())[:10]  # First 10 keys
            }
            
            # Get Redis stats
            redis_stats = {}
            if self.redis_client:
                try:
                    info = await self.redis_client.info()
                    redis_stats = {
                        "connected": True,
                        "used_memory": info.get("used_memory_human", "0B"),
                        "connected_clients": info.get("connected_clients", 0),
                        "total_commands_processed": info.get("total_commands_processed", 0)
                    }
                except Exception:
                    redis_stats = {"connected": False}
            
            # Get database cache stats
            db_cache_query = select(
                func.count(CacheEntry.id).label('total_entries'),
                func.count(func.case((CacheEntry.expires_at > datetime.utcnow(), 1))).label('active_entries')
            )
            db_cache_result = await self.session.execute(db_cache_query)
            db_cache_stats = db_cache_result.first()
            
            # Calculate hit rate
            total_requests = self.cache_stats["hits"] + self.cache_stats["misses"]
            hit_rate = (self.cache_stats["hits"] / total_requests * 100) if total_requests > 0 else 0
            
            return {
                "cache_stats": self.cache_stats,
                "hit_rate": hit_rate,
                "memory_cache": memory_stats,
                "redis_cache": redis_stats,
                "database_cache": {
                    "total_entries": db_cache_stats.total_entries or 0,
                    "active_entries": db_cache_stats.active_entries or 0
                },
                "cache_layers": self.cache_layers,
                "default_ttl": self.default_ttl
            }
            
        except Exception as e:
            raise DatabaseError(f"Failed to get cache stats: {str(e)}")
    
    def cache_result(
        self,
        ttl: int = 3600,
        key_prefix: str = "",
        layers: Optional[List[str]] = None,
        serialize: bool = True
    ):
        """Decorator to cache function results."""
        def decorator(func: Callable):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # Generate cache key
                cache_key = self._generate_cache_key(func.__name__, args, kwargs, key_prefix)
                
                # Try to get from cache
                cached_result = await self.get(cache_key, deserialize=serialize)
                if cached_result is not None:
                    return cached_result
                
                # Execute function
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                # Cache the result
                await self.set(cache_key, result, ttl=ttl, layers=layers, serialize=serialize)
                
                return result
            
            return wrapper
        return decorator
    
    def _generate_cache_key(
        self,
        func_name: str,
        args: tuple,
        kwargs: dict,
        prefix: str = ""
    ) -> str:
        """Generate cache key from function name and arguments."""
        key_data = {
            "func": func_name,
            "args": args,
            "kwargs": sorted(kwargs.items())
        }
        
        key_string = json.dumps(key_data, sort_keys=True, default=str)
        key_hash = hashlib.md5(key_string.encode()).hexdigest()
        
        return f"{prefix}{func_name}:{key_hash}" if prefix else f"{func_name}:{key_hash}"
    
    async def warm_cache(self, keys_and_values: Dict[str, Any], ttl: int = 3600) -> int:
        """Warm cache with multiple key-value pairs."""
        try:
            success_count = 0
            
            for key, value in keys_and_values.items():
                success = await self.set(key, value, ttl=ttl)
                if success:
                    success_count += 1
            
            return success_count
            
        except Exception as e:
            return 0
    
    async def get_or_set(
        self,
        key: str,
        value_func: Callable,
        ttl: int = 3600,
        layers: Optional[List[str]] = None
    ) -> Any:
        """Get value from cache or set it using a function."""
        try:
            # Try to get from cache
            cached_value = await self.get(key)
            if cached_value is not None:
                return cached_value
            
            # Execute function to get value
            if asyncio.iscoroutinefunction(value_func):
                value = await value_func()
            else:
                value = value_func()
            
            # Cache the value
            await self.set(key, value, ttl=ttl, layers=layers)
            
            return value
            
        except Exception as e:
            # If caching fails, still return the function result
            if asyncio.iscoroutinefunction(value_func):
                return await value_func()
            else:
                return value_func()
    
    async def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate cache entries matching a pattern."""
        try:
            return await self.clear(pattern=pattern)
        except Exception as e:
            return 0
    
    async def get_memory_usage(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        try:
            import sys
            
            memory_stats = {
                "cache_entries": len(self.memory_cache),
                "estimated_size_bytes": sys.getsizeof(self.memory_cache),
                "average_key_size": 0,
                "average_value_size": 0
            }
            
            if self.memory_cache:
                total_key_size = sum(sys.getsizeof(key) for key in self.memory_cache.keys())
                total_value_size = sum(sys.getsizeof(value) for value in self.memory_cache.values())
                
                memory_stats["average_key_size"] = total_key_size / len(self.memory_cache)
                memory_stats["average_value_size"] = total_value_size / len(self.memory_cache)
            
            return memory_stats
            
        except Exception as e:
            return {"error": str(e)}
    
    async def optimize_cache(self) -> Dict[str, Any]:
        """Optimize cache by cleaning expired entries and optimizing memory usage."""
        try:
            optimization_results = {
                "expired_entries_removed": 0,
                "memory_optimized": False,
                "database_cleaned": False
            }
            
            # Clean expired database entries
            expired_query = CacheEntry.__table__.delete().where(
                CacheEntry.expires_at <= datetime.utcnow()
            )
            expired_result = await self.session.execute(expired_query)
            await self.session.commit()
            optimization_results["expired_entries_removed"] = expired_result.rowcount
            optimization_results["database_cleaned"] = True
            
            # Optimize memory cache (remove oldest entries if too large)
            if len(self.memory_cache) > 1000:  # Arbitrary limit
                # Remove 20% of oldest entries
                keys_to_remove = list(self.memory_cache.keys())[:200]
                for key in keys_to_remove:
                    del self.memory_cache[key]
                optimization_results["memory_optimized"] = True
            
            return optimization_results
            
        except Exception as e:
            return {"error": str(e)}
    
    async def close(self):
        """Close cache connections."""
        try:
            if self.redis_client:
                await self.redis_client.close()
        except Exception as e:
            pass

























