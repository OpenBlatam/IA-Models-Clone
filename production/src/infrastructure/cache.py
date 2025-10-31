from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

# Constants
BUFFER_SIZE = 1024

import asyncio
import logging
import json
import hashlib
import time
from typing import Any, Optional, Dict, List, Union
from contextlib import asynccontextmanager
import redis.asyncio as redis
import orjson
import lz4.frame
import zstandard as zstd
from pydantic import BaseModel
from src.core.config import RedisSettings, CacheSettings
from src.core.exceptions import CacheException
from typing import Any, List, Dict, Optional
"""
⚡ Ultra-Optimized Cache Service
===============================

Production-grade caching with:
- Redis integration
- Compression
- Intelligent TTL
- Cache warming
- Performance monitoring
- Distributed locking
"""





class CacheServiceConfig(BaseModel):
    """Cache service configuration"""
    
    # Redis settings
    url: str = "redis://localhost:6379/0"
    password: Optional[str] = None
    db: int = 0
    max_connections: int = 20
    socket_timeout: int = 5
    socket_connect_timeout: int = 5
    
    # Cache settings
    enabled: bool = True
    default_ttl: int = 3600
    max_size: int = 1000
    compression: bool = True
    persistence: bool = False
    
    # Performance settings
    compression_threshold: int = 1024  # bytes
    batch_size: int = 100
    max_retries: int = 3
    
    class Config:
        env_prefix = "CACHE_"


class CacheService:
    """
    Ultra-optimized cache service with Redis integration,
    compression, and intelligent caching strategies.
    """
    
    def __init__(self, redis_settings: RedisSettings, cache_settings: CacheSettings):
        
    """__init__ function."""
self.config = CacheServiceConfig(
            url=redis_settings.URL,
            password=redis_settings.PASSWORD.get_secret_value() if redis_settings.PASSWORD else None,
            db=redis_settings.DB,
            max_connections=redis_settings.MAX_CONNECTIONS,
            socket_timeout=redis_settings.SOCKET_TIMEOUT,
            socket_connect_timeout=redis_settings.SOCKET_CONNECT_TIMEOUT,
            enabled=cache_settings.ENABLED,
            default_ttl=cache_settings.TTL,
            max_size=cache_settings.MAX_SIZE,
            compression=cache_settings.COMPRESSION,
            persistence=cache_settings.PERSISTENCE
        )
        
        self.logger = logging.getLogger(__name__)
        
        # Redis connection pool
        self.redis_pool = None
        self.redis_client = None
        
        # Performance metrics
        self.get_count = 0
        self.set_count = 0
        self.hit_count = 0
        self.miss_count = 0
        self.delete_count = 0
        self.compression_count = 0
        self.total_get_time = 0.0
        self.total_set_time = 0.0
        
        # Cache warming
        self.warmup_keys = set()
        self.warmup_task = None
        
        # Health status
        self.is_healthy = False
        self.last_health_check = None
        
        # Compression algorithms
        self.compressors = {
            'lz4': self._compress_lz4,
            'zstd': self._compress_zstd,
            'none': lambda x: x
        }
        
        self.decompressors = {
            'lz4': self._decompress_lz4,
            'zstd': self._decompress_zstd,
            'none': lambda x: x
        }
        
        self.logger.info("Cache Service initialized")
    
    async def initialize(self) -> Any:
        """Initialize cache service and Redis connection"""
        
        self.logger.info("Initializing Cache Service...")
        
        try:
            # Create Redis connection pool
            self.redis_pool = redis.ConnectionPool.from_url(
                self.config.url,
                password=self.config.password,
                db=self.config.db,
                max_connections=self.config.max_connections,
                socket_timeout=self.config.socket_timeout,
                socket_connect_timeout=self.config.socket_connect_timeout,
                decode_responses=False  # Keep as bytes for compression
            )
            
            # Create Redis client
            self.redis_client = redis.Redis(connection_pool=self.redis_pool)
            
            # Test connection
            await self._test_connection()
            
            # Initialize cache warming
            if self.config.enabled:
                await self._initialize_warmup()
            
            # Set health status
            self.is_healthy = True
            self.last_health_check = asyncio.get_event_loop().time()
            
            self.logger.info("Cache Service initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Cache Service: {e}")
            raise CacheException("initialization", reason=str(e))
    
    async def cleanup(self) -> Any:
        """Cleanup cache service resources"""
        
        self.logger.info("Cleaning up Cache Service...")
        
        try:
            # Stop warmup task
            if self.warmup_task and not self.warmup_task.done():
                self.warmup_task.cancel()
                try:
                    await self.warmup_task
                except asyncio.CancelledError:
                    pass
            
            # Close Redis client
            if self.redis_client:
                await self.redis_client.close()
            
            # Close connection pool
            if self.redis_pool:
                await self.redis_pool.disconnect()
            
            self.logger.info("Cache Service cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during cache cleanup: {e}")
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache with optimization"""
        
        if not self.config.enabled:
            return None
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Get raw data from Redis
            raw_data = await self.redis_client.get(key)
            
            if raw_data is None:
                self.miss_count += 1
                return None
            
            # Deserialize and decompress
            data = await self._deserialize_data(raw_data)
            
            self.hit_count += 1
            return data
            
        except Exception as e:
            self.logger.error(f"Cache get failed for key {key}: {e}")
            raise CacheException("get", key=key, reason=str(e))
            
        finally:
            self.get_count += 1
            self.total_get_time += asyncio.get_event_loop().time() - start_time
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache with optimization"""
        
        if not self.config.enabled:
            return False
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Serialize and compress data
            serialized_data = await self._serialize_data(value)
            
            # Set TTL
            if ttl is None:
                ttl = self.config.default_ttl
            
            # Store in Redis
            await self.redis_client.setex(key, ttl, serialized_data)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Cache set failed for key {key}: {e}")
            raise CacheException("set", key=key, reason=str(e))
            
        finally:
            self.set_count += 1
            self.total_set_time += asyncio.get_event_loop().time() - start_time
    
    async def delete(self, key: str) -> bool:
        """Delete value from cache"""
        
        if not self.config.enabled:
            return False
        
        try:
            result = await self.redis_client.delete(key)
            self.delete_count += 1
            return result > 0
            
        except Exception as e:
            self.logger.error(f"Cache delete failed for key {key}: {e}")
            raise CacheException("delete", key=key, reason=str(e))
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        
        if not self.config.enabled:
            return False
        
        try:
            return await self.redis_client.exists(key) > 0
            
        except Exception as e:
            self.logger.error(f"Cache exists check failed for key {key}: {e}")
            return False
    
    async def increment(self, key: str, amount: int = 1) -> int:
        """Increment counter in cache"""
        
        if not self.config.enabled:
            return 0
        
        try:
            return await self.redis_client.incrby(key, amount)
            
        except Exception as e:
            self.logger.error(f"Cache increment failed for key {key}: {e}")
            raise CacheException("increment", key=key, reason=str(e))
    
    async def expire(self, key: str, ttl: int) -> bool:
        """Set expiration for key"""
        
        if not self.config.enabled:
            return False
        
        try:
            return await self.redis_client.expire(key, ttl)
            
        except Exception as e:
            self.logger.error(f"Cache expire failed for key {key}: {e}")
            raise CacheException("expire", key=key, reason=str(e))
    
    async def clear_pattern(self, pattern: str) -> int:
        """Clear keys matching pattern"""
        
        if not self.config.enabled:
            return 0
        
        try:
            keys = await self.redis_client.keys(pattern)
            if keys:
                deleted = await self.redis_client.delete(*keys)
                self.delete_count += deleted
                return deleted
            return 0
            
        except Exception as e:
            self.logger.error(f"Cache clear pattern failed for {pattern}: {e}")
            raise CacheException("clear_pattern", pattern=pattern, reason=str(e))
    
    async def get_many(self, keys: List[str]) -> Dict[str, Any]:
        """Get multiple values from cache"""
        
        if not self.config.enabled:
            return {}
        
        try:
            # Use pipeline for better performance
            async with self.redis_client.pipeline() as pipe:
                for key in keys:
                    pipe.get(key)
                
                results = await pipe.execute()
            
            # Process results
            data = {}
            for key, result in zip(keys, results):
                if result is not None:
                    data[key] = await self._deserialize_data(result)
                    self.hit_count += 1
                else:
                    self.miss_count += 1
            
            self.get_count += len(keys)
            return data
            
        except Exception as e:
            self.logger.error(f"Cache get_many failed: {e}")
            raise CacheException("get_many", reason=str(e))
    
    async def set_many(self, data: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """Set multiple values in cache"""
        
        if not self.config.enabled:
            return False
        
        try:
            # Use pipeline for better performance
            async with self.redis_client.pipeline() as pipe:
                for key, value in data.items():
                    serialized_data = await self._serialize_data(value)
                    if ttl is None:
                        ttl = self.config.default_ttl
                    pipe.setex(key, ttl, serialized_data)
                
                await pipe.execute()
            
            self.set_count += len(data)
            return True
            
        except Exception as e:
            self.logger.error(f"Cache set_many failed: {e}")
            raise CacheException("set_many", reason=str(e))
    
    @asynccontextmanager
    async def lock(self, key: str, timeout: int = 10):
        """Distributed lock for cache operations"""
        
        lock_key = f"lock:{key}"
        lock_value = str(time.time())
        
        try:
            # Try to acquire lock
            acquired = await self.redis_client.set(
                lock_key, 
                lock_value.encode(), 
                ex=timeout, 
                nx=True
            )
            
            if not acquired:
                raise CacheException("lock acquisition", key=key, reason="Lock already held")
            
            yield
            
        finally:
            # Release lock (only if we own it)
            try:
                current_value = await self.redis_client.get(lock_key)
                if current_value and current_value.decode() == lock_value:
                    await self.redis_client.delete(lock_key)
            except Exception as e:
                self.logger.error(f"Failed to release lock {key}: {e}")
    
    async def warm_cache(self, warmup_data: Dict[str, Any], ttl: int = 3600):
        """Warm up cache with frequently accessed data"""
        
        if not self.config.enabled:
            return
        
        try:
            self.logger.info(f"Warming cache with {len(warmup_data)} items")
            
            # Set warmup data
            await self.set_many(warmup_data, ttl)
            
            # Track warmup keys
            self.warmup_keys.update(warmup_data.keys())
            
            self.logger.info("Cache warmup completed")
            
        except Exception as e:
            self.logger.error(f"Cache warmup failed: {e}")
            raise CacheException("warmup", reason=str(e))
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform cache health check"""
        
        try:
            start_time = asyncio.get_event_loop().time()
            
            # Test Redis connection
            await self.redis_client.ping()
            
            response_time = asyncio.get_event_loop().time() - start_time
            
            # Get Redis info
            info = await self.redis_client.info()
            
            # Update health status
            self.is_healthy = True
            self.last_health_check = asyncio.get_event_loop().time()
            
            return {
                "status": "healthy",
                "response_time": response_time,
                "redis_version": info.get("redis_version"),
                "connected_clients": info.get("connected_clients"),
                "used_memory": info.get("used_memory_human"),
                "hit_rate": self._get_hit_rate(),
                "compression_rate": self._get_compression_rate(),
                "warmup_keys": len(self.warmup_keys)
            }
            
        except Exception as e:
            self.is_healthy = False
            self.logger.error(f"Cache health check failed: {e}")
            
            return {
                "status": "unhealthy",
                "error": str(e),
                "last_health_check": self.last_health_check
            }
    
    async def _test_connection(self) -> Any:
        """Test Redis connection"""
        
        try:
            await self.redis_client.ping()
            self.logger.info("Redis connection test successful")
            
        except Exception as e:
            self.logger.error(f"Redis connection test failed: {e}")
            raise CacheException("connection test", reason=str(e))
    
    async def _initialize_warmup(self) -> Any:
        """Initialize cache warming"""
        
        try:
            # Start warmup task
            self.warmup_task = asyncio.create_task(self._warmup_loop())
            
        except Exception as e:
            self.logger.error(f"Failed to initialize warmup: {e}")
    
    async def _warmup_loop(self) -> Any:
        """Background task for cache warming"""
        
        while True:
            try:
                # Warm up frequently accessed data
                await self._warmup_frequent_data()
                
                # Wait before next warmup
                await asyncio.sleep(300)  # 5 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Warmup loop error: {e}")
                await asyncio.sleep(60)  # Wait 1 minute on error
    
    async def _warmup_frequent_data(self) -> Any:
        """Warm up frequently accessed data"""
        
        try:
            # This is a simplified warmup implementation
            # In production, you'd analyze access patterns and warm up accordingly
            
            frequent_data = {
                "config:app": {"version": "2.0.0", "environment": "production"},
                "stats:daily": {"requests": 0, "errors": 0, "cache_hits": 0},
                "health:status": {"status": "healthy", "timestamp": time.time()}
            }
            
            await self.warm_cache(frequent_data, ttl=1800)  # 30 minutes
            
        except Exception as e:
            self.logger.error(f"Frequent data warmup failed: {e}")
    
    async def _serialize_data(self, data: Any) -> bytes:
        """Serialize and compress data"""
        
        try:
            # Serialize to JSON
            if isinstance(data, (dict, list)):
                serialized = orjson.dumps(data)
            else:
                serialized = str(data).encode('utf-8')
            
            # Compress if enabled and data is large enough
            if (self.config.compression and 
                len(serialized) > self.config.compression_threshold):
                
                compressed = await self._compress_zstd(serialized)
                self.compression_count += 1
                
                # Add compression header
                return b'zstd:' + compressed
            else:
                return b'none:' + serialized
                
        except Exception as e:
            self.logger.error(f"Data serialization failed: {e}")
            raise CacheException("serialization", reason=str(e))
    
    async def _deserialize_data(self, raw_data: bytes) -> Any:
        """Deserialize and decompress data"""
        
        try:
            # Check compression header
            if raw_data.startswith(b'zstd:'):
                compressed_data = raw_data[5:]
                decompressed = await self._decompress_zstd(compressed_data)
                return orjson.loads(decompressed)
            elif raw_data.startswith(b'none:'):
                serialized_data = raw_data[5:]
                return orjson.loads(serialized_data)
            else:
                # Legacy format (no header)
                return orjson.loads(raw_data)
                
        except Exception as e:
            self.logger.error(f"Data deserialization failed: {e}")
            raise CacheException("deserialization", reason=str(e))
    
    async def _compress_zstd(self, data: bytes) -> bytes:
        """Compress data using Zstandard"""
        
        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None, 
                zstd.compress, 
                data, 
                3  # Compression level
            )
        except Exception as e:
            self.logger.error(f"Zstandard compression failed: {e}")
            return data
    
    async def _decompress_zstd(self, data: bytes) -> bytes:
        """Decompress data using Zstandard"""
        
        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None, 
                zstd.decompress, 
                data
            )
        except Exception as e:
            self.logger.error(f"Zstandard decompression failed: {e}")
            return data
    
    async def _compress_lz4(self, data: bytes) -> bytes:
        """Compress data using LZ4"""
        
        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None, 
                lz4.frame.compress, 
                data
            )
        except Exception as e:
            self.logger.error(f"LZ4 compression failed: {e}")
            return data
    
    async def _decompress_lz4(self, data: bytes) -> bytes:
        """Decompress data using LZ4"""
        
        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None, 
                lz4.frame.decompress, 
                data
            )
        except Exception as e:
            self.logger.error(f"LZ4 decompression failed: {e}")
            return data
    
    def _get_hit_rate(self) -> float:
        """Calculate cache hit rate"""
        
        total_requests = self.hit_count + self.miss_count
        if total_requests == 0:
            return 0.0
        
        return (self.hit_count / total_requests) * 100
    
    def _get_compression_rate(self) -> float:
        """Calculate compression rate"""
        
        if self.set_count == 0:
            return 0.0
        
        return (self.compression_count / self.set_count) * 100
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get cache performance metrics"""
        
        return {
            "get_count": self.get_count,
            "set_count": self.set_count,
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "delete_count": self.delete_count,
            "compression_count": self.compression_count,
            "hit_rate": self._get_hit_rate(),
            "compression_rate": self._get_compression_rate(),
            "average_get_time": self._get_average_get_time(),
            "average_set_time": self._get_average_set_time(),
            "is_healthy": self.is_healthy,
            "last_health_check": self.last_health_check,
            "warmup_keys": len(self.warmup_keys)
        }
    
    def _get_average_get_time(self) -> float:
        """Get average get operation time"""
        
        if self.get_count == 0:
            return 0.0
        
        return self.total_get_time / self.get_count
    
    def _get_average_set_time(self) -> float:
        """Get average set operation time"""
        
        if self.set_count == 0:
            return 0.0
        
        return self.total_set_time / self.set_count 