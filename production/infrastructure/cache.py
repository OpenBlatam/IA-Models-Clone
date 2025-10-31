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
import json
import logging
import pickle
from typing import Any, Optional, Dict, List
from datetime import datetime, timedelta
import redis.asyncio as redis
from redis.asyncio import ConnectionPool
                import gzip
                import gzip
from typing import Any, List, Dict, Optional
"""
Cache Infrastructure
===================

Redis cache service implementation with advanced caching features.
"""



logger = logging.getLogger(__name__)


class RedisCacheService:
    """Redis implementation of cache service with advanced features."""
    
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        pool_size: int = 20,
        max_connections: int = 50,
        ttl: int = 3600,
        max_size: int = 10000,
        compression_threshold: int = 1024
    ):
        
    """__init__ function."""
self.redis_url = redis_url
        self.pool_size = pool_size
        self.max_connections = max_connections
        self.default_ttl = ttl
        self.max_size = max_size
        self.compression_threshold = compression_threshold
        self.pool = None
        self.client = None
        self._initialized = False
        self._stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "deletes": 0
        }
    
    async def initialize(self) -> Any:
        """Initialize Redis connection pool and client."""
        if self._initialized:
            return
        
        try:
            # Create connection pool
            self.pool = ConnectionPool.from_url(
                self.redis_url,
                max_connections=self.max_connections,
                decode_responses=False  # Keep as bytes for compression
            )
            
            # Create client
            self.client = redis.Redis(connection_pool=self.pool)
            
            # Test connection
            await self.client.ping()
            
            self._initialized = True
            logger.info("Redis cache service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Redis cache service: {e}")
            raise
    
    async def cleanup(self) -> Any:
        """Cleanup Redis connections."""
        if self.client:
            await self.client.close()
        if self.pool:
            await self.pool.disconnect()
        self._initialized = False
        logger.info("Redis cache service cleaned up")
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache with automatic decompression."""
        if not self._initialized:
            raise RuntimeError("Cache service not initialized")
        
        try:
            # Get value from Redis
            value = await self.client.get(key)
            
            if value is None:
                self._stats["misses"] += 1
                return None
            
            # Deserialize and decompress if needed
            deserialized = await self._deserialize_value(value)
            
            self._stats["hits"] += 1
            logger.debug(f"Cache hit for key: {key}")
            return deserialized
            
        except Exception as e:
            logger.error(f"Error getting value for key {key}: {e}")
            self._stats["misses"] += 1
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache with automatic compression."""
        if not self._initialized:
            raise RuntimeError("Cache service not initialized")
        
        try:
            # Serialize and compress if needed
            serialized = await self._serialize_value(value)
            
            # Set value with TTL
            ttl = ttl or self.default_ttl
            result = await self.client.setex(key, ttl, serialized)
            
            if result:
                self._stats["sets"] += 1
                logger.debug(f"Cache set for key: {key} (TTL: {ttl}s)")
            
            return bool(result)
            
        except Exception as e:
            logger.error(f"Error setting value for key {key}: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete value from cache."""
        if not self._initialized:
            raise RuntimeError("Cache service not initialized")
        
        try:
            result = await self.client.delete(key)
            if result:
                self._stats["deletes"] += 1
                logger.debug(f"Cache delete for key: {key}")
            return bool(result)
            
        except Exception as e:
            logger.error(f"Error deleting key {key}: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        if not self._initialized:
            raise RuntimeError("Cache service not initialized")
        
        try:
            return bool(await self.client.exists(key))
        except Exception as e:
            logger.error(f"Error checking existence of key {key}: {e}")
            return False
    
    async def increment(self, key: str, amount: int = 1) -> int:
        """Increment counter in cache."""
        if not self._initialized:
            raise RuntimeError("Cache service not initialized")
        
        try:
            return await self.client.incrby(key, amount)
        except Exception as e:
            logger.error(f"Error incrementing key {key}: {e}")
            return 0
    
    async def get_many(self, keys: List[str]) -> Dict[str, Any]:
        """Get multiple values from cache."""
        if not self._initialized:
            raise RuntimeError("Cache service not initialized")
        
        try:
            # Get values from Redis
            values = await self.client.mget(keys)
            
            # Process results
            result = {}
            for key, value in zip(keys, values):
                if value is not None:
                    result[key] = await self._deserialize_value(value)
                    self._stats["hits"] += 1
                else:
                    self._stats["misses"] += 1
            
            logger.debug(f"Cache multi-get: {len(result)}/{len(keys)} hits")
            return result
            
        except Exception as e:
            logger.error(f"Error getting multiple values: {e}")
            return {}
    
    async def set_many(self, data: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """Set multiple values in cache."""
        if not self._initialized:
            raise RuntimeError("Cache service not initialized")
        
        try:
            # Prepare pipeline
            pipeline = self.client.pipeline()
            
            # Add set operations to pipeline
            ttl = ttl or self.default_ttl
            for key, value in data.items():
                serialized = await self._serialize_value(value)
                pipeline.setex(key, ttl, serialized)
            
            # Execute pipeline
            results = await pipeline.execute()
            
            # Count successful sets
            successful = sum(1 for result in results if result)
            self._stats["sets"] += successful
            
            logger.debug(f"Cache multi-set: {successful}/{len(data)} successful")
            return successful == len(data)
            
        except Exception as e:
            logger.error(f"Error setting multiple values: {e}")
            return False
    
    async def clear_pattern(self, pattern: str) -> int:
        """Clear cache entries matching pattern."""
        if not self._initialized:
            raise RuntimeError("Cache service not initialized")
        
        try:
            # Scan for keys matching pattern
            keys_to_delete = []
            async for key in self.client.scan_iter(match=pattern):
                keys_to_delete.append(key)
            
            if keys_to_delete:
                # Delete keys in batches
                deleted = 0
                batch_size = 1000
                for i in range(0, len(keys_to_delete), batch_size):
                    batch = keys_to_delete[i:i + batch_size]
                    result = await self.client.delete(*batch)
                    deleted += result
                
                self._stats["deletes"] += deleted
                logger.info(f"Cleared {deleted} cache entries matching pattern: {pattern}")
                return deleted
            
            return 0
            
        except Exception as e:
            logger.error(f"Error clearing pattern {pattern}: {e}")
            return 0
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if not self._initialized:
            raise RuntimeError("Cache service not initialized")
        
        try:
            # Get Redis info
            info = await self.client.info()
            
            # Calculate hit ratio
            total_requests = self._stats["hits"] + self._stats["misses"]
            hit_ratio = self._stats["hits"] / total_requests if total_requests > 0 else 0.0
            
            # Get memory usage
            memory_info = await self.client.memory_usage()
            
            return {
                "hits": self._stats["hits"],
                "misses": self._stats["misses"],
                "sets": self._stats["sets"],
                "deletes": self._stats["deletes"],
                "hit_ratio": hit_ratio,
                "total_requests": total_requests,
                "memory_usage": memory_info,
                "redis_info": {
                    "connected_clients": info.get("connected_clients", 0),
                    "used_memory": info.get("used_memory", 0),
                    "used_memory_peak": info.get("used_memory_peak", 0),
                    "keyspace_hits": info.get("keyspace_hits", 0),
                    "keyspace_misses": info.get("keyspace_misses", 0)
                },
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {}
    
    async def _serialize_value(self, value: Any) -> bytes:
        """Serialize and compress value if needed."""
        try:
            # Serialize to JSON first
            json_data = json.dumps(value, default=str)
            
            # Compress if above threshold
            if len(json_data) > self.compression_threshold:
                compressed = gzip.compress(json_data.encode('utf-8'))
                # Add compression marker
                return b"GZIP:" + compressed
            else:
                return json_data.encode('utf-8')
                
        except Exception as e:
            logger.error(f"Error serializing value: {e}")
            # Fallback to pickle
            return pickle.dumps(value)
    
    async def _deserialize_value(self, value: bytes) -> Any:
        """Deserialize and decompress value if needed."""
        try:
            # Check if compressed
            if value.startswith(b"GZIP:"):
                compressed_data = value[5:]  # Remove "GZIP:" prefix
                json_data = gzip.decompress(compressed_data).decode('utf-8')
            else:
                json_data = value.decode('utf-8')
            
            # Deserialize JSON
            return json.loads(json_data)
            
        except Exception as e:
            logger.error(f"Error deserializing value: {e}")
            # Fallback to pickle
            try:
                return pickle.loads(value)
            except Exception as pickle_error:
                logger.error(f"Error with pickle fallback: {pickle_error}")
                return None
    
    async def set_with_tags(self, key: str, value: Any, tags: List[str], ttl: Optional[int] = None) -> bool:
        """Set value with tags for easier invalidation."""
        if not self._initialized:
            raise RuntimeError("Cache service not initialized")
        
        try:
            # Set the main value
            success = await self.set(key, value, ttl)
            if not success:
                return False
            
            # Store tags for the key
            tag_key = f"tags:{key}"
            await self.set(tag_key, tags, ttl)
            
            # Add key to tag sets
            for tag in tags:
                tag_set_key = f"tag_set:{tag}"
                await self.client.sadd(tag_set_key, key)
                if ttl:
                    await self.client.expire(tag_set_key, ttl)
            
            return True
            
        except Exception as e:
            logger.error(f"Error setting value with tags for key {key}: {e}")
            return False
    
    async def invalidate_by_tags(self, tags: List[str]) -> int:
        """Invalidate all keys with specified tags."""
        if not self._initialized:
            raise RuntimeError("Cache service not initialized")
        
        try:
            keys_to_delete = set()
            
            # Get all keys for each tag
            for tag in tags:
                tag_set_key = f"tag_set:{tag}"
                keys = await self.client.smembers(tag_set_key)
                keys_to_delete.update(keys)
            
            if keys_to_delete:
                # Delete keys and their tag metadata
                pipeline = self.client.pipeline()
                for key in keys_to_delete:
                    pipeline.delete(key)
                    pipeline.delete(f"tags:{key}")
                
                # Delete tag sets
                for tag in tags:
                    pipeline.delete(f"tag_set:{tag}")
                
                results = await pipeline.execute()
                deleted = sum(1 for result in results if result)
                
                logger.info(f"Invalidated {deleted} cache entries by tags: {tags}")
                return deleted
            
            return 0
            
        except Exception as e:
            logger.error(f"Error invalidating by tags {tags}: {e}")
            return 0 