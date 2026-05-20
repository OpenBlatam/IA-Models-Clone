"""
Redis Client
============

Advanced Redis client for caching and session management.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
import json
import pickle
from datetime import datetime, timedelta
import redis.asyncio as redis
from redis.asyncio import ConnectionPool

logger = logging.getLogger(__name__)

class RedisClient:
    """
    Advanced Redis client with connection pooling and caching.
    
    Features:
    - Connection pooling
    - Automatic serialization/deserialization
    - TTL management
    - Pub/Sub support
    - Session management
    - Performance monitoring
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        max_connections: int = 20,
        decode_responses: bool = True
    ):
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self.max_connections = max_connections
        self.decode_responses = decode_responses
        
        self.pool = None
        self.redis = None
        self.connected = False
        
    async def initialize(self):
        """Initialize Redis connection."""
        logger.info("Initializing Redis Client...")
        
        try:
            # Create connection pool
            self.pool = ConnectionPool(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
                max_connections=self.max_connections,
                decode_responses=self.decode_responses
            )
            
            # Create Redis client
            self.redis = redis.Redis(connection_pool=self.pool)
            
            # Test connection
            await self.redis.ping()
            self.connected = True
            
            logger.info("Redis Client initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Redis Client: {str(e)}")
            raise
    
    async def set(
        self, 
        key: str, 
        value: Any, 
        ttl: Optional[int] = None,
        serialize: bool = True
    ) -> bool:
        """
        Set a key-value pair.
        
        Args:
            key: Redis key
            value: Value to store
            ttl: Time to live in seconds
            serialize: Whether to serialize the value
            
        Returns:
            Success status
        """
        try:
            if not self.connected:
                raise Exception("Redis not connected")
            
            # Serialize value if needed
            if serialize and not isinstance(value, (str, int, float, bool)):
                value = json.dumps(value, default=str)
            
            # Set with TTL
            if ttl:
                result = await self.redis.setex(key, ttl, value)
            else:
                result = await self.redis.set(key, value)
            
            return bool(result)
            
        except Exception as e:
            logger.error(f"Failed to set key {key}: {str(e)}")
            return False
    
    async def get(
        self, 
        key: str, 
        deserialize: bool = True,
        default: Any = None
    ) -> Any:
        """
        Get a value by key.
        
        Args:
            key: Redis key
            deserialize: Whether to deserialize the value
            default: Default value if key not found
            
        Returns:
            Value or default
        """
        try:
            if not self.connected:
                raise Exception("Redis not connected")
            
            value = await self.redis.get(key)
            
            if value is None:
                return default
            
            # Deserialize if needed
            if deserialize and isinstance(value, str):
                try:
                    return json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    return value
            
            return value
            
        except Exception as e:
            logger.error(f"Failed to get key {key}: {str(e)}")
            return default
    
    async def delete(self, key: str) -> bool:
        """Delete a key."""
        try:
            if not self.connected:
                raise Exception("Redis not connected")
            
            result = await self.redis.delete(key)
            return bool(result)
            
        except Exception as e:
            logger.error(f"Failed to delete key {key}: {str(e)}")
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        try:
            if not self.connected:
                raise Exception("Redis not connected")
            
            result = await self.redis.exists(key)
            return bool(result)
            
        except Exception as e:
            logger.error(f"Failed to check key existence {key}: {str(e)}")
            return False
    
    async def expire(self, key: str, ttl: int) -> bool:
        """Set TTL for a key."""
        try:
            if not self.connected:
                raise Exception("Redis not connected")
            
            result = await self.redis.expire(key, ttl)
            return bool(result)
            
        except Exception as e:
            logger.error(f"Failed to set TTL for key {key}: {str(e)}")
            return False
    
    async def ttl(self, key: str) -> int:
        """Get TTL for a key."""
        try:
            if not self.connected:
                raise Exception("Redis not connected")
            
            return await self.redis.ttl(key)
            
        except Exception as e:
            logger.error(f"Failed to get TTL for key {key}: {str(e)}")
            return -1
    
    async def keys(self, pattern: str = "*") -> List[str]:
        """Get keys matching pattern."""
        try:
            if not self.connected:
                raise Exception("Redis not connected")
            
            keys = await self.redis.keys(pattern)
            return [key.decode() if isinstance(key, bytes) else key for key in keys]
            
        except Exception as e:
            logger.error(f"Failed to get keys with pattern {pattern}: {str(e)}")
            return []
    
    async def hset(self, name: str, mapping: Dict[str, Any]) -> int:
        """Set hash fields."""
        try:
            if not self.connected:
                raise Exception("Redis not connected")
            
            # Serialize values
            serialized_mapping = {}
            for k, v in mapping.items():
                if not isinstance(v, (str, int, float, bool)):
                    serialized_mapping[k] = json.dumps(v, default=str)
                else:
                    serialized_mapping[k] = v
            
            return await self.redis.hset(name, mapping=serialized_mapping)
            
        except Exception as e:
            logger.error(f"Failed to set hash {name}: {str(e)}")
            return 0
    
    async def hget(self, name: str, key: str) -> Any:
        """Get hash field value."""
        try:
            if not self.connected:
                raise Exception("Redis not connected")
            
            value = await self.redis.hget(name, key)
            
            if value is None:
                return None
            
            # Try to deserialize
            if isinstance(value, str):
                try:
                    return json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    return value
            
            return value
            
        except Exception as e:
            logger.error(f"Failed to get hash field {name}.{key}: {str(e)}")
            return None
    
    async def hgetall(self, name: str) -> Dict[str, Any]:
        """Get all hash fields."""
        try:
            if not self.connected:
                raise Exception("Redis not connected")
            
            data = await self.redis.hgetall(name)
            
            # Deserialize values
            result = {}
            for k, v in data.items():
                if isinstance(v, str):
                    try:
                        result[k] = json.loads(v)
                    except (json.JSONDecodeError, TypeError):
                        result[k] = v
                else:
                    result[k] = v
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to get all hash fields {name}: {str(e)}")
            return {}
    
    async def hdel(self, name: str, *keys: str) -> int:
        """Delete hash fields."""
        try:
            if not self.connected:
                raise Exception("Redis not connected")
            
            return await self.redis.hdel(name, *keys)
            
        except Exception as e:
            logger.error(f"Failed to delete hash fields {name}: {str(e)}")
            return 0
    
    async def lpush(self, name: str, *values: Any) -> int:
        """Push values to list."""
        try:
            if not self.connected:
                raise Exception("Redis not connected")
            
            # Serialize values
            serialized_values = []
            for value in values:
                if not isinstance(value, (str, int, float, bool)):
                    serialized_values.append(json.dumps(value, default=str))
                else:
                    serialized_values.append(value)
            
            return await self.redis.lpush(name, *serialized_values)
            
        except Exception as e:
            logger.error(f"Failed to push to list {name}: {str(e)}")
            return 0
    
    async def rpop(self, name: str) -> Any:
        """Pop value from list."""
        try:
            if not self.connected:
                raise Exception("Redis not connected")
            
            value = await self.redis.rpop(name)
            
            if value is None:
                return None
            
            # Try to deserialize
            if isinstance(value, str):
                try:
                    return json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    return value
            
            return value
            
        except Exception as e:
            logger.error(f"Failed to pop from list {name}: {str(e)}")
            return None
    
    async def llen(self, name: str) -> int:
        """Get list length."""
        try:
            if not self.connected:
                raise Exception("Redis not connected")
            
            return await self.redis.llen(name)
            
        except Exception as e:
            logger.error(f"Failed to get list length {name}: {str(e)}")
            return 0
    
    async def lrange(self, name: str, start: int, end: int) -> List[Any]:
        """Get list range."""
        try:
            if not self.connected:
                raise Exception("Redis not connected")
            
            values = await self.redis.lrange(name, start, end)
            
            # Deserialize values
            result = []
            for value in values:
                if isinstance(value, str):
                    try:
                        result.append(json.loads(value))
                    except (json.JSONDecodeError, TypeError):
                        result.append(value)
                else:
                    result.append(value)
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to get list range {name}: {str(e)}")
            return []
    
    async def publish(self, channel: str, message: Any) -> int:
        """Publish message to channel."""
        try:
            if not self.connected:
                raise Exception("Redis not connected")
            
            # Serialize message
            if not isinstance(message, (str, int, float, bool)):
                message = json.dumps(message, default=str)
            
            return await self.redis.publish(channel, message)
            
        except Exception as e:
            logger.error(f"Failed to publish to channel {channel}: {str(e)}")
            return 0
    
    async def subscribe(self, *channels: str):
        """Subscribe to channels."""
        try:
            if not self.connected:
                raise Exception("Redis not connected")
            
            pubsub = self.redis.pubsub()
            await pubsub.subscribe(*channels)
            
            async for message in pubsub.listen():
                if message['type'] == 'message':
                    # Try to deserialize message
                    data = message['data']
                    if isinstance(data, str):
                        try:
                            data = json.loads(data)
                        except (json.JSONDecodeError, TypeError):
                            pass
                    
                    yield {
                        'channel': message['channel'],
                        'data': data,
                        'timestamp': datetime.utcnow().isoformat()
                    }
            
        except Exception as e:
            logger.error(f"Failed to subscribe to channels {channels}: {str(e)}")
            raise
    
    async def get_info(self) -> Dict[str, Any]:
        """Get Redis server info."""
        try:
            if not self.connected:
                raise Exception("Redis not connected")
            
            info = await self.redis.info()
            return info
            
        except Exception as e:
            logger.error(f"Failed to get Redis info: {str(e)}")
            return {}
    
    async def get_memory_usage(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        try:
            if not self.connected:
                raise Exception("Redis not connected")
            
            info = await self.get_info()
            
            return {
                "used_memory": info.get("used_memory", 0),
                "used_memory_human": info.get("used_memory_human", "0B"),
                "used_memory_peak": info.get("used_memory_peak", 0),
                "used_memory_peak_human": info.get("used_memory_peak_human", "0B"),
                "used_memory_rss": info.get("used_memory_rss", 0),
                "used_memory_rss_human": info.get("used_memory_rss_human", "0B"),
                "maxmemory": info.get("maxmemory", 0),
                "maxmemory_human": info.get("maxmemory_human", "0B")
            }
            
        except Exception as e:
            logger.error(f"Failed to get memory usage: {str(e)}")
            return {}
    
    async def cleanup(self):
        """Cleanup Redis connection."""
        try:
            if self.redis:
                await self.redis.close()
            
            if self.pool:
                await self.pool.disconnect()
            
            self.connected = False
            logger.info("Redis Client cleaned up successfully")
            
        except Exception as e:
            logger.error(f"Failed to cleanup Redis Client: {str(e)}")











