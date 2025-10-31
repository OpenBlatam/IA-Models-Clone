from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

from typing import Any, Dict, List, Optional, Union, TypeVar, Generic
import json
import pickle
from datetime import datetime, timedelta
import redis
from functools import wraps
import logging
from pydantic import BaseModel
from typing import Any, List, Dict, Optional
import asyncio
"""
Redis Manager - Enhanced Onyx Integration
Redis-based caching and data management for Onyx.
"""

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=BaseModel)

class RedisManager:
    """Redis manager for Onyx with optimized caching and data management."""
    
    def __init__(self, host: str = 'localhost', port: int = 6379, db: int = 0):
        """Initialize Redis connection."""
        self.redis = redis.Redis(
            host=host,
            port=port,
            db=db,
            decode_responses=True,
            socket_timeout=5,
            socket_connect_timeout=5
        )
        self._model_cache = {}
        self._context_cache = {}
    
    def _generate_key(self, prefix: str, identifier: str) -> str:
        """Generate a Redis key with prefix and identifier."""
        return f"onyx:{prefix}:{identifier}"
    
    def cache_model(self, model: T, prefix: str, identifier: str, 
                   expire: Optional[int] = None) -> None:
        """Cache a model instance in Redis."""
        key = self._generate_key(prefix, identifier)
        try:
            # Serialize model to JSON
            data = model.model_dump_json()
            self.redis.set(key, data)
            if expire:
                self.redis.expire(key, expire)
            logger.debug(f"Cached model {prefix}:{identifier}")
        except Exception as e:
            logger.error(f"Error caching model: {e}")
    
    def get_cached_model(self, model_class: type[T], prefix: str, 
                        identifier: str) -> Optional[T]:
        """Retrieve a cached model instance from Redis."""
        key = self._generate_key(prefix, identifier)
        try:
            data = self.redis.get(key)
            if data:
                return model_class.model_validate_json(data)
            return None
        except Exception as e:
            logger.error(f"Error retrieving cached model: {e}")
            return None
    
    def cache_context(self, context: Dict[str, Any], prefix: str, 
                     identifier: str, expire: Optional[int] = None) -> None:
        """Cache context data in Redis."""
        key = self._generate_key(prefix, identifier)
        try:
            data = json.dumps(context)
            self.redis.set(key, data)
            if expire:
                self.redis.expire(key, expire)
            logger.debug(f"Cached context {prefix}:{identifier}")
        except Exception as e:
            logger.error(f"Error caching context: {e}")
    
    def get_cached_context(self, prefix: str, identifier: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached context data from Redis."""
        key = self._generate_key(prefix, identifier)
        try:
            data = self.redis.get(key)
            if data:
                return json.loads(data)
            return None
        except Exception as e:
            logger.error(f"Error retrieving cached context: {e}")
            return None
    
    def cache_prompt(self, prompt: str, prefix: str, identifier: str, 
                    expire: Optional[int] = None) -> None:
        """Cache a prompt in Redis."""
        key = self._generate_key(prefix, identifier)
        try:
            self.redis.set(key, prompt)
            if expire:
                self.redis.expire(key, expire)
            logger.debug(f"Cached prompt {prefix}:{identifier}")
        except Exception as e:
            logger.error(f"Error caching prompt: {e}")
    
    def get_cached_prompt(self, prefix: str, identifier: str) -> Optional[str]:
        """Retrieve a cached prompt from Redis."""
        key = self._generate_key(prefix, identifier)
        try:
            return self.redis.get(key)
        except Exception as e:
            logger.error(f"Error retrieving cached prompt: {e}")
            return None
    
    def cache_metrics(self, metrics: Dict[str, float], prefix: str, 
                     identifier: str, expire: Optional[int] = None) -> None:
        """Cache metrics data in Redis."""
        key = self._generate_key(prefix, identifier)
        try:
            data = json.dumps(metrics)
            self.redis.set(key, data)
            if expire:
                self.redis.expire(key, expire)
            logger.debug(f"Cached metrics {prefix}:{identifier}")
        except Exception as e:
            logger.error(f"Error caching metrics: {e}")
    
    def get_cached_metrics(self, prefix: str, identifier: str) -> Optional[Dict[str, float]]:
        """Retrieve cached metrics from Redis."""
        key = self._generate_key(prefix, identifier)
        try:
            data = self.redis.get(key)
            if data:
                return json.loads(data)
            return None
        except Exception as e:
            logger.error(f"Error retrieving cached metrics: {e}")
            return None
    
    def increment_counter(self, prefix: str, identifier: str, 
                         amount: int = 1) -> int:
        """Increment a counter in Redis."""
        key = self._generate_key(prefix, identifier)
        try:
            return self.redis.incrby(key, amount)
        except Exception as e:
            logger.error(f"Error incrementing counter: {e}")
            return 0
    
    def get_counter(self, prefix: str, identifier: str) -> int:
        """Get counter value from Redis."""
        key = self._generate_key(prefix, identifier)
        try:
            value = self.redis.get(key)
            return int(value) if value else 0
        except Exception as e:
            logger.error(f"Error getting counter: {e}")
            return 0
    
    def add_to_set(self, prefix: str, identifier: str, 
                  value: str) -> None:
        """Add value to a Redis set."""
        key = self._generate_key(prefix, identifier)
        try:
            self.redis.sadd(key, value)
            logger.debug(f"Added to set {prefix}:{identifier}")
        except Exception as e:
            logger.error(f"Error adding to set: {e}")
    
    def get_set_members(self, prefix: str, identifier: str) -> List[str]:
        """Get all members of a Redis set."""
        key = self._generate_key(prefix, identifier)
        try:
            return list(self.redis.smembers(key))
        except Exception as e:
            logger.error(f"Error getting set members: {e}")
            return []
    
    def delete_key(self, prefix: str, identifier: str) -> None:
        """Delete a key from Redis."""
        key = self._generate_key(prefix, identifier)
        try:
            self.redis.delete(key)
            logger.debug(f"Deleted key {prefix}:{identifier}")
        except Exception as e:
            logger.error(f"Error deleting key: {e}")
    
    def clear_prefix(self, prefix: str) -> None:
        """Clear all keys with a specific prefix."""
        try:
            pattern = f"onyx:{prefix}:*"
            keys = self.redis.keys(pattern)
            if keys:
                self.redis.delete(*keys)
            logger.debug(f"Cleared prefix {prefix}")
        except Exception as e:
            logger.error(f"Error clearing prefix: {e}")

# Decorator for caching function results
def cache_result(prefix: str, expire: Optional[int] = None):
    """Decorator to cache function results in Redis."""
    def decorator(func) -> Any:
        @wraps(func)
        async def wrapper(self, *args, **kwargs) -> Any:
            # Generate cache key from function name and arguments
            key_parts = [func.__name__]
            key_parts.extend(str(arg) for arg in args)
            key_parts.extend(f"{k}:{v}" for k, v in sorted(kwargs.items()))
            identifier = ":".join(key_parts)
            
            # Try to get cached result
            cached_result = self.redis_manager.get_cached_context(prefix, identifier)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = await func(self, *args, **kwargs)
            self.redis_manager.cache_context(result, prefix, identifier, expire)
            return result
        return wrapper
    return decorator

# Global Redis manager instance
redis_manager = RedisManager()

# Example usage:
"""
# Cache a model
redis_manager.cache_model(
    model=brand_voice,
    prefix='brand_voice',
    identifier='brand_123',
    expire=3600  # 1 hour
)

# Get cached model
cached_model = redis_manager.get_cached_model(
    model_class=BrandVoice,
    prefix='brand_voice',
    identifier='brand_123'
)

# Cache context
redis_manager.cache_context(
    context={'user_id': '123', 'preferences': {'theme': 'dark'}},
    prefix='user_context',
    identifier='user_123',
    expire=1800  # 30 minutes
)

# Get cached context
cached_context = redis_manager.get_cached_context(
    prefix='user_context',
    identifier='user_123'
)

# Cache prompt
redis_manager.cache_prompt(
    prompt='Generate content for {brand}',
    prefix='prompts',
    identifier='content_generation',
    expire=86400  # 24 hours
)

# Get cached prompt
cached_prompt = redis_manager.get_cached_prompt(
    prefix='prompts',
    identifier='content_generation'
)

# Cache metrics
redis_manager.cache_metrics(
    metrics={'clicks': 100, 'conversions': 10},
    prefix='campaign_metrics',
    identifier='campaign_123',
    expire=3600  # 1 hour
)

# Get cached metrics
cached_metrics = redis_manager.get_cached_metrics(
    prefix='campaign_metrics',
    identifier='campaign_123'
)

# Use counter
redis_manager.increment_counter(
    prefix='api_calls',
    identifier='endpoint_123'
)

# Get counter value
api_calls = redis_manager.get_counter(
    prefix='api_calls',
    identifier='endpoint_123'
)

# Add to set
redis_manager.add_to_set(
    prefix='active_users',
    identifier='session_123',
    value='user_456'
)

# Get set members
active_users = redis_manager.get_set_members(
    prefix='active_users',
    identifier='session_123'
)

# Delete key
redis_manager.delete_key(
    prefix='brand_voice',
    identifier='brand_123'
)

# Clear prefix
redis_manager.clear_prefix('brand_voice')

# Use decorator
@cache_result(prefix='api_results', expire=300)  # 5 minutes
async def get_api_data(self, endpoint: str, params: dict):
    
    """get_api_data function."""
# API call implementation
    pass
""" 