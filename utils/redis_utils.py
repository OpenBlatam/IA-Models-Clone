from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
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
import time
from .redis_config import RedisConfig, get_config
from .error_system import (
from typing import Any, List, Dict, Optional
import asyncio
"""
Redis Utilities - Onyx Integration
Utility functions for Redis operations in Onyx with enhanced error handling.
"""
    error_factory,
    ErrorContext,
    CacheError,
    ValidationError,
    SystemError,
    handle_errors,
    ErrorCategory
)

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=BaseModel)

class DateTimeEncoder(json.JSONEncoder):
    """Custom JSON encoder for datetime objects."""
    def default(self, obj) -> Any:
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

class RedisUtils:
    """Utility functions for Redis operations with enhanced error handling."""
    
    def __init__(self, config: Optional[RedisConfig] = None):
        """Initialize Redis utilities with configuration."""
        self.config = config or get_config()
        self.redis = redis.Redis(
            host=self.config.host,
            port=self.config.port,
            db=self.config.db,
            password=self.config.password,
            max_connections=self.config.max_connections,
            socket_timeout=self.config.socket_timeout,
            socket_connect_timeout=self.config.socket_connect_timeout,
            decode_responses=True
        )
    
    def _retry_operation(self, operation: callable, *args, **kwargs) -> Any:
        """Retry a Redis operation with exponential backoff."""
        for attempt in range(self.config.max_retries):
            try:
                return operation(*args, **kwargs)
            except redis.RedisError as e:
                if attempt == self.config.max_retries - 1:
                    raise
                delay = self.config.retry_delay * (2 ** attempt)
                logger.warning(f"Redis operation failed, retrying in {delay}s: {e}")
                time.sleep(delay)
    
    def _serialize(self, data: Any) -> str:
        """Serialize data for Redis storage."""
        if isinstance(data, BaseModel):
            return data.model_dump_json()
        return json.dumps(data, cls=DateTimeEncoder)
    
    def _deserialize(self, data: str, model_class: Optional[type[T]] = None) -> Any:
        """Deserialize data from Redis storage."""
        if not data:
            return None
        if model_class:
            return model_class.model_validate_json(data)
        return json.loads(data)
    
    def _generate_key(self, prefix: str, identifier: str) -> str:
        """Generate a Redis key with prefix and identifier."""
        return f"onyx:{prefix}:{identifier}"
    
    @handle_errors(ErrorCategory.CACHE, operation="cache_data")
    def cache_data(self, data: Any, prefix: str, identifier: str, 
                  expire: Optional[int] = None) -> None:
        """Cache data in Redis with retry mechanism and enhanced error handling."""
        # Guard clause: Validate input parameters
        if not prefix or not prefix.strip():
            context = ErrorContext(operation="cache_data", additional_data={"prefix": prefix})
            raise error_factory.create_validation_error(
                "Cache prefix cannot be empty",
                field="prefix",
                value=prefix,
                context=context
            )
        
        if not identifier or not identifier.strip():
            context = ErrorContext(operation="cache_data", additional_data={"identifier": identifier})
            raise error_factory.create_validation_error(
                "Cache identifier cannot be empty",
                field="identifier",
                value=identifier,
                context=context
            )
        
        if data is None:
            logger.warning(f"Attempting to cache None data for {prefix}:{identifier}")
            return
        
        key = self._generate_key(prefix, identifier)
        try:
            logger.info(f"Caching data for key: {key}")
            serialized_data = self._serialize(data)
            self._retry_operation(
                self.redis.set,
                key,
                serialized_data,
                ex=expire or self.config.default_expire
            )
            logger.info(f"Successfully cached data for {prefix}:{identifier} (expires in {expire or self.config.default_expire}s)")
        except redis.ConnectionError as e:
            context = ErrorContext(
                operation="cache_data",
                additional_data={"key": key, "prefix": prefix, "identifier": identifier}
            )
            raise error_factory.create_cache_error(
                f"Unable to connect to Redis cache: {str(e)}",
                cache_key=key,
                operation="set",
                context=context,
                original_exception=e
            )
        except redis.RedisError as e:
            context = ErrorContext(
                operation="cache_data",
                additional_data={"key": key, "prefix": prefix, "identifier": identifier}
            )
            raise error_factory.create_cache_error(
                f"Redis error during cache operation: {str(e)}",
                cache_key=key,
                operation="set",
                context=context,
                original_exception=e
            )
        except Exception as e:
            context = ErrorContext(
                operation="cache_data",
                additional_data={"key": key, "prefix": prefix, "identifier": identifier}
            )
            raise error_factory.create_system_error(
                f"Unexpected error during cache operation: {str(e)}",
                component="redis_utils",
                context=context,
                original_exception=e
            )
    
    @handle_errors(ErrorCategory.CACHE, operation="get_cached_data")
    def get_cached_data(self, prefix: str, identifier: str, 
                       model_class: Optional[type[T]] = None) -> Optional[Any]:
        """Retrieve cached data from Redis with retry mechanism and enhanced error handling."""
        # Guard clause: Validate input parameters
        if not prefix or not prefix.strip():
            context = ErrorContext(operation="get_cached_data", additional_data={"prefix": prefix})
            logger.error("Cache retrieval failed: Prefix cannot be empty")
            return None
        
        if not identifier or not identifier.strip():
            context = ErrorContext(operation="get_cached_data", additional_data={"identifier": identifier})
            logger.error("Cache retrieval failed: Identifier cannot be empty")
            return None
        
        key = self._generate_key(prefix, identifier)
        try:
            logger.debug(f"Retrieving cached data for key: {key}")
            data = self._retry_operation(self.redis.get, key)
            if data:
                deserialized_data = self._deserialize(data, model_class)
                logger.debug(f"Successfully retrieved cached data for {prefix}:{identifier}")
                return deserialized_data
            else:
                logger.debug(f"No cached data found for {prefix}:{identifier}")
                return None
        except redis.ConnectionError as e:
            context = ErrorContext(
                operation="get_cached_data",
                additional_data={"key": key, "prefix": prefix, "identifier": identifier}
            )
            logger.error(f"Redis connection error: {str(e)}")
            logger.warning("Cache retrieval failed due to connection issues. Returning None.")
            return None
        except redis.RedisError as e:
            context = ErrorContext(
                operation="get_cached_data",
                additional_data={"key": key, "prefix": prefix, "identifier": identifier}
            )
            logger.error(f"Redis error: {str(e)}")
            logger.warning("Cache retrieval failed due to Redis error. Returning None.")
            return None
        except Exception as e:
            context = ErrorContext(
                operation="get_cached_data",
                additional_data={"key": key, "prefix": prefix, "identifier": identifier}
            )
            logger.error(f"Unexpected error: {str(e)}", exc_info=True)
            logger.warning("Cache retrieval failed due to unexpected error. Returning None.")
            return None
    
    @handle_errors(ErrorCategory.CACHE, operation="cache_batch")
    def cache_batch(self, data_dict: Dict[str, Any], prefix: str, 
                   expire: Optional[int] = None) -> None:
        """Cache multiple data items in Redis with pipeline and enhanced error handling."""
        # Guard clause: Validate input parameters
        if not prefix or not prefix.strip():
            context = ErrorContext(operation="cache_batch", additional_data={"prefix": prefix})
            raise error_factory.create_validation_error(
                "Cache prefix cannot be empty",
                field="prefix",
                value=prefix,
                context=context
            )
        
        if not data_dict:
            logger.warning("Batch cache operation: Empty data dictionary provided")
            return
        
        try:
            logger.info(f"Caching batch data for {len(data_dict)} items with prefix: {prefix}")
            with self.redis.pipeline() as pipe:
                for identifier, data in data_dict.items():
                    if data is not None:  # Skip None values
                        key = self._generate_key(prefix, identifier)
                        serialized_data = self._serialize(data)
                        pipe.set(key, serialized_data)
                        if expire:
                            pipe.expire(key, expire)
                pipe.execute()
            logger.info(f"Successfully cached batch data for {len(data_dict)} items with prefix: {prefix}")
        except redis.ConnectionError as e:
            context = ErrorContext(
                operation="cache_batch",
                additional_data={"prefix": prefix, "item_count": len(data_dict)}
            )
            raise error_factory.create_cache_error(
                f"Unable to connect to Redis cache during batch operation: {str(e)}",
                operation="batch_set",
                context=context,
                original_exception=e
            )
        except redis.RedisError as e:
            context = ErrorContext(
                operation="cache_batch",
                additional_data={"prefix": prefix, "item_count": len(data_dict)}
            )
            raise error_factory.create_cache_error(
                f"Redis error during batch cache operation: {str(e)}",
                operation="batch_set",
                context=context,
                original_exception=e
            )
        except Exception as e:
            context = ErrorContext(
                operation="cache_batch",
                additional_data={"prefix": prefix, "item_count": len(data_dict)}
            )
            raise error_factory.create_system_error(
                f"Unexpected error during batch cache operation: {str(e)}",
                component="redis_utils",
                context=context,
                original_exception=e
            )
    
    @handle_errors(ErrorCategory.CACHE, operation="get_cached_batch")
    def get_cached_batch(self, prefix: str, identifiers: List[str], 
                        model_class: Optional[type[T]] = None) -> Dict[str, Any]:
        """Retrieve multiple cached data items from Redis with pipeline and enhanced error handling."""
        # Guard clause: Validate input parameters
        if not prefix or not prefix.strip():
            context = ErrorContext(operation="get_cached_batch", additional_data={"prefix": prefix})
            logger.error("Batch cache retrieval failed: Prefix cannot be empty")
            return {}
        
        if not identifiers:
            logger.warning("Batch cache retrieval: Empty identifiers list provided")
            return {}
        
        try:
            logger.debug(f"Retrieving batch cached data for {len(identifiers)} items with prefix: {prefix}")
            with self.redis.pipeline() as pipe:
                for identifier in identifiers:
                    key = self._generate_key(prefix, identifier)
                    pipe.get(key)
                results = pipe.execute()
            
            cached_data = {
                identifier: self._deserialize(data, model_class)
                for identifier, data in zip(identifiers, results)
                if data is not None
            }
            
            logger.debug(f"Successfully retrieved {len(cached_data)} items from batch cache for prefix: {prefix}")
            return cached_data
        except redis.ConnectionError as e:
            context = ErrorContext(
                operation="get_cached_batch",
                additional_data={"prefix": prefix, "identifier_count": len(identifiers)}
            )
            logger.error(f"Redis connection error: {str(e)}")
            logger.warning("Batch cache retrieval failed due to connection issues. Returning empty dict.")
            return {}
        except redis.RedisError as e:
            context = ErrorContext(
                operation="get_cached_batch",
                additional_data={"prefix": prefix, "identifier_count": len(identifiers)}
            )
            logger.error(f"Redis error: {str(e)}")
            logger.warning("Batch cache retrieval failed due to Redis error. Returning empty dict.")
            return {}
        except Exception as e:
            context = ErrorContext(
                operation="get_cached_batch",
                additional_data={"prefix": prefix, "identifier_count": len(identifiers)}
            )
            logger.error(f"Unexpected error: {str(e)}", exc_info=True)
            logger.warning("Batch cache retrieval failed due to unexpected error. Returning empty dict.")
            return {}
    
    @handle_errors(ErrorCategory.CACHE, operation="delete_batch")
    def delete_batch(self, prefix: str, identifiers: List[str]) -> None:
        """Delete multiple keys from Redis with pipeline and enhanced error handling."""
        # Guard clause: Validate input parameters
        if not prefix or not prefix.strip():
            context = ErrorContext(operation="delete_batch", additional_data={"prefix": prefix})
            raise error_factory.create_validation_error(
                "Cache prefix cannot be empty",
                field="prefix",
                value=prefix,
                context=context
            )
        
        if not identifiers:
            logger.warning("Batch delete operation: Empty identifiers list provided")
            return
        
        try:
            logger.info(f"Deleting batch keys for {len(identifiers)} items with prefix: {prefix}")
            with self.redis.pipeline() as pipe:
                for identifier in identifiers:
                    key = self._generate_key(prefix, identifier)
                    pipe.delete(key)
                pipe.execute()
            logger.info(f"Successfully deleted batch keys for {len(identifiers)} items with prefix: {prefix}")
        except redis.ConnectionError as e:
            context = ErrorContext(
                operation="delete_batch",
                additional_data={"prefix": prefix, "identifier_count": len(identifiers)}
            )
            raise error_factory.create_cache_error(
                f"Unable to connect to Redis cache during batch delete operation: {str(e)}",
                operation="batch_delete",
                context=context,
                original_exception=e
            )
        except redis.RedisError as e:
            context = ErrorContext(
                operation="delete_batch",
                additional_data={"prefix": prefix, "identifier_count": len(identifiers)}
            )
            raise error_factory.create_cache_error(
                f"Redis error during batch delete operation: {str(e)}",
                operation="batch_delete",
                context=context,
                original_exception=e
            )
        except Exception as e:
            context = ErrorContext(
                operation="delete_batch",
                additional_data={"prefix": prefix, "identifier_count": len(identifiers)}
            )
            raise error_factory.create_system_error(
                f"Unexpected error during batch delete operation: {str(e)}",
                component="redis_utils",
                context=context,
                original_exception=e
            )
    
    @handle_errors(ErrorCategory.CACHE, operation="scan_keys")
    def scan_keys(self, prefix: str, pattern: str = "*") -> List[str]:
        """Scan Redis keys with a specific prefix and pattern with enhanced error handling."""
        # Guard clause: Validate input parameters
        if not prefix or not prefix.strip():
            context = ErrorContext(operation="scan_keys", additional_data={"prefix": prefix})
            logger.error("Key scan operation failed: Prefix cannot be empty")
            return []
        
        try:
            logger.debug(f"Scanning keys with prefix: {prefix}, pattern: {pattern}")
            cursor = 0
            keys = []
            while True:
                cursor, found_keys = self._retry_operation(
                    self.redis.scan,
                    cursor,
                    match=f"onyx:{prefix}:{pattern}"
                )
                keys.extend(found_keys)
                if cursor == 0:
                    break
            
            logger.debug(f"Successfully scanned {len(keys)} keys with prefix: {prefix}")
            return keys
        except redis.ConnectionError as e:
            context = ErrorContext(
                operation="scan_keys",
                additional_data={"prefix": prefix, "pattern": pattern}
            )
            logger.error(f"Redis connection error: {str(e)}")
            logger.warning("Key scan failed due to connection issues. Returning empty list.")
            return []
        except redis.RedisError as e:
            context = ErrorContext(
                operation="scan_keys",
                additional_data={"prefix": prefix, "pattern": pattern}
            )
            logger.error(f"Redis error: {str(e)}")
            logger.warning("Key scan failed due to Redis error. Returning empty list.")
            return []
        except Exception as e:
            context = ErrorContext(
                operation="scan_keys",
                additional_data={"prefix": prefix, "pattern": pattern}
            )
            logger.error(f"Unexpected error: {str(e)}", exc_info=True)
            logger.warning("Key scan failed due to unexpected error. Returning empty list.")
            return []
    
    @handle_errors(ErrorCategory.CACHE, operation="get_memory_usage")
    def get_memory_usage(self) -> Dict[str, int]:
        """Get memory usage statistics from Redis with enhanced error handling."""
        try:
            logger.debug("Retrieving Redis memory usage statistics")
            info = self._retry_operation(self.redis.info, "memory")
            memory_stats = {
                "used_memory": info["used_memory"],
                "used_memory_peak": info["used_memory_peak"],
                "used_memory_lua": info["used_memory_lua"],
                "used_memory_scripts": info["used_memory_scripts"]
            }
            logger.debug("Successfully retrieved Redis memory usage statistics")
            return memory_stats
        except redis.ConnectionError as e:
            context = ErrorContext(operation="get_memory_usage")
            logger.error(f"Redis connection error: {str(e)}")
            logger.warning("Memory usage retrieval failed due to connection issues. Returning empty dict.")
            return {}
        except redis.RedisError as e:
            context = ErrorContext(operation="get_memory_usage")
            logger.error(f"Redis error: {str(e)}")
            logger.warning("Memory usage retrieval failed due to Redis error. Returning empty dict.")
            return {}
        except Exception as e:
            context = ErrorContext(operation="get_memory_usage")
            logger.error(f"Unexpected error: {str(e)}", exc_info=True)
            logger.warning("Memory usage retrieval failed due to unexpected error. Returning empty dict.")
            return {}
    
    @handle_errors(ErrorCategory.CACHE, operation="get_stats")
    def get_stats(self) -> Dict[str, Any]:
        """Get Redis statistics with enhanced error handling."""
        try:
            logger.debug("Retrieving Redis statistics")
            info = self._retry_operation(self.redis.info)
            stats = {
                "clients": info["clients"],
                "memory": info["memory"],
                "stats": info["stats"],
                "replication": info["replication"]
            }
            logger.debug("Successfully retrieved Redis statistics")
            return stats
        except redis.ConnectionError as e:
            context = ErrorContext(operation="get_stats")
            logger.error(f"Redis connection error: {str(e)}")
            logger.warning("Stats retrieval failed due to connection issues. Returning empty dict.")
            return {}
        except redis.RedisError as e:
            context = ErrorContext(operation="get_stats")
            logger.error(f"Redis error: {str(e)}")
            logger.warning("Stats retrieval failed due to Redis error. Returning empty dict.")
            return {}
        except Exception as e:
            context = ErrorContext(operation="get_stats")
            logger.error(f"Unexpected error: {str(e)}", exc_info=True)
            logger.warning("Stats retrieval failed due to unexpected error. Returning empty dict.")
            return {}
    
    @handle_errors(ErrorCategory.CACHE, operation="delete_key")
    def delete_key(self, prefix: str, identifier: str) -> None:
        """Delete a single key from Redis with enhanced error handling."""
        # Guard clause: Validate input parameters
        if not prefix or not prefix.strip():
            context = ErrorContext(operation="delete_key", additional_data={"prefix": prefix})
            raise error_factory.create_validation_error(
                "Cache prefix cannot be empty",
                field="prefix",
                value=prefix,
                context=context
            )
        
        if not identifier or not identifier.strip():
            context = ErrorContext(operation="delete_key", additional_data={"identifier": identifier})
            raise error_factory.create_validation_error(
                "Cache identifier cannot be empty",
                field="identifier",
                value=identifier,
                context=context
            )
        
        key = self._generate_key(prefix, identifier)
        try:
            logger.info(f"Deleting key: {key}")
            self._retry_operation(self.redis.delete, key)
            logger.info(f"Successfully deleted key: {key}")
        except redis.ConnectionError as e:
            context = ErrorContext(
                operation="delete_key",
                additional_data={"key": key, "prefix": prefix, "identifier": identifier}
            )
            raise error_factory.create_cache_error(
                f"Unable to connect to Redis cache during delete operation: {str(e)}",
                cache_key=key,
                operation="delete",
                context=context,
                original_exception=e
            )
        except redis.RedisError as e:
            context = ErrorContext(
                operation="delete_key",
                additional_data={"key": key, "prefix": prefix, "identifier": identifier}
            )
            raise error_factory.create_cache_error(
                f"Redis error during delete operation: {str(e)}",
                cache_key=key,
                operation="delete",
                context=context,
                original_exception=e
            )
        except Exception as e:
            context = ErrorContext(
                operation="delete_key",
                additional_data={"key": key, "prefix": prefix, "identifier": identifier}
            )
            raise error_factory.create_system_error(
                f"Unexpected error during delete operation: {str(e)}",
                component="redis_utils",
                context=context,
                original_exception=e
            )

# Global Redis utilities instance
redis_utils = RedisUtils()

# Example usage:
"""
# Cache data
redis_utils.cache_data(
    data={'user_id': '123', 'preferences': {'theme': 'dark'}},
    prefix='user_data',
    identifier='user_123',
    expire=3600  # 1 hour
)

# Get cached data
cached_data = redis_utils.get_cached_data(
    prefix='user_data',
    identifier='user_123'
)

# Cache batch data
redis_utils.cache_batch(
    data_dict={
        'user_123': {'name': 'John'},
        'user_456': {'name': 'Jane'}
    },
    prefix='user_data',
    expire=3600
)

# Get cached batch data
cached_batch = redis_utils.get_cached_batch(
    prefix='user_data',
    identifiers=['user_123', 'user_456']
)

# Delete batch keys
redis_utils.delete_batch(
    prefix='user_data',
    identifiers=['user_123', 'user_456']
)

# Scan keys
keys = redis_utils.scan_keys(
    prefix='user_data',
    pattern='user_*'
)

# Get memory usage
memory_usage = redis_utils.get_memory_usage()

# Get Redis stats
stats = redis_utils.get_stats()
""" 