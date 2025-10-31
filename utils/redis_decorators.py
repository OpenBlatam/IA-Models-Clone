from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
TIMEOUT_SECONDS = 60

from typing import Any, Callable, Dict, List, Optional, Union, TypeVar, Generic
from functools import wraps
import inspect
import hashlib
import json
import time
import logging
from .redis_utils import RedisUtils
from .redis_config import get_config
from .redis_decorators import redis_decorators
from typing import Any, List, Dict, Optional
import asyncio
"""
Redis Decorators - Onyx Integration
Decorators for Redis caching in Onyx.
"""

logger = logging.getLogger(__name__)

T = TypeVar('T')
F = TypeVar('F', bound=Callable[..., Any])

class RedisDecorators:
    """Decorators for Redis caching in Onyx."""
    
    def __init__(self, config: Any = None):
        """Initialize Redis decorators."""
        self.config = config or {}
        self.redis_utils = RedisUtils(get_config())
        # Support both dict and Pydantic model
        if isinstance(self.config, dict):
            self.default_ttl = self.config.get("default_ttl", 3600)
        else:
            self.default_ttl = getattr(self.config, "default_ttl", 3600)
    
    def cache(
        self,
        prefix: str,
        ttl: Optional[int] = None,
        key_builder: Optional[Callable[..., str]] = None
    ) -> Callable[[F], F]:
        """Cache function results in Redis."""
        def decorator(func: F) -> F:
            @wraps(func)
            async def wrapper(*args: Any, **kwargs: Any) -> Any:
                # Generate cache key
                if key_builder:
                    cache_key = key_builder(*args, **kwargs)
                else:
                    cache_key = self._generate_cache_key(func, args, kwargs)
                
                # Try to get cached result
                cached_result = self.redis_utils.get_cached_data(
                    prefix=prefix,
                    identifier=cache_key
                )
                
                if cached_result is not None:
                    logger.debug(f"Cache hit for {func.__name__}")
                    return cached_result
                
                # Get fresh result
                result = await func(*args, **kwargs)
                
                # Cache result
                self.redis_utils.cache_data(
                    data=result,
                    prefix=prefix,
                    identifier=cache_key,
                    expire=ttl or self.default_ttl
                )
                
                logger.debug(f"Cached result for {func.__name__}")
                return result
            
            return wrapper
        return decorator
    
    def cache_model(
        self,
        prefix: str,
        ttl: Optional[int] = None,
        key_builder: Optional[Callable[..., str]] = None
    ) -> Callable[[F], F]:
        """Cache model results in Redis."""
        def decorator(func: F) -> F:
            @wraps(func)
            async def wrapper(*args: Any, **kwargs: Any) -> Any:
                # Generate cache key
                if key_builder:
                    cache_key = key_builder(*args, **kwargs)
                else:
                    cache_key = self._generate_cache_key(func, args, kwargs)
                
                # Try to get cached model
                cached_model = self.redis_utils.get_cached_data(
                    prefix=prefix,
                    identifier=cache_key,
                    model_class=inspect.signature(func).return_annotation
                )
                
                if cached_model is not None:
                    logger.debug(f"Cache hit for {func.__name__}")
                    return cached_model
                
                # Get fresh model
                model = await func(*args, **kwargs)
                
                # Cache model
                self.redis_utils.cache_data(
                    data=model,
                    prefix=prefix,
                    identifier=cache_key,
                    expire=ttl or self.default_ttl
                )
                
                logger.debug(f"Cached model for {func.__name__}")
                return model
            
            return wrapper
        return decorator
    
    def cache_batch(
        self,
        prefix: str,
        ttl: Optional[int] = None,
        key_builder: Optional[Callable[..., List[str]]] = None
    ) -> Callable[[F], F]:
        """Cache batch function results in Redis."""
        def decorator(func: F) -> F:
            @wraps(func)
            async def wrapper(*args: Any, **kwargs: Any) -> Any:
                # Generate cache keys
                if key_builder:
                    cache_keys = key_builder(*args, **kwargs)
                else:
                    cache_keys = self._generate_batch_cache_keys(func, args, kwargs)
                
                # Try to get cached results
                cached_results = self.redis_utils.get_cached_batch(
                    prefix=prefix,
                    identifiers=cache_keys
                )
                
                if cached_results:
                    logger.debug(f"Cache hit for {func.__name__}")
                    return cached_results
                
                # Get fresh results
                results = await func(*args, **kwargs)
                
                # Cache results
                self.redis_utils.cache_batch(
                    data_dict=results,
                    prefix=prefix,
                    expire=ttl or self.default_ttl
                )
                
                logger.debug(f"Cached batch results for {func.__name__}")
                return results
            
            return wrapper
        return decorator
    
    def invalidate(
        self,
        prefix: str,
        key_builder: Optional[Callable[..., str]] = None
    ) -> Callable[[F], F]:
        """Invalidate cached data in Redis."""
        def decorator(func: F) -> F:
            @wraps(func)
            async def wrapper(*args: Any, **kwargs: Any) -> Any:
                # Generate cache key
                if key_builder:
                    cache_key = key_builder(*args, **kwargs)
                else:
                    cache_key = self._generate_cache_key(func, args, kwargs)
                
                # Execute function
                result = await func(*args, **kwargs)
                
                # Invalidate cache
                self.redis_utils.delete_key(
                    prefix=prefix,
                    identifier=cache_key
                )
                
                logger.debug(f"Invalidated cache for {func.__name__}")
                return result
            
            return wrapper
        return decorator
    
    def _generate_cache_key(self, func: Callable, args: tuple, kwargs: dict) -> str:
        """Generate a cache key for a function call."""
        # Get function signature
        sig = inspect.signature(func)
        
        # Bind arguments
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        
        # Create key parts
        key_parts = [
            func.__name__,
            func.__module__,
            str(bound_args.arguments)
        ]
        
        # Create hash
        key_string = "|".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _generate_batch_cache_keys(self, func: Callable, args: tuple, kwargs: dict) -> List[str]:
        """Generate cache keys for a batch function call."""
        # Get function signature
        sig = inspect.signature(func)
        
        # Bind arguments
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        
        # Get batch argument
        batch_arg = next(
            (arg for arg in bound_args.arguments.values() if isinstance(arg, (list, dict))),
            None
        )
        
        if batch_arg is None:
            raise ValueError("No batch argument found")
        
        # Generate keys for each item
        if isinstance(batch_arg, list):
            return [
                self._generate_cache_key(func, (item,), {})
                for item in batch_arg
            ]
        else:
            return [
                self._generate_cache_key(func, (key, value), {})
                for key, value in batch_arg.items()
            ]

# Global Redis decorators instance
redis_decorators = RedisDecorators()

# Example usage:
"""

# Cache function results
@redis_decorators.cache(
    prefix="function_results",
    ttl=3600  # 1 hour
)
async def get_user_data(user_id: str) -> Dict[str, Any]:
    # Function implementation
    return {"user_id": user_id, "data": {...}}

# Cache model results
@redis_decorators.cache_model(
    prefix="user_models",
    ttl=3600  # 1 hour
)
async def get_user_model(user_id: str) -> UserModel:
    # Function implementation
    return UserModel(...)

# Cache batch results
@redis_decorators.cache_batch(
    prefix="batch_results",
    ttl=3600  # 1 hour
)
async def get_batch_data(user_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    # Function implementation
    return {user_id: {...} for user_id in user_ids}

# Invalidate cache
@redis_decorators.invalidate(
    prefix="user_data"
)
async def update_user_data(user_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
    # Function implementation
    return {"user_id": user_id, "data": data}

# Custom key builder
def custom_key_builder(user_id: str, **kwargs) -> str:
    return f"user:{user_id}:{kwargs.get('version', 'v1')}"

@redis_decorators.cache(
    prefix="custom_keys",
    key_builder=custom_key_builder
)
async def get_custom_data(user_id: str, version: str = "v1") -> Dict[str, Any]:
    # Function implementation
    return {"user_id": user_id, "version": version, "data": {...}}
""" 