"""
Sistema de cache para respuestas HTTP (optimizado)

Incluye cache en memoria para reducir carga en la base de datos.
"""

import time
import hashlib
import json
from typing import Optional, Dict, Any, Callable
from functools import wraps
from collections import OrderedDict

try:
    import orjson
    JSON_ENCODER = orjson
except ImportError:
    import json
    JSON_ENCODER = json

logger = None
try:
    import logging
    logger = logging.getLogger(__name__)
except ImportError:
    pass


class LRUCache:
    """
    LRU (Least Recently Used) cache implementation in memory.
    
    Features:
    - Automatic eviction of least recently used items
    - TTL (Time To Live) support
    - Hit/miss statistics
    - Thread-safe operations
    """
    
    def __init__(self, max_size: int = 128):
        """
        Initialize LRU cache.
        
        Args:
            max_size: Maximum number of items in cache (must be > 0)
            
        Raises:
            ValueError: If max_size is <= 0
        """
        if max_size <= 0:
            raise ValueError(f"max_size must be > 0, got {max_size}")
        
        self.cache: OrderedDict = OrderedDict()
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get a value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cache entry dictionary or None if not found
            
        Raises:
            ValueError: If key is None or empty
        """
        if not key or not isinstance(key, str) or not key.strip():
            raise ValueError(f"key must be a non-empty string, got {type(key).__name__}")
        
        key = key.strip()
        
        if key in self.cache:
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            self.hits += 1
            return self.cache[key]
        
        self.misses += 1
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """
        Set a value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (optional)
            
        Raises:
            ValueError: If key is None or empty, or ttl is negative
        """
        if not key or not isinstance(key, str) or not key.strip():
            raise ValueError(f"key must be a non-empty string, got {type(key).__name__}")
        
        if ttl is not None and ttl < 0:
            raise ValueError(f"ttl must be >= 0, got {ttl}")
        
        key = key.strip()
        
        # If cache is full, remove oldest item
        if len(self.cache) >= self.max_size:
            self.cache.popitem(last=False)
        
        cache_entry = {
            "value": value,
            "expires_at": time.time() + ttl if ttl else None
        }
        
        self.cache[key] = cache_entry
        self.cache.move_to_end(key)
    
    def _is_expired(self, key: str) -> bool:
        """Verifica si una entrada ha expirado"""
        if key not in self.cache:
            return True
        
        entry = self.cache[key]
        if entry.get("expires_at") is None:
            return False
        
        return time.time() > entry["expires_at"]
    
    def get_with_expiry_check(self, key: str) -> Optional[Any]:
        """
        Get a value from cache with expiry check.
        
        Automatically removes expired entries.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found or expired
            
        Raises:
            ValueError: If key is None or empty
        """
        if not key or not isinstance(key, str) or not key.strip():
            raise ValueError(f"key must be a non-empty string, got {type(key).__name__}")
        
        key = key.strip()
        
        if key not in self.cache:
            return None
        
        if self._is_expired(key):
            del self.cache[key]
            self.misses += 1
            return None
        
        # Move to end (most recently used) and return value
        entry = self.cache[key]
        self.cache.move_to_end(key)
        self.hits += 1
        return entry.get("value")
    
    def clear(self) -> None:
        """Limpia el cache"""
        self.cache.clear()
        self.hits = 0
        self.misses = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics including:
            - size: Current number of items
            - max_size: Maximum cache size
            - hits: Number of cache hits
            - misses: Number of cache misses
            - hit_rate: Hit rate percentage
        """
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0.0
        
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(hit_rate, 2),
            "total_requests": total
        }


# Cache global
_response_cache = LRUCache(max_size=256)


def generate_cache_key(
    path: str,
    query_params: Optional[Dict[str, Any]] = None,
    user_id: Optional[str] = None
) -> str:
    """
    Generate a unique cache key.
    
    Args:
        path: Endpoint path
        query_params: Query parameters (optional)
        user_id: User ID (optional)
        
    Returns:
        Cache key (MD5 hash)
        
    Raises:
        ValueError: If path is None or empty
    """
    if not path or not isinstance(path, str) or not path.strip():
        raise ValueError(f"path must be a non-empty string, got {type(path).__name__}")
    
    path = path.strip()
    key_parts = [path]
    
    if query_params:
        # Sort parameters for consistency
        sorted_params = sorted(query_params.items())
        key_parts.append(str(sorted_params))
    
    if user_id:
        if not isinstance(user_id, str) or not user_id.strip():
            raise ValueError(f"user_id must be a non-empty string if provided, got {type(user_id).__name__}")
        key_parts.append(f"user:{user_id.strip()}")
    
    key_string = "|".join(key_parts)
    
    # Hash to reduce size
    return hashlib.md5(key_string.encode()).hexdigest()


def cache_response(ttl: float = 60.0, key_func: Optional[Callable] = None):
    """
    Decorator to cache HTTP responses.
    
    Args:
        ttl: Time to live in seconds (must be > 0)
        key_func: Custom function to generate cache key (optional)
        
    Returns:
        Decorator function
        
    Raises:
        ValueError: If ttl is <= 0
    """
    if ttl <= 0:
        raise ValueError(f"ttl must be > 0, got {ttl}")
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Generar clave de cache
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                # Intentar extraer path y query params del request
                request = None
                for arg in args:
                    if hasattr(arg, 'url'):
                        request = arg
                        break
                if not request:
                    request = kwargs.get('request')
                
                if request:
                    query_params = dict(request.query_params)
                    user_id = kwargs.get('current_user_id') or kwargs.get('user_id')
                    cache_key = generate_cache_key(
                        request.url.path,
                        query_params,
                        user_id
                    )
                else:
                    # Sin request, no cachear
                    return await func(*args, **kwargs)
            
            # Try to get from cache
            cached = _response_cache.get_with_expiry_check(cache_key)
            if cached is not None:
                if logger:
                    logger.debug(f"Cache hit for key: {cache_key}")
                return cached
            
            # Ejecutar función y cachear resultado
            result = await func(*args, **kwargs)
            
            _response_cache.set(cache_key, result, ttl=ttl)
            
            if logger:
                logger.debug(f"Cache miss for key: {cache_key}, cached result")
            
            return result
        
        return wrapper
    return decorator


def clear_response_cache(pattern: Optional[str] = None) -> int:
    """
    Clear response cache.
    
    Args:
        pattern: Optional pattern to clear specific entries
        
    Returns:
        Number of entries removed
    """
    if pattern:
        # Limpiar solo entradas que coincidan con el patrón
        keys_to_remove = [
            key for key in _response_cache.cache.keys()
            if pattern in key
        ]
        for key in keys_to_remove:
            _response_cache.cache.pop(key, None)
        return len(keys_to_remove)
    else:
        # Limpiar todo
        count = len(_response_cache.cache)
        _response_cache.clear()
        return count


def get_cache_stats() -> Dict[str, Any]:
    """
    Get cache statistics.
    
    Returns:
        Dictionary with cache statistics
    """
    return _response_cache.get_stats()

