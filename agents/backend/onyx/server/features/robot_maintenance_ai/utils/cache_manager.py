"""
Cache manager for API responses and ML predictions.
"""

import hashlib
import logging
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from collections import OrderedDict

from .json_helpers import safe_json_dumps

logger = logging.getLogger(__name__)


class CacheManager:
    """
    Simple in-memory cache with TTL support.
    """
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        """
        Initialize cache manager.
        
        Args:
            max_size: Maximum number of cached items
            default_ttl: Default TTL in seconds
        """
        self.cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self.max_size = max_size
        self.default_ttl = default_ttl
    
    def _generate_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments."""
        key_data = safe_json_dumps({"args": args, "kwargs": kwargs})
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
        
        Returns:
            Cached value or None if not found/expired
        """
        if key not in self.cache:
            return None
        
        item = self.cache[key]
        expires_at = item.get("expires_at")
        
        if expires_at and datetime.now() > expires_at:
            del self.cache[key]
            return None
        
        self.cache.move_to_end(key)
        return item["value"]
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (uses default if None)
        """
        if len(self.cache) >= self.max_size:
            self.cache.popitem(last=False)
        
        ttl = ttl or self.default_ttl
        expires_at = datetime.now() + timedelta(seconds=ttl)
        
        self.cache[key] = {
            "value": value,
            "expires_at": expires_at,
            "created_at": datetime.now()
        }
        self.cache.move_to_end(key)
    
    def clear(self) -> None:
        """Clear all cached items."""
        self.cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache stats
        """
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "default_ttl": self.default_ttl
        }






