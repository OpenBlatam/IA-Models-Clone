"""
Cache Service
=============

Service for caching workflow results and intermediate data.
"""

import logging
import hashlib
import json
import time
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import OrderedDict

logger = logging.getLogger(__name__)

# Constants
DEFAULT_CACHE_SIZE = 1000
DEFAULT_TTL_SECONDS = 3600  # 1 hour
MAX_CACHE_SIZE = 10000
MIN_TTL_SECONDS = 60  # 1 minute


@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    key: str
    value: Dict[str, Any]
    created_at: datetime
    expires_at: datetime
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    
    def is_expired(self) -> bool:
        """Check if cache entry is expired"""
        return datetime.now() > self.expires_at
    
    def touch(self) -> None:
        """Update access metadata"""
        self.access_count += 1
        self.last_accessed = datetime.now()


class CacheService:
    """
    Service for caching workflow results and intermediate data.
    
    Features:
    - LRU (Least Recently Used) eviction
    - TTL (Time To Live) support
    - Size limits
    - Access tracking
    """
    
    def __init__(
        self,
        max_size: int = DEFAULT_CACHE_SIZE,
        default_ttl: float = DEFAULT_TTL_SECONDS
    ):
        """
        Initialize cache service.
        
        Args:
            max_size: Maximum number of cache entries (default: 1000)
            default_ttl: Default TTL in seconds (default: 3600)
        """
        self.max_size = min(max_size, MAX_CACHE_SIZE)
        self.default_ttl = max(default_ttl, MIN_TTL_SECONDS)
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.hits = 0
        self.misses = 0
    
    def _generate_key(
        self,
        operation_type: str,
        image_url: str,
        **kwargs
    ) -> str:
        """
        Generate cache key from parameters.
        
        Args:
            operation_type: Type of operation
            image_url: Image URL
            **kwargs: Additional parameters
            
        Returns:
            Cache key string
        """
        # Create a deterministic key from parameters
        key_data = {
            "operation_type": operation_type,
            "image_url": image_url,
            **kwargs
        }
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_string.encode()).hexdigest()
    
    def get(
        self,
        operation_type: str,
        image_url: str,
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        """
        Get value from cache.
        
        Args:
            operation_type: Type of operation
            image_url: Image URL
            **kwargs: Additional parameters for key generation
            
        Returns:
            Cached value if found and not expired, None otherwise
        """
        key = self._generate_key(operation_type, image_url, **kwargs)
        
        if key not in self.cache:
            self.misses += 1
            return None
        
        entry = self.cache[key]
        
        # Check if expired
        if entry.is_expired():
            del self.cache[key]
            self.misses += 1
            logger.debug(f"Cache entry expired: {key[:16]}...")
            return None
        
        # Update access metadata and move to end (LRU)
        entry.touch()
        self.cache.move_to_end(key)
        self.hits += 1
        
        logger.debug(f"Cache hit: {key[:16]}...")
        return entry.value
    
    def set(
        self,
        operation_type: str,
        image_url: str,
        value: Dict[str, Any],
        ttl: Optional[float] = None,
        **kwargs
    ) -> None:
        """
        Set value in cache.
        
        Args:
            operation_type: Type of operation
            image_url: Image URL
            value: Value to cache
            ttl: Time to live in seconds (default: from config)
            **kwargs: Additional parameters for key generation
        """
        key = self._generate_key(operation_type, image_url, **kwargs)
        ttl = ttl or self.default_ttl
        
        # Remove expired entries first
        self._cleanup_expired()
        
        # Evict if at capacity
        if len(self.cache) >= self.max_size:
            # Remove least recently used (first item)
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
            logger.debug(f"Cache evicted LRU entry: {oldest_key[:16]}...")
        
        # Create new entry
        now = datetime.now()
        entry = CacheEntry(
            key=key,
            value=value,
            created_at=now,
            expires_at=now + timedelta(seconds=ttl),
            access_count=0,
            last_accessed=None
        )
        
        self.cache[key] = entry
        logger.debug(f"Cache set: {key[:16]}... (TTL: {ttl}s)")
    
    def delete(
        self,
        operation_type: str,
        image_url: str,
        **kwargs
    ) -> bool:
        """
        Delete value from cache.
        
        Args:
            operation_type: Type of operation
            image_url: Image URL
            **kwargs: Additional parameters for key generation
            
        Returns:
            True if entry was deleted, False if not found
        """
        key = self._generate_key(operation_type, image_url, **kwargs)
        
        if key in self.cache:
            del self.cache[key]
            logger.debug(f"Cache deleted: {key[:16]}...")
            return True
        
        return False
    
    def clear(self) -> int:
        """
        Clear all cache entries.
        
        Returns:
            Number of entries cleared
        """
        count = len(self.cache)
        self.cache.clear()
        logger.info(f"Cache cleared: {count} entries removed")
        return count
    
    def _cleanup_expired(self) -> int:
        """
        Remove expired entries from cache.
        
        Returns:
            Number of entries removed
        """
        expired_keys = [
            key for key, entry in self.cache.items()
            if entry.is_expired()
        ]
        
        for key in expired_keys:
            del self.cache[key]
        
        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
        
        return len(expired_keys)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics:
            - size: int - Current cache size
            - max_size: int - Maximum cache size
            - hits: int - Number of cache hits
            - misses: int - Number of cache misses
            - hit_rate: float - Hit rate percentage
            - entries: List[Dict] - Cache entry metadata
        """
        self._cleanup_expired()
        
        total_requests = self.hits + self.misses
        hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0.0
        
        entries = []
        for key, entry in self.cache.items():
            entries.append({
                "key": key[:16] + "...",  # Truncate for privacy
                "created_at": entry.created_at.isoformat(),
                "expires_at": entry.expires_at.isoformat(),
                "access_count": entry.access_count,
                "last_accessed": entry.last_accessed.isoformat() if entry.last_accessed else None,
                "is_expired": entry.is_expired()
            })
        
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(hit_rate, 2),
            "entries": entries
        }
    
    def reset_stats(self) -> None:
        """Reset cache statistics (hits/misses)"""
        self.hits = 0
        self.misses = 0
        logger.debug("Cache statistics reset")


# Global cache service instance
_cache_service: Optional[CacheService] = None


def get_cache_service() -> CacheService:
    """Get or create cache service instance"""
    global _cache_service
    if _cache_service is None:
        _cache_service = CacheService()
    return _cache_service
