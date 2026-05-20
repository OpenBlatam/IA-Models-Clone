"""
Cache Manager for Contabilidad Mexicana AI
===========================================

Intelligent caching system for API responses to reduce costs and improve performance.
"""

import hashlib
import json
import logging
import time
from typing import Dict, Any, Optional
from collections import OrderedDict

logger = logging.getLogger(__name__)


class ResponseCache:
    """LRU cache for API responses with TTL support."""
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        """
        Initialize response cache.
        
        Args:
            max_size: Maximum number of cached items
            default_ttl: Default time-to-live in seconds
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: OrderedDict = OrderedDict()
        self.timestamps: Dict[str, float] = {}
        self.hit_count = 0
        self.miss_count = 0
    
    def _generate_key(self, service_name: str, params: Dict[str, Any]) -> str:
        """Generate cache key from service name and parameters."""
        # Sort keys for consistent hashing
        params_str = json.dumps(params, sort_keys=True, default=str)
        content = f"{service_name}:{params_str}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get(
        self,
        service_name: str,
        params: Dict[str, Any],
        ttl: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached response.
        
        Args:
            service_name: Name of the service
            params: Service parameters
            ttl: Optional TTL override
        
        Returns:
            Cached response or None
        """
        key = self._generate_key(service_name, params)
        ttl = ttl or self.default_ttl
        
        if key not in self.cache:
            self.miss_count += 1
            return None
        
        # Check TTL
        timestamp = self.timestamps.get(key, 0)
        if time.time() - timestamp > ttl:
            # Expired, remove from cache
            del self.cache[key]
            del self.timestamps[key]
            self.miss_count += 1
            return None
        
        # Move to end (LRU)
        result = self.cache.pop(key)
        self.cache[key] = result
        self.hit_count += 1
        
        logger.debug(f"Cache hit for {service_name}")
        
        # Notify metrics if available
        try:
            from .metrics_collector import MetricsCollector
            # This will be set by ContadorAI if metrics are available
            pass
        except ImportError:
            pass
        
        return result
    
    def put(
        self,
        service_name: str,
        params: Dict[str, Any],
        response: Dict[str, Any],
        ttl: Optional[int] = None
    ) -> None:
        """
        Store response in cache.
        
        Args:
            service_name: Name of the service
            params: Service parameters
            response: Response to cache
            ttl: Optional TTL override
        """
        key = self._generate_key(service_name, params)
        
        # Evict if cache is full
        if len(self.cache) >= self.max_size:
            # Remove oldest (first) item
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
            del self.timestamps[oldest_key]
        
        # Store response
        self.cache[key] = response
        self.timestamps[key] = time.time()
        
        logger.debug(f"Cached response for {service_name}")
    
    def clear(self) -> None:
        """Clear all cached items."""
        self.cache.clear()
        self.timestamps.clear()
        self.hit_count = 0
        self.miss_count = 0
        logger.info("Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.hit_count + self.miss_count
        hit_rate = (self.hit_count / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hit_count,
            "misses": self.miss_count,
            "hit_rate": f"{hit_rate:.2f}%",
            "total_requests": total_requests
        }
