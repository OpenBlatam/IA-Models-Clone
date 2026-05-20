from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

from typing import Optional, Dict, Any
from ..domain.repositories import ICacheRepository
from ..domain.entities import CaptionResponse
import asyncio
import time
import json
    from cachetools import TTLCache, LRUCache
        from ..domain.entities import (
from typing import Any, List, Dict, Optional
import logging
"""
Instagram Captions API v13.0 - Cache Repository Implementation

Infrastructure layer implementation of cache repository.
"""


try:
    ADVANCED_CACHE = True
except ImportError:
    ADVANCED_CACHE = False


class InMemoryCacheRepository(ICacheRepository):
    """In-memory cache repository implementation."""
    
    def __init__(self, max_size: int = 10000, default_ttl: int = 3600):
        
    """__init__ function."""
self.max_size = max_size
        self.default_ttl = default_ttl
        
        if ADVANCED_CACHE:
            self.cache = TTLCache(maxsize=max_size, ttl=default_ttl)
        else:
            self.cache = {}
            
        self.stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "deletes": 0
        }
    
    async def get(self, key: str) -> Optional[CaptionResponse]:
        """Get cached response."""
        try:
            if key in self.cache:
                self.stats["hits"] += 1
                cached_data = self.cache[key]
                
                # Convert back to CaptionResponse if needed
                if isinstance(cached_data, dict):
                    # Reconstruct from dict
                    return self._dict_to_response(cached_data)
                return cached_data
            else:
                self.stats["misses"] += 1
                return None
        except Exception:
            self.stats["misses"] += 1
            return None
    
    async def set(self, key: str, response: CaptionResponse, ttl: Optional[int] = None) -> None:
        """Set cached response."""
        try:
            # Convert to dict for storage
            cache_data = response.to_dict()
            cache_data["_cached_at"] = time.time()
            
            if ADVANCED_CACHE:
                self.cache[key] = cache_data
            else:
                # Simple dict cache with manual TTL
                self.cache[key] = {
                    "data": cache_data,
                    "expires_at": time.time() + (ttl or self.default_ttl)
                }
            
            self.stats["sets"] += 1
        except Exception:
            pass  # Fail silently for cache operations
    
    async def delete(self, key: str) -> bool:
        """Delete cached response."""
        try:
            if key in self.cache:
                del self.cache[key]
                self.stats["deletes"] += 1
                return True
            return False
        except Exception:
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        return key in self.cache
    
    async def clear(self) -> None:
        """Clear all cached data."""
        self.cache.clear()
        
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.stats["hits"] + self.stats["misses"]
        hit_rate = self.stats["hits"] / max(total_requests, 1)
        
        return {
            "hits": self.stats["hits"],
            "misses": self.stats["misses"],
            "hit_rate": hit_rate,
            "total_items": len(self.cache),
            "max_size": self.max_size,
            "cache_type": "TTLCache" if ADVANCED_CACHE else "SimpleDict"
        }
    
    def _dict_to_response(self, data: Dict[str, Any]) -> CaptionResponse:
        """Convert dict back to CaptionResponse."""
        # Simplified conversion - in real implementation would be more robust
            RequestId, Hashtags, QualityMetrics, PerformanceMetrics
        )
        
        return CaptionResponse(
            request_id=RequestId(data["request_id"]),
            caption=data["caption"],
            hashtags=Hashtags(data["hashtags"]),
            quality_metrics=QualityMetrics(
                score=data["quality_score"],
                engagement_prediction=data["engagement_prediction"],
                virality_score=data["virality_score"],
                readability_score=data.get("readability_score", 75.0)
            ),
            performance_metrics=PerformanceMetrics(
                processing_time=data["processing_time"],
                cache_hit=data["cache_hit"],
                provider_used=data.get("provider_used", "unknown"),
                model_used=data.get("model_used", "unknown"),
                confidence_score=data.get("confidence_score", 0.8)
            )
        ) 