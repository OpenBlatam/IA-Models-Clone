"""
Intelligent Cache
================

AI-powered intelligent caching with predictive prefetching.
"""

import logging
import time
import asyncio
from typing import Dict, Any, Optional, List, Callable
from collections import defaultdict, deque
from aws.modules.ports.cache_port import CachePort

logger = logging.getLogger(__name__)


class IntelligentCache:
    """Intelligent cache with predictive prefetching."""
    
    def __init__(self, cache: CachePort, prediction_window: int = 5):
        self.cache = cache
        self.prediction_window = prediction_window
        self._access_patterns: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self._access_times: Dict[str, List[float]] = defaultdict(list)
        self._correlations: Dict[str, List[str]] = defaultdict(list)
        self._prefetch_queue: List[str] = []
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value with access pattern tracking."""
        value = await self.cache.get(key)
        
        # Track access pattern
        current_time = time.time()
        self._access_patterns[key].append(current_time)
        self._access_times[key].append(current_time)
        
        # Keep only recent access times
        if len(self._access_times[key]) > 100:
            self._access_times[key] = self._access_times[key][-50:]
        
        # Predict next access
        if value is not None:
            await self._predict_and_prefetch(key)
        
        return value
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set value in cache."""
        await self.cache.set(key, value, ttl)
    
    async def _predict_and_prefetch(self, current_key: str):
        """Predict and prefetch likely next keys."""
        # Find correlated keys (accessed together)
        correlated = self._find_correlated_keys(current_key)
        
        # Prefetch correlated keys
        for key in correlated[:self.prediction_window]:
            if key not in self._prefetch_queue:
                self._prefetch_queue.append(key)
                asyncio.create_task(self._prefetch_key(key))
    
    def _find_correlated_keys(self, key: str) -> List[str]:
        """Find keys that are frequently accessed together."""
        # Simple correlation: keys accessed within short time window
        if key not in self._access_times:
            return []
        
        current_accesses = self._access_times[key]
        if len(current_accesses) < 2:
            return []
        
        # Find keys accessed within 1 second of this key
        correlated = []
        for other_key, other_accesses in self._access_times.items():
            if other_key == key:
                continue
            
            # Count co-occurrences
            co_occurrences = 0
            for access_time in current_accesses[-10:]:  # Last 10 accesses
                for other_time in other_accesses[-10:]:
                    if abs(access_time - other_time) < 1.0:  # Within 1 second
                        co_occurrences += 1
                        break
            
            if co_occurrences >= 2:
                correlated.append((other_key, co_occurrences))
        
        # Sort by co-occurrences
        correlated.sort(key=lambda x: x[1], reverse=True)
        return [key for key, _ in correlated]
    
    async def _prefetch_key(self, key: str):
        """Prefetch key if not in cache."""
        cached = await self.cache.get(key)
        if cached is None:
            # In production, trigger prefetch logic
            logger.debug(f"Prefetching key: {key}")
    
    def get_access_stats(self) -> Dict[str, Any]:
        """Get access pattern statistics."""
        return {
            "tracked_keys": len(self._access_patterns),
            "total_accesses": sum(len(pattern) for pattern in self._access_patterns.values()),
            "prefetch_queue_size": len(self._prefetch_queue),
            "most_accessed": sorted(
                [
                    (key, len(pattern))
                    for key, pattern in self._access_patterns.items()
                ],
                key=lambda x: x[1],
                reverse=True
            )[:10]
        }

