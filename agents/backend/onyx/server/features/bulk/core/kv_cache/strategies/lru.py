"""
LRU (Least Recently Used) eviction strategy.
"""
from __future__ import annotations

from kv_cache.strategies.base import BaseEvictionStrategy
from kv_cache.types import CacheDict, AccessTimesDict, AccessCountsDict


class LRUEvictionStrategy(BaseEvictionStrategy):
    """Evict least recently used entries."""
    
    def select_eviction_candidates(
        self,
        cache: CacheDict,
        access_times: AccessTimesDict,
        access_counts: AccessCountsDict,
        num_to_evict: int
    ) -> list[int]:
        """Select least recently used entries."""
        if not access_times:
            # Fallback: evict first entries
            return list(cache.keys())[:num_to_evict]
        
        # Sort by access time (oldest first)
        sorted_positions = sorted(
            access_times.items(),
            key=lambda x: x[1]
        )
        
        return [pos for pos, _ in sorted_positions[:num_to_evict]]

