"""
LFU (Least Frequently Used) eviction strategy.
"""
from __future__ import annotations

from kv_cache.strategies.base import BaseEvictionStrategy
from kv_cache.types import CacheDict, AccessTimesDict, AccessCountsDict


class LFUEvictionStrategy(BaseEvictionStrategy):
    """Evict least frequently used entries."""
    
    def select_eviction_candidates(
        self,
        cache: CacheDict,
        access_times: AccessTimesDict,
        access_counts: AccessCountsDict,
        num_to_evict: int
    ) -> list[int]:
        """Select least frequently used entries."""
        if not access_counts:
            # Fallback: evict first entries
            return list(cache.keys())[:num_to_evict]
        
        # Sort by access count (lowest first)
        sorted_positions = sorted(
            access_counts.items(),
            key=lambda x: x[1]
        )
        
        return [pos for pos, _ in sorted_positions[:num_to_evict]]

