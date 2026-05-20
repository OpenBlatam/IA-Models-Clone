"""
Base eviction strategy interface.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from kv_cache.types import CacheDict, AccessTimesDict, AccessCountsDict


class BaseEvictionStrategy(ABC):
    """Base class for cache eviction strategies."""
    
    @abstractmethod
    def select_eviction_candidates(
        self,
        cache: CacheDict,
        access_times: AccessTimesDict,
        access_counts: AccessCountsDict,
        num_to_evict: int
    ) -> list[int]:
        """
        Select cache entries to evict.
        
        Args:
            cache: Cache dictionary
            access_times: Access time for each position
            access_counts: Access count for each position
            num_to_evict: Number of entries to evict
            
        Returns:
            List of positions to evict
        """
        pass

