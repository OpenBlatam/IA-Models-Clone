"""
Adaptive eviction strategy combining LRU and LFU.
"""
from __future__ import annotations

from kv_cache.strategies.base import BaseEvictionStrategy
from kv_cache.types import CacheDict, AccessTimesDict, AccessCountsDict


class AdaptiveEvictionStrategy(BaseEvictionStrategy):
    """
    Adaptive eviction strategy combining LRU and LFU scores.
    
    Uses weighted combination of recency and frequency for better
    cache performance in diverse workloads.
    """
    
    def __init__(self, recency_weight: float = 0.5, frequency_weight: float = 0.5):
        """
        Initialize adaptive strategy.
        
        Args:
            recency_weight: Weight for recency (LRU) component
            frequency_weight: Weight for frequency (LFU) component
        """
        if abs(recency_weight + frequency_weight - 1.0) > 1e-6:
            raise ValueError("Weights must sum to 1.0")
        
        self.recency_weight = recency_weight
        self.frequency_weight = frequency_weight
    
    def select_eviction_candidates(
        self,
        cache: CacheDict,
        access_times: AccessTimesDict,
        access_counts: AccessCountsDict,
        num_to_evict: int
    ) -> list[int]:
        """Select entries using adaptive scoring."""
        if not access_times and not access_counts:
            # Fallback: evict first entries
            return list(cache.keys())[:num_to_evict]
        
        # Normalize scores
        max_time = max(access_times.values()) if access_times else 1.0
        max_count = max(access_counts.values()) if access_counts else 1
        
        # Calculate adaptive scores
        scores = {}
        for pos in cache.keys():
            time_score = access_times.get(pos, 0) / max_time if max_time > 0 else 0
            count_score = access_counts.get(pos, 0) / max_count if max_count > 0 else 0
            
            # Lower score = more likely to evict
            scores[pos] = (
                self.recency_weight * time_score +
                self.frequency_weight * count_score
            )
        
        # Sort by score (lowest = evict first)
        sorted_positions = sorted(
            scores.items(),
            key=lambda x: x[1]
        )
        
        return [pos for pos, _ in sorted_positions[:num_to_evict]]

