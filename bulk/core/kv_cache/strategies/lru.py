"""
LRU (Least Recently Used) eviction strategy.
"""
from typing import List, Dict, Tuple
import torch

from kv_cache.strategies.base import BaseEvictionStrategy


class LRUEvictionStrategy(BaseEvictionStrategy):
    """Evict least recently used entries."""
    
    def select_eviction_candidates(
        self,
        cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
        access_times: Dict[int, float],
        access_counts: Dict[int, int],
        num_to_evict: int
    ) -> List[int]:
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

