"""
LFU (Least Frequently Used) eviction strategy.
"""
from typing import List, Dict, Tuple
import torch

from kv_cache.strategies.base import BaseEvictionStrategy


class LFUEvictionStrategy(BaseEvictionStrategy):
    """Evict least frequently used entries."""
    
    def select_eviction_candidates(
        self,
        cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
        access_times: Dict[int, float],
        access_counts: Dict[int, int],
        num_to_evict: int
    ) -> List[int]:
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

