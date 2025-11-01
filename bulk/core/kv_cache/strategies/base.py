"""
Base eviction strategy interface.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple
import torch


class BaseEvictionStrategy(ABC):
    """Base class for cache eviction strategies."""
    
    @abstractmethod
    def select_eviction_candidates(
        self,
        cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
        access_times: Dict[int, float],
        access_counts: Dict[int, int],
        num_to_evict: int
    ) -> List[int]:
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

