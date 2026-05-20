"""
Cache warmup strategies.

Provides different strategies for pre-warming the cache.
"""
from __future__ import annotations

import logging
from typing import Callable, Any, List, Optional
from enum import Enum

import torch

from kv_cache.types import TensorPair

logger = logging.getLogger(__name__)


class WarmupStrategy(Enum):
    """Cache warmup strategies."""
    SEQUENTIAL = "sequential"
    RANDOM = "random"
    FREQUENCY_BASED = "frequency_based"
    PREDICTIVE = "predictive"
    ADAPTIVE = "adaptive"


class CacheWarmup:
    """
    Cache warmup manager.
    
    Provides different strategies for pre-warming the cache.
    """
    
    def __init__(self, cache: Any):
        """
        Initialize cache warmup.
        
        Args:
            cache: Cache instance
        """
        self.cache = cache
        self.warmup_history: List[int] = []
    
    def warmup_sequential(
        self,
        positions: List[int],
        compute_fn: Callable[[int], TensorPair],
        batch_size: int = 10
    ) -> int:
        """
        Warmup cache sequentially.
        
        Args:
            positions: List of positions to warm
            compute_fn: Function to compute values
            batch_size: Batch size for processing
            
        Returns:
            Number of entries warmed
        """
        warmed = 0
        for i in range(0, len(positions), batch_size):
            batch = positions[i:i + batch_size]
            for pos in batch:
                if self.cache.get(pos) is None:
                    try:
                        key, value = compute_fn(pos)
                        self.cache.put(pos, key, value)
                        warmed += 1
                        self.warmup_history.append(pos)
                    except Exception as e:
                        logger.warning(f"Failed to warm position {pos}: {e}")
        
        logger.info(f"Warmed {warmed} entries sequentially")
        return warmed
    
    def warmup_random(
        self,
        positions: List[int],
        compute_fn: Callable[[int], TensorPair],
        num_samples: Optional[int] = None
    ) -> int:
        """
        Warmup cache with random sampling.
        
        Args:
            positions: List of positions to warm
            compute_fn: Function to compute values
            num_samples: Number of random samples (None = all)
            
        Returns:
            Number of entries warmed
        """
        import random
        
        if num_samples is None:
            num_samples = len(positions)
        
        sampled = random.sample(positions, min(num_samples, len(positions)))
        return self.warmup_sequential(sampled, compute_fn)
    
    def warmup_frequency_based(
        self,
        positions: List[int],
        compute_fn: Callable[[int], TensorPair],
        frequency_map: dict[int, int],
        top_k: int = 100
    ) -> int:
        """
        Warmup cache based on access frequency.
        
        Args:
            positions: List of positions to warm
            compute_fn: Function to compute values
            frequency_map: Map of position -> access frequency
            top_k: Number of top positions to warm
            
        Returns:
            Number of entries warmed
        """
        # Sort by frequency
        sorted_positions = sorted(
            positions,
            key=lambda p: frequency_map.get(p, 0),
            reverse=True
        )[:top_k]
        
        return self.warmup_sequential(sorted_positions, compute_fn)
    
    def warmup_predictive(
        self,
        positions: List[int],
        compute_fn: Callable[[int], TensorPair],
        predictor: Callable[[int], List[int]],
        depth: int = 2
    ) -> int:
        """
        Warmup cache using predictive strategy.
        
        Args:
            positions: Initial positions to warm
            compute_fn: Function to compute values
            predictor: Function that predicts related positions
            depth: Prediction depth
            
        Returns:
            Number of entries warmed
        """
        warmed = 0
        to_warm = set(positions)
        visited = set()
        
        for _ in range(depth):
            current_batch = list(to_warm - visited)
            for pos in current_batch:
                if self.cache.get(pos) is None:
                    try:
                        key, value = compute_fn(pos)
                        self.cache.put(pos, key, value)
                        warmed += 1
                        self.warmup_history.append(pos)
                    except Exception as e:
                        logger.warning(f"Failed to warm position {pos}: {e}")
                
                visited.add(pos)
                
                # Predict related positions
                predicted = predictor(pos)
                to_warm.update(predicted)
        
        logger.info(f"Warmed {warmed} entries predictively")
        return warmed
    
    def warmup_adaptive(
        self,
        positions: List[int],
        compute_fn: Callable[[int], TensorPair],
        strategy: WarmupStrategy = WarmupStrategy.SEQUENTIAL
    ) -> int:
        """
        Adaptive warmup that chooses strategy based on cache state.
        
        Args:
            positions: List of positions to warm
            compute_fn: Function to compute values
            strategy: Preferred strategy
            
        Returns:
            Number of entries warmed
        """
        stats = self.cache.get_stats()
        cache_size = stats.get("num_entries", 0)
        max_tokens = stats.get("max_tokens", 0)
        
        # Choose strategy based on cache state
        if cache_size > max_tokens * 0.8:
            # Cache nearly full, use frequency-based
            logger.info("Cache nearly full, using frequency-based warmup")
            # Fallback to sequential if no frequency data
            return self.warmup_sequential(positions, compute_fn)
        elif strategy == WarmupStrategy.FREQUENCY_BASED:
            # Use frequency if available
            return self.warmup_sequential(positions, compute_fn)
        elif strategy == WarmupStrategy.RANDOM:
            return self.warmup_random(positions, compute_fn)
        else:
            return self.warmup_sequential(positions, compute_fn)
    
    def get_warmup_stats(self) -> dict[str, Any]:
        """
        Get warmup statistics.
        
        Returns:
            Dictionary with warmup stats
        """
        return {
            "total_warmed": len(self.warmup_history),
            "unique_positions": len(set(self.warmup_history)),
            "warmup_history": self.warmup_history[-100:]  # Last 100
        }

