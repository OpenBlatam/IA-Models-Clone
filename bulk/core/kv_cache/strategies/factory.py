"""
Factory for eviction strategies.

Provides easy creation of eviction strategies following factory pattern.
"""
from typing import Optional

from kv_cache.config import CacheStrategy
from kv_cache.strategies.base import BaseEvictionStrategy
from kv_cache.strategies.lru import LRUEvictionStrategy
from kv_cache.strategies.lfu import LFUEvictionStrategy
from kv_cache.strategies.adaptive import AdaptiveEvictionStrategy


def create_eviction_strategy(
    strategy: CacheStrategy,
    **kwargs
) -> BaseEvictionStrategy:
    """
    Create eviction strategy based on configuration.
    
    Args:
        strategy: Cache strategy enum
        **kwargs: Additional arguments for strategy initialization
        
    Returns:
        Eviction strategy instance
        
    Raises:
        ValueError: If strategy is unknown
    """
    if strategy == CacheStrategy.LRU:
        return LRUEvictionStrategy()
    elif strategy == CacheStrategy.LFU:
        return LFUEvictionStrategy()
    elif strategy == CacheStrategy.ADAPTIVE:
        recency_weight = kwargs.get("recency_weight", 0.5)
        frequency_weight = kwargs.get("frequency_weight", 0.5)
        return AdaptiveEvictionStrategy(
            recency_weight=recency_weight,
            frequency_weight=frequency_weight
        )
    else:
        # Default to LRU
        return LRUEvictionStrategy()

