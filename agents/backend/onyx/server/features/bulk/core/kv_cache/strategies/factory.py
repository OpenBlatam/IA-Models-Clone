"""
Factory for eviction strategies.

Provides easy creation of eviction strategies following factory pattern.
Uses registry pattern for extensibility.
"""
from __future__ import annotations

from kv_cache.config import CacheStrategy
from kv_cache.strategies.base import BaseEvictionStrategy
from kv_cache.strategies.registry import (
    _STRATEGY_REGISTRY,
    is_strategy_registered,
    get_registered_strategies
)
from kv_cache.constants import DEFAULT_ADAPTIVE_RECENCY_WEIGHT, DEFAULT_ADAPTIVE_FREQUENCY_WEIGHT
from kv_cache.exceptions import CacheStrategyError


def create_eviction_strategy(
    strategy: CacheStrategy,
    **kwargs: float
) -> BaseEvictionStrategy:
    """
    Create eviction strategy based on configuration.
    
    Uses registry pattern for extensible strategy creation.
    
    Args:
        strategy: Cache strategy enum
        **kwargs: Additional arguments for strategy initialization
            - recency_weight: For AdaptiveEvictionStrategy
            - frequency_weight: For AdaptiveEvictionStrategy
        
    Returns:
        Eviction strategy instance
        
    Raises:
        CacheStrategyError: If strategy is not registered or unknown
    """
    # Check if strategy is registered
    if not is_strategy_registered(strategy):
        available = [s.value for s in get_registered_strategies()]
        raise CacheStrategyError(
            f"Strategy {strategy.value} is not registered. "
            f"Available strategies: {available}"
        )
    
    # Get strategy class from registry
    strategy_class = _STRATEGY_REGISTRY[strategy]
    
    # Create instance with kwargs if needed
    if strategy == CacheStrategy.ADAPTIVE:
        recency_weight = kwargs.get("recency_weight", DEFAULT_ADAPTIVE_RECENCY_WEIGHT)
        frequency_weight = kwargs.get("frequency_weight", DEFAULT_ADAPTIVE_FREQUENCY_WEIGHT)
        return strategy_class(
            recency_weight=recency_weight,
            frequency_weight=frequency_weight
        )
    else:
        # Simple strategies don't need kwargs
        return strategy_class()

