"""
Registry for eviction strategies.

Provides centralized registration and discovery of strategies.
"""
from __future__ import annotations

from typing import Type

from kv_cache.config import CacheStrategy
from kv_cache.strategies.base import BaseEvictionStrategy

# Registry mapping strategy to implementation class
_STRATEGY_REGISTRY: dict[CacheStrategy, Type[BaseEvictionStrategy]] = {}


def register_strategy(
    strategy: CacheStrategy,
    strategy_class: Type[BaseEvictionStrategy]
) -> None:
    """
    Register an eviction strategy.
    
    Args:
        strategy: Cache strategy enum
        strategy_class: Strategy implementation class
    """
    _STRATEGY_REGISTRY[strategy] = strategy_class


def get_registered_strategies() -> list[CacheStrategy]:
    """
    Get list of registered strategies.
    
    Returns:
        List of registered cache strategies
    """
    return list(_STRATEGY_REGISTRY.keys())


def is_strategy_registered(strategy: CacheStrategy) -> bool:
    """
    Check if strategy is registered.
    
    Args:
        strategy: Cache strategy to check
        
    Returns:
        True if strategy is registered
    """
    return strategy in _STRATEGY_REGISTRY


# Auto-register built-in strategies
def _auto_register() -> None:
    """Auto-register built-in strategies."""
    from kv_cache.strategies.lru import LRUEvictionStrategy
    from kv_cache.strategies.lfu import LFUEvictionStrategy
    from kv_cache.strategies.adaptive import AdaptiveEvictionStrategy
    
    register_strategy(CacheStrategy.LRU, LRUEvictionStrategy)
    register_strategy(CacheStrategy.LFU, LFUEvictionStrategy)
    register_strategy(CacheStrategy.ADAPTIVE, AdaptiveEvictionStrategy)


# Initialize registry on import
_auto_register()



