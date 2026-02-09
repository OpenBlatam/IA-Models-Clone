"""
Edge Cache Strategy
==================

Edge caching strategies.
"""

import logging
import asyncio
from typing import Dict, Any, Optional, Callable
from enum import Enum

logger = logging.getLogger(__name__)


class CacheStrategy(Enum):
    """Cache strategies."""
    CACHE_FIRST = "cache_first"
    NETWORK_FIRST = "network_first"
    CACHE_ONLY = "cache_only"
    NETWORK_ONLY = "network_only"
    STALE_WHILE_REVALIDATE = "stale_while_revalidate"


class EdgeCacheStrategy:
    """Edge cache strategy manager."""
    
    def __init__(self, default_strategy: CacheStrategy = CacheStrategy.CACHE_FIRST):
        self.default_strategy = default_strategy
        self._strategies: Dict[str, CacheStrategy] = {}
    
    def set_strategy(self, resource: str, strategy: CacheStrategy):
        """Set cache strategy for resource."""
        self._strategies[resource] = strategy
        logger.info(f"Set cache strategy for {resource}: {strategy.value}")
    
    def get_strategy(self, resource: str) -> CacheStrategy:
        """Get cache strategy for resource."""
        return self._strategies.get(resource, self.default_strategy)
    
    async def get(
        self,
        resource: str,
        cache_get: Callable,
        network_get: Callable
    ) -> Any:
        """Get resource using strategy."""
        strategy = self.get_strategy(resource)
        
        if strategy == CacheStrategy.CACHE_FIRST:
            cached = await cache_get()
            if cached is not None:
                return cached
            return await network_get()
        
        elif strategy == CacheStrategy.NETWORK_FIRST:
            try:
                return await network_get()
            except Exception:
                return await cache_get()
        
        elif strategy == CacheStrategy.CACHE_ONLY:
            return await cache_get()
        
        elif strategy == CacheStrategy.NETWORK_ONLY:
            return await network_get()
        
        elif strategy == CacheStrategy.STALE_WHILE_REVALIDATE:
            cached = await cache_get()
            # Revalidate in background
            asyncio.create_task(network_get())
            return cached
        
        return await network_get()

