"""
Async operations for KV Cache.

Provides async/await support for non-blocking cache operations.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Coroutine

import torch

from kv_cache.types import TensorPair

logger = logging.getLogger(__name__)


class AsyncCacheOperations:
    """
    Async wrapper for cache operations.
    
    Allows non-blocking cache operations for I/O-bound scenarios.
    """
    
    def __init__(self, cache: Any):
        """
        Initialize async cache operations.
        
        Args:
            cache: Cache instance (BaseKVCache or compatible)
        """
        self.cache = cache
        self._executor = None
    
    async def get_async(self, position: int) -> TensorPair | None:
        """
        Async get operation.
        
        Args:
            position: Cache position
            
        Returns:
            Cached entry or None
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self.cache.get,
            position
        )
    
    async def put_async(
        self,
        position: int,
        key: torch.Tensor,
        value: torch.Tensor
    ) -> None:
        """
        Async put operation.
        
        Args:
            position: Cache position
            key: Key tensor
            value: Value tensor
        """
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            self._executor,
            self.cache.put,
            position,
            key,
            value
        )
    
    async def batch_get_async(self, positions: list[int]) -> list[TensorPair | None]:
        """
        Async batch get operation.
        
        Args:
            positions: List of cache positions
            
        Returns:
            List of cached entries
        """
        loop = asyncio.get_event_loop()
        tasks = [self.get_async(pos) for pos in positions]
        return await asyncio.gather(*tasks)
    
    async def batch_put_async(
        self,
        entries: list[tuple[int, torch.Tensor, torch.Tensor]]
    ) -> None:
        """
        Async batch put operation.
        
        Args:
            entries: List of (position, key, value) tuples
        """
        loop = asyncio.get_event_loop()
        tasks = [
            self.put_async(pos, key, value)
            for pos, key, value in entries
        ]
        await asyncio.gather(*tasks)
    
    async def clear_async(self) -> None:
        """Async clear operation."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            self._executor,
            self.cache.clear
        )
    
    async def get_stats_async(self) -> dict[str, Any]:
        """Async get stats operation."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self.cache.get_stats
        )

