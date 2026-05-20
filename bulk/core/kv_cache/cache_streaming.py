"""
Cache streaming utilities.

Provides streaming capabilities for cache operations.
"""
from __future__ import annotations

import logging
from typing import Dict, Any, Optional, Iterator, AsyncIterator, Callable
import asyncio

logger = logging.getLogger(__name__)


class CacheStreamer:
    """
    Cache streamer.
    
    Provides streaming operations for cache.
    """
    
    def __init__(self, cache: Any):
        """
        Initialize streamer.
        
        Args:
            cache: Cache instance
        """
        self.cache = cache
    
    def stream_get(self, positions: Iterator[int]) -> Iterator[Optional[Any]]:
        """
        Stream get operations.
        
        Args:
            positions: Iterator of positions
            
        Yields:
            Cached values or None
        """
        for position in positions:
            yield self.cache.get(position)
    
    def stream_put(
        self,
        items: Iterator[tuple[int, Any]]
    ) -> Iterator[bool]:
        """
        Stream put operations.
        
        Args:
            items: Iterator of (position, value) tuples
            
        Yields:
            Success status
        """
        for position, value in items:
            try:
                self.cache.put(position, value)
                yield True
            except Exception as e:
                logger.error(f"Failed to put {position}: {e}")
                yield False
    
    def stream_batch_get(
        self,
        positions: Iterator[int],
        batch_size: int = 100
    ) -> Iterator[Dict[int, Optional[Any]]]:
        """
        Stream batch get operations.
        
        Args:
            positions: Iterator of positions
            batch_size: Batch size
            
        Yields:
            Dictionary of position -> value
        """
        batch = []
        for position in positions:
            batch.append(position)
            
            if len(batch) >= batch_size:
                results = {}
                for pos in batch:
                    results[pos] = self.cache.get(pos)
                yield results
                batch = []
        
        # Yield remaining
        if batch:
            results = {}
            for pos in batch:
                results[pos] = self.cache.get(pos)
            yield results
    
    async def async_stream_get(
        self,
        positions: AsyncIterator[int]
    ) -> AsyncIterator[Optional[Any]]:
        """
        Async stream get operations.
        
        Args:
            positions: Async iterator of positions
            
        Yields:
            Cached values or None
        """
        async for position in positions:
            # Run in executor to avoid blocking
            loop = asyncio.get_event_loop()
            value = await loop.run_in_executor(None, self.cache.get, position)
            yield value
    
    async def async_stream_put(
        self,
        items: AsyncIterator[tuple[int, Any]]
    ) -> AsyncIterator[bool]:
        """
        Async stream put operations.
        
        Args:
            items: Async iterator of (position, value) tuples
            
        Yields:
            Success status
        """
        async for position, value in items:
            try:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, self.cache.put, position, value)
                yield True
            except Exception as e:
                logger.error(f"Failed to put {position}: {e}")
                yield False


class CachePipeline:
    """
    Cache processing pipeline.
    
    Provides pipeline for processing cache operations.
    """
    
    def __init__(self, cache: Any):
        """
        Initialize pipeline.
        
        Args:
            cache: Cache instance
        """
        self.cache = cache
        self.stages: list[Callable] = []
    
    def add_stage(self, stage: Callable) -> None:
        """
        Add processing stage.
        
        Args:
            stage: Processing function
        """
        self.stages.append(stage)
    
    def process(self, items: Iterator[Any]) -> Iterator[Any]:
        """
        Process items through pipeline.
        
        Args:
            items: Iterator of items
            
        Yields:
            Processed items
        """
        for item in items:
            result = item
            for stage in self.stages:
                result = stage(result, self.cache)
            yield result


class CacheBatchProcessor:
    """
    Batch processor for cache operations.
    
    Provides efficient batch processing.
    """
    
    def __init__(self, cache: Any, batch_size: int = 100):
        """
        Initialize batch processor.
        
        Args:
            cache: Cache instance
            batch_size: Batch size
        """
        self.cache = cache
        self.batch_size = batch_size
    
    def process_batches(
        self,
        items: Iterator[Any],
        process_fn: Callable[[list[Any], Any], list[Any]]
    ) -> Iterator[list[Any]]:
        """
        Process items in batches.
        
        Args:
            items: Iterator of items
            process_fn: Processing function (batch, cache) -> results
            
        Yields:
            Processed batches
        """
        batch = []
        for item in items:
            batch.append(item)
            
            if len(batch) >= self.batch_size:
                results = process_fn(batch, self.cache)
                yield results
                batch = []
        
        # Yield remaining
        if batch:
            results = process_fn(batch, self.cache)
            yield results
    
    async def async_process_batches(
        self,
        items: AsyncIterator[Any],
        process_fn: Callable[[list[Any], Any], list[Any]]
    ) -> AsyncIterator[list[Any]]:
        """
        Async process items in batches.
        
        Args:
            items: Async iterator of items
            process_fn: Processing function
            
        Yields:
            Processed batches
        """
        batch = []
        async for item in items:
            batch.append(item)
            
            if len(batch) >= self.batch_size:
                loop = asyncio.get_event_loop()
                results = await loop.run_in_executor(
                    None,
                    process_fn,
                    batch,
                    self.cache
                )
                yield results
                batch = []
        
        # Yield remaining
        if batch:
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                process_fn,
                batch,
                self.cache
            )
            yield results

