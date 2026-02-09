"""
Stream and Buffer Utilities for Piel Mejorador AI SAM3
====================================================

Unified stream and buffer pattern utilities.
"""

import asyncio
import logging
from typing import TypeVar, Iterator, AsyncIterator, Callable, Optional, List, Any
from collections import deque
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

T = TypeVar('T')


class Buffer:
    """Generic buffer for data."""
    
    def __init__(self, max_size: Optional[int] = None):
        """
        Initialize buffer.
        
        Args:
            max_size: Maximum buffer size (None = unlimited)
        """
        self._buffer = deque(maxlen=max_size)
        self.max_size = max_size
    
    def put(self, item: T):
        """
        Put item in buffer.
        
        Args:
            item: Item to add
        """
        self._buffer.append(item)
    
    def get(self) -> Optional[T]:
        """
        Get item from buffer.
        
        Returns:
            Item or None if empty
        """
        if self._buffer:
            return self._buffer.popleft()
        return None
    
    def peek(self) -> Optional[T]:
        """
        Peek at first item without removing.
        
        Returns:
            Item or None if empty
        """
        if self._buffer:
            return self._buffer[0]
        return None
    
    def size(self) -> int:
        """Get buffer size."""
        return len(self._buffer)
    
    def empty(self) -> bool:
        """Check if buffer is empty."""
        return len(self._buffer) == 0
    
    def full(self) -> bool:
        """Check if buffer is full."""
        if self.max_size is None:
            return False
        return len(self._buffer) >= self.max_size
    
    def clear(self):
        """Clear buffer."""
        self._buffer.clear()
    
    def to_list(self) -> List[T]:
        """Convert buffer to list."""
        return list(self._buffer)


class AsyncBuffer:
    """Async buffer for data."""
    
    def __init__(self, max_size: int = 0):
        """
        Initialize async buffer.
        
        Args:
            max_size: Maximum buffer size (0 = unlimited)
        """
        self._queue = asyncio.Queue(maxsize=max_size)
        self.max_size = max_size
    
    async def put(self, item: T):
        """
        Put item in buffer.
        
        Args:
            item: Item to add
        """
        await self._queue.put(item)
    
    async def get(self) -> T:
        """
        Get item from buffer.
        
        Returns:
            Item
        """
        return await self._queue.get()
    
    async def get_nowait(self) -> Optional[T]:
        """
        Get item without waiting.
        
        Returns:
            Item or None if empty
        """
        try:
            return self._queue.get_nowait()
        except asyncio.QueueEmpty:
            return None
    
    async def size(self) -> int:
        """Get buffer size."""
        return self._queue.qsize()
    
    async def empty(self) -> bool:
        """Check if buffer is empty."""
        return self._queue.empty()
    
    async def full(self) -> bool:
        """Check if buffer is full."""
        return self._queue.full()
    
    async def clear(self):
        """Clear buffer."""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break


class Stream:
    """Generic stream for data processing."""
    
    def __init__(self, source: Iterator[T]):
        """
        Initialize stream.
        
        Args:
            source: Source iterator
        """
        self._source = source
        self._buffer = Buffer()
    
    def map(self, func: Callable[[T], Any]) -> "Stream":
        """
        Map stream items.
        
        Args:
            func: Mapping function
            
        Returns:
            Self for chaining
        """
        mapped = (func(item) for item in self._source)
        return Stream(mapped)
    
    def filter(self, predicate: Callable[[T], bool]) -> "Stream":
        """
        Filter stream items.
        
        Args:
            predicate: Filter predicate
            
        Returns:
            Self for chaining
        """
        filtered = (item for item in self._source if predicate(item))
        return Stream(filtered)
    
    def take(self, count: int) -> "Stream":
        """
        Take first N items.
        
        Args:
            count: Number of items to take
            
        Returns:
            Self for chaining
        """
        taken = (item for i, item in enumerate(self._source) if i < count)
        return Stream(taken)
    
    def skip(self, count: int) -> "Stream":
        """
        Skip first N items.
        
        Args:
            count: Number of items to skip
            
        Returns:
            Self for chaining
        """
        skipped = (item for i, item in enumerate(self._source) if i >= count)
        return Stream(skipped)
    
    def collect(self) -> List[T]:
        """
        Collect all items into list.
        
        Returns:
            List of items
        """
        return list(self._source)
    
    def __iter__(self) -> Iterator[T]:
        """Get iterator."""
        return self._source


class AsyncStream:
    """Async stream for data processing."""
    
    def __init__(self, source: AsyncIterator[T]):
        """
        Initialize async stream.
        
        Args:
            source: Source async iterator
        """
        self._source = source
    
    async def map(self, func: Callable[[T], Any]) -> "AsyncStream":
        """
        Map stream items.
        
        Args:
            func: Mapping function
            
        Returns:
            Self for chaining
        """
        async def mapped():
            async for item in self._source:
                yield func(item)
        return AsyncStream(mapped())
    
    async def filter(self, predicate: Callable[[T], bool]) -> "AsyncStream":
        """
        Filter stream items.
        
        Args:
            predicate: Filter predicate
            
        Returns:
            Self for chaining
        """
        async def filtered():
            async for item in self._source:
                if predicate(item):
                    yield item
        return AsyncStream(filtered())
    
    async def collect(self) -> List[T]:
        """
        Collect all items into list.
        
        Returns:
            List of items
        """
        result = []
        async for item in self._source:
            result.append(item)
        return result
    
    def __aiter__(self) -> AsyncIterator[T]:
        """Get async iterator."""
        return self._source


class StreamUtils:
    """Unified stream utilities."""
    
    @staticmethod
    def create_buffer(max_size: Optional[int] = None) -> Buffer:
        """
        Create buffer.
        
        Args:
            max_size: Maximum buffer size
            
        Returns:
            Buffer
        """
        return Buffer(max_size)
    
    @staticmethod
    def create_async_buffer(max_size: int = 0) -> AsyncBuffer:
        """
        Create async buffer.
        
        Args:
            max_size: Maximum buffer size
            
        Returns:
            AsyncBuffer
        """
        return AsyncBuffer(max_size)
    
    @staticmethod
    def create_stream(source: Iterator[T]) -> Stream:
        """
        Create stream.
        
        Args:
            source: Source iterator
            
        Returns:
            Stream
        """
        return Stream(source)
    
    @staticmethod
    def create_async_stream(source: AsyncIterator[T]) -> AsyncStream:
        """
        Create async stream.
        
        Args:
            source: Source async iterator
            
        Returns:
            AsyncStream
        """
        return AsyncStream(source)


# Convenience functions
def create_buffer(max_size: Optional[int] = None) -> Buffer:
    """Create buffer."""
    return StreamUtils.create_buffer(max_size)


def create_async_buffer(max_size: int = 0) -> AsyncBuffer:
    """Create async buffer."""
    return StreamUtils.create_async_buffer(max_size)


def create_stream(source: Iterator[T]) -> Stream:
    """Create stream."""
    return StreamUtils.create_stream(source)




