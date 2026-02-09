"""
Iterator and Generator Utilities for Piel Mejorador AI SAM3
==========================================================

Unified iterator and generator pattern utilities.
"""

import logging
from typing import TypeVar, Iterator, Iterable, Callable, Optional, List, Any, Generator
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

T = TypeVar('T')
R = TypeVar('R')


class BaseIterator(ABC):
    """Base iterator interface."""
    
    @abstractmethod
    def __iter__(self) -> Iterator[T]:
        """Get iterator."""
        pass
    
    @abstractmethod
    def __next__(self) -> T:
        """Get next item."""
        pass


class FunctionIterator:
    """Iterator using a function."""
    
    def __init__(
        self,
        items: Iterable[T],
        transform: Optional[Callable[[T], R]] = None,
        filter_func: Optional[Callable[[T], bool]] = None
    ):
        """
        Initialize function iterator.
        
        Args:
            items: Items to iterate
            transform: Optional transformation function
            filter_func: Optional filter function
        """
        self._items = iter(items)
        self._transform = transform
        self._filter = filter_func
    
    def __iter__(self) -> Iterator[R]:
        """Get iterator."""
        return self
    
    def __next__(self) -> R:
        """Get next item."""
        while True:
            item = next(self._items)
            
            # Filter if provided
            if self._filter and not self._filter(item):
                continue
            
            # Transform if provided
            if self._transform:
                return self._transform(item)
            
            return item


class ChunkedIterator:
    """Iterator that yields chunks of items."""
    
    def __init__(self, items: Iterable[T], chunk_size: int):
        """
        Initialize chunked iterator.
        
        Args:
            items: Items to iterate
            chunk_size: Size of each chunk
        """
        self._items = iter(items)
        self._chunk_size = chunk_size
    
    def __iter__(self) -> Iterator[List[T]]:
        """Get iterator."""
        return self
    
    def __next__(self) -> List[T]:
        """Get next chunk."""
        chunk = []
        try:
            for _ in range(self._chunk_size):
                chunk.append(next(self._items))
        except StopIteration:
            if not chunk:
                raise
        return chunk


class BatchedIterator:
    """Iterator that yields batches with overlap."""
    
    def __init__(
        self,
        items: Iterable[T],
        batch_size: int,
        overlap: int = 0
    ):
        """
        Initialize batched iterator.
        
        Args:
            items: Items to iterate
            batch_size: Size of each batch
            overlap: Number of overlapping items between batches
        """
        self._items = list(items)  # Convert to list for indexing
        self._batch_size = batch_size
        self._overlap = overlap
        self._index = 0
    
    def __iter__(self) -> Iterator[List[T]]:
        """Get iterator."""
        return self
    
    def __next__(self) -> List[T]:
        """Get next batch."""
        if self._index >= len(self._items):
            raise StopIteration
        
        end_index = min(self._index + self._batch_size, len(self._items))
        batch = self._items[self._index:end_index]
        self._index += self._batch_size - self._overlap
        
        return batch


class WindowedIterator:
    """Iterator that yields sliding windows."""
    
    def __init__(
        self,
        items: Iterable[T],
        window_size: int,
        step: int = 1
    ):
        """
        Initialize windowed iterator.
        
        Args:
            items: Items to iterate
            window_size: Size of window
            step: Step size between windows
        """
        self._items = list(items)
        self._window_size = window_size
        self._step = step
        self._index = 0
    
    def __iter__(self) -> Iterator[List[T]]:
        """Get iterator."""
        return self
    
    def __next__(self) -> List[T]:
        """Get next window."""
        if self._index + self._window_size > len(self._items):
            raise StopIteration
        
        window = self._items[self._index:self._index + self._window_size]
        self._index += self._step
        
        return window


class IteratorUtils:
    """Unified iterator utilities."""
    
    @staticmethod
    def create_iterator(
        items: Iterable[T],
        transform: Optional[Callable[[T], R]] = None,
        filter_func: Optional[Callable[[T], bool]] = None
    ) -> FunctionIterator:
        """
        Create iterator from iterable.
        
        Args:
            items: Items to iterate
            transform: Optional transformation function
            filter_func: Optional filter function
            
        Returns:
            FunctionIterator
        """
        return FunctionIterator(items, transform, filter_func)
    
    @staticmethod
    def create_chunked_iterator(
        items: Iterable[T],
        chunk_size: int
    ) -> ChunkedIterator:
        """
        Create chunked iterator.
        
        Args:
            items: Items to iterate
            chunk_size: Size of each chunk
            
        Returns:
            ChunkedIterator
        """
        return ChunkedIterator(items, chunk_size)
    
    @staticmethod
    def create_batched_iterator(
        items: Iterable[T],
        batch_size: int,
        overlap: int = 0
    ) -> BatchedIterator:
        """
        Create batched iterator.
        
        Args:
            items: Items to iterate
            batch_size: Size of each batch
            overlap: Number of overlapping items
            
        Returns:
            BatchedIterator
        """
        return BatchedIterator(items, batch_size, overlap)
    
    @staticmethod
    def create_windowed_iterator(
        items: Iterable[T],
        window_size: int,
        step: int = 1
    ) -> WindowedIterator:
        """
        Create windowed iterator.
        
        Args:
            items: Items to iterate
            window_size: Size of window
            step: Step size between windows
            
        Returns:
            WindowedIterator
        """
        return WindowedIterator(items, window_size, step)
    
    @staticmethod
    def map_iterator(
        items: Iterable[T],
        func: Callable[[T], R]
    ) -> Generator[R, None, None]:
        """
        Map iterator.
        
        Args:
            items: Items to iterate
            func: Mapping function
            
        Yields:
            Mapped items
        """
        for item in items:
            yield func(item)
    
    @staticmethod
    def filter_iterator(
        items: Iterable[T],
        predicate: Callable[[T], bool]
    ) -> Generator[T, None, None]:
        """
        Filter iterator.
        
        Args:
            items: Items to iterate
            predicate: Filter predicate
            
        Yields:
            Filtered items
        """
        for item in items:
            if predicate(item):
                yield item
    
    @staticmethod
    def take_iterator(
        items: Iterable[T],
        count: int
    ) -> Generator[T, None, None]:
        """
        Take first N items.
        
        Args:
            items: Items to iterate
            count: Number of items to take
            
        Yields:
            First N items
        """
        for i, item in enumerate(items):
            if i >= count:
                break
            yield item
    
    @staticmethod
    def skip_iterator(
        items: Iterable[T],
        count: int
    ) -> Generator[T, None, None]:
        """
        Skip first N items.
        
        Args:
            items: Items to iterate
            count: Number of items to skip
            
        Yields:
            Items after skipping
        """
        for i, item in enumerate(items):
            if i >= count:
                yield item
    
    @staticmethod
    def enumerate_iterator(
        items: Iterable[T],
        start: int = 0
    ) -> Generator[tuple[int, T], None, None]:
        """
        Enumerate iterator.
        
        Args:
            items: Items to iterate
            start: Starting index
            
        Yields:
            (index, item) tuples
        """
        for i, item in enumerate(items, start=start):
            yield (i, item)
    
    @staticmethod
    def zip_iterator(
        *iterables: Iterable[Any]
    ) -> Generator[tuple, None, None]:
        """
        Zip multiple iterables.
        
        Args:
            *iterables: Iterables to zip
            
        Yields:
            Tuples of items
        """
        return zip(*iterables)
    
    @staticmethod
    def chain_iterator(
        *iterables: Iterable[T]
    ) -> Generator[T, None, None]:
        """
        Chain multiple iterables.
        
        Args:
            *iterables: Iterables to chain
            
        Yields:
            Items from all iterables
        """
        from itertools import chain
        for item in chain(*iterables):
            yield item


# Convenience functions
def create_iterator(items: Iterable[T], **kwargs) -> FunctionIterator:
    """Create iterator."""
    return IteratorUtils.create_iterator(items, **kwargs)


def create_chunked_iterator(items: Iterable[T], chunk_size: int) -> ChunkedIterator:
    """Create chunked iterator."""
    return IteratorUtils.create_chunked_iterator(items, chunk_size)


def create_batched_iterator(items: Iterable[T], **kwargs) -> BatchedIterator:
    """Create batched iterator."""
    return IteratorUtils.create_batched_iterator(items, **kwargs)

