"""
Stream utilities
Lazy evaluation and streaming operations
"""

from typing import TypeVar, Callable, Iterator, Optional, List, Any
from functools import reduce

T = TypeVar('T')
U = TypeVar('U')


class Stream:
    """
    Stream for lazy evaluation and chaining operations
    """
    
    def __init__(self, iterator: Iterator[T]):
        self.iterator = iterator
    
    @classmethod
    def of(cls, items: List[T]) -> 'Stream':
        """Create stream from list"""
        return cls(iter(items))
    
    def map(self, func: Callable[[T], U]) -> 'Stream':
        """Map function over stream"""
        return Stream(func(item) for item in self.iterator)
    
    def filter(self, predicate: Callable[[T], bool]) -> 'Stream':
        """Filter stream by predicate"""
        return Stream(item for item in self.iterator if predicate(item))
    
    def flat_map(self, func: Callable[[T], Iterator[U]]) -> 'Stream':
        """FlatMap over stream"""
        return Stream(
            item
            for sublist in self.iterator
            for item in func(sublist)
        )
    
    def take(self, n: int) -> 'Stream':
        """Take first n items"""
        def take_iterator():
            for i, item in enumerate(self.iterator):
                if i >= n:
                    break
                yield item
        
        return Stream(take_iterator())
    
    def drop(self, n: int) -> 'Stream':
        """Drop first n items"""
        def drop_iterator():
            for i, item in enumerate(self.iterator):
                if i >= n:
                    yield item
        
        return Stream(drop_iterator())
    
    def take_while(self, predicate: Callable[[T], bool]) -> 'Stream':
        """Take items while predicate is true"""
        return Stream(
            item
            for item in self.iterator
            if predicate(item)
        )
    
    def reduce(self, func: Callable[[U, T], U], initial: Optional[U] = None) -> U:
        """Reduce stream"""
        if initial is not None:
            return reduce(func, self.iterator, initial)
        return reduce(func, self.iterator)
    
    def collect(self) -> List[T]:
        """Collect stream to list"""
        return list(self.iterator)
    
    def first(self) -> Optional[T]:
        """Get first item"""
        try:
            return next(self.iterator)
        except StopIteration:
            return None
    
    def count(self) -> int:
        """Count items in stream"""
        return sum(1 for _ in self.iterator)
    
    def any_match(self, predicate: Callable[[T], bool]) -> bool:
        """Check if any item matches predicate"""
        return any(predicate(item) for item in self.iterator)
    
    def all_match(self, predicate: Callable[[T], bool]) -> bool:
        """Check if all items match predicate"""
        return all(predicate(item) for item in self.iterator)


def stream(items: List[T]) -> Stream:
    """Create stream from list"""
    return Stream.of(items)

