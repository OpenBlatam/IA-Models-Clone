"""
Iterator utilities
Advanced iterator patterns
"""

from typing import TypeVar, Iterator, Callable, Optional, List, Any
from itertools import islice, cycle, repeat, chain

T = TypeVar('T')
U = TypeVar('U')


def take(iterator: Iterator[T], n: int) -> Iterator[T]:
    """
    Take first n items from iterator
    
    Args:
        iterator: Iterator to take from
        n: Number of items to take
    
    Returns:
        Iterator with first n items
    """
    return islice(iterator, n)


def drop(iterator: Iterator[T], n: int) -> Iterator[T]:
    """
    Drop first n items from iterator
    
    Args:
        iterator: Iterator to drop from
        n: Number of items to drop
    
    Returns:
        Iterator without first n items
    """
    return islice(iterator, n, None)


def take_while(iterator: Iterator[T], predicate: Callable[[T], bool]) -> Iterator[T]:
    """
    Take items while predicate is true
    
    Args:
        iterator: Iterator to take from
        predicate: Predicate function
    
    Returns:
        Iterator with items while predicate is true
    """
    for item in iterator:
        if predicate(item):
            yield item
        else:
            break


def drop_while(iterator: Iterator[T], predicate: Callable[[T], bool]) -> Iterator[T]:
    """
    Drop items while predicate is true
    
    Args:
        iterator: Iterator to drop from
        predicate: Predicate function
    
    Returns:
        Iterator without items while predicate is true
    """
    dropping = True
    for item in iterator:
        if dropping and predicate(item):
            continue
        dropping = False
        yield item


def chunk(iterator: Iterator[T], size: int) -> Iterator[List[T]]:
    """
    Chunk iterator into lists of specified size
    
    Args:
        iterator: Iterator to chunk
        size: Size of each chunk
    
    Returns:
        Iterator of chunks
    """
    chunk_list = []
    for item in iterator:
        chunk_list.append(item)
        if len(chunk_list) == size:
            yield chunk_list
            chunk_list = []
    
    if chunk_list:
        yield chunk_list


def pairwise(iterator: Iterator[T]) -> Iterator[tuple[T, T]]:
    """
    Pairwise iteration
    
    Args:
        iterator: Iterator to pair
    
    Returns:
        Iterator of pairs
    """
    prev = None
    for item in iterator:
        if prev is not None:
            yield (prev, item)
        prev = item


def window(iterator: Iterator[T], size: int) -> Iterator[List[T]]:
    """
    Sliding window over iterator
    
    Args:
        iterator: Iterator to window
        size: Window size
    
    Returns:
        Iterator of windows
    """
    window_list = []
    for item in iterator:
        window_list.append(item)
        if len(window_list) == size:
            yield window_list.copy()
            window_list.pop(0)
    
    # Yield remaining items if any
    while window_list:
        yield window_list.copy()
        window_list.pop(0)


def interleave(*iterators: Iterator[T]) -> Iterator[T]:
    """
    Interleave multiple iterators
    
    Args:
        *iterators: Iterators to interleave
    
    Returns:
        Interleaved iterator
    """
    iterators_list = list(iterators)
    while iterators_list:
        for it in iterators_list[:]:
            try:
                yield next(it)
            except StopIteration:
                iterators_list.remove(it)

