"""Iterator utilities."""

from typing import Iterator, List, Callable, Any, Optional
from itertools import islice, cycle, repeat, chain


def take(iterable: Iterator, n: int) -> List:
    """
    Take first n items from iterator.
    
    Args:
        iterable: Iterator
        n: Number of items
        
    Returns:
        List of items
    """
    return list(islice(iterable, n))


def drop(iterable: Iterator, n: int) -> Iterator:
    """
    Drop first n items from iterator.
    
    Args:
        iterable: Iterator
        n: Number of items to drop
        
    Returns:
        Iterator without first n items
    """
    return islice(iterable, n, None)


def take_while(predicate: Callable, iterable: Iterator) -> List:
    """
    Take items while predicate is true.
    
    Args:
        predicate: Function that returns bool
        iterable: Iterator
        
    Returns:
        List of items
    """
    result = []
    for item in iterable:
        if predicate(item):
            result.append(item)
        else:
            break
    return result


def drop_while(predicate: Callable, iterable: Iterator) -> Iterator:
    """
    Drop items while predicate is true.
    
    Args:
        predicate: Function that returns bool
        iterable: Iterator
        
    Returns:
        Iterator without dropped items
    """
    started = False
    for item in iterable:
        if not predicate(item):
            started = True
        if started:
            yield item


def enumerate_items(iterable: Iterator, start: int = 0) -> Iterator:
    """
    Enumerate items with custom start.
    
    Args:
        iterable: Iterator
        start: Starting index
        
    Yields:
        Tuples of (index, item)
    """
    for item in iterable:
        yield (start, item)
        start += 1


def zip_longest_fill(*iterables: Iterator, fillvalue: Any = None) -> Iterator:
    """
    Zip iterables, filling shorter ones with fillvalue.
    
    Args:
        *iterables: Iterables to zip
        fillvalue: Value to fill shorter iterables
        
    Yields:
        Tuples of items
    """
    from itertools import zip_longest
    return zip_longest(*iterables, fillvalue=fillvalue)


def cycle_items(items: List, n: Optional[int] = None) -> Iterator:
    """
    Cycle through items.
    
    Args:
        items: List of items
        n: Optional number of cycles (None = infinite)
        
    Yields:
        Items in cycle
    """
    if n is None:
        return cycle(items)
    else:
        for _ in range(n):
            for item in items:
                yield item


def repeat_item(item: Any, n: Optional[int] = None) -> Iterator:
    """
    Repeat item.
    
    Args:
        item: Item to repeat
        n: Optional number of times (None = infinite)
        
    Yields:
        Repeated item
    """
    if n is None:
        return repeat(item)
    else:
        return repeat(item, n)


def chain_iterables(*iterables: Iterator) -> Iterator:
    """
    Chain multiple iterables.
    
    Args:
        *iterables: Iterables to chain
        
    Yields:
        Items from all iterables
    """
    return chain(*iterables)


def pairwise(iterable: Iterator) -> Iterator:
    """
    Get pairs of consecutive items.
    
    Args:
        iterable: Iterator
        
    Yields:
        Tuples of (item, next_item)
    """
    from itertools import tee
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

