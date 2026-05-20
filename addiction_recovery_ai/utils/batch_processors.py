"""
Batch processing utilities
Advanced batch processing patterns
"""

from typing import TypeVar, List, Callable, Optional
import asyncio

try:
    from utils.async_helpers import run_in_batches, run_parallel
except ImportError:
    from ..async_helpers import run_in_batches, run_parallel

T = TypeVar('T')
U = TypeVar('U')


async def batch_map(
    items: List[T],
    func: Callable[[T], U],
    batch_size: int = 10,
    max_concurrent: int = 5
) -> List[U]:
    """
    Map function over items in batches
    
    Args:
        items: List of items to process
        func: Function to apply
        batch_size: Size of each batch
        max_concurrent: Maximum concurrent batches
    
    Returns:
        List of results
    """
    async def process_batch(batch: List[T]) -> List[U]:
        if asyncio.iscoroutinefunction(func):
            return await asyncio.gather(*[func(item) for item in batch])
        return [func(item) for item in batch]
    
    batches = [items[i:i + batch_size] for i in range(0, len(items), batch_size)]
    results = await run_parallel([process_batch(batch) for batch in batches], max_concurrent)
    
    return [item for sublist in results for item in sublist]


async def batch_filter(
    items: List[T],
    predicate: Callable[[T], bool],
    batch_size: int = 10
) -> List[T]:
    """
    Filter items in batches
    
    Args:
        items: List of items to filter
        predicate: Predicate function
        batch_size: Size of each batch
    
    Returns:
        Filtered list
    """
    async def process_batch(batch: List[T]) -> List[T]:
        if asyncio.iscoroutinefunction(predicate):
            results = await asyncio.gather(*[predicate(item) for item in batch])
            return [item for item, result in zip(batch, results) if result]
        return [item for item in batch if predicate(item)]
    
    batches = [items[i:i + batch_size] for i in range(0, len(items), batch_size)]
    results = await asyncio.gather(*[process_batch(batch) for batch in batches])
    
    return [item for sublist in results for item in sublist]


async def batch_reduce(
    items: List[T],
    func: Callable[[U, T], U],
    initial: U,
    batch_size: int = 10
) -> U:
    """
    Reduce items in batches
    
    Args:
        items: List of items to reduce
        func: Reduction function
        initial: Initial value
        batch_size: Size of each batch
    
    Returns:
        Reduced value
    """
    batches = [items[i:i + batch_size] for i in range(0, len(items), batch_size)]
    
    result = initial
    for batch in batches:
        for item in batch:
            if asyncio.iscoroutinefunction(func):
                result = await func(result, item)
            else:
                result = func(result, item)
    
    return result

