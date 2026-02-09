"""Parallel processing utilities."""

import asyncio
from typing import List, Callable, Any, TypeVar
from functools import reduce
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

T = TypeVar('T')
R = TypeVar('R')


async def parallel_map(
    func: Callable[[T], R],
    items: List[T],
    max_workers: int = 5
) -> List[R]:
    """
    Map function over items in parallel (async).
    
    Args:
        func: Function to apply
        items: List of items
        max_workers: Maximum concurrent workers
        
    Returns:
        List of results
    """
    semaphore = asyncio.Semaphore(max_workers)
    
    async def process_item(item: T) -> R:
        async with semaphore:
            if asyncio.iscoroutinefunction(func):
                return await func(item)
            else:
                return func(item)
    
    tasks = [process_item(item) for item in items]
    return await asyncio.gather(*tasks)


def parallel_map_threads(
    func: Callable[[T], R],
    items: List[T],
    max_workers: int = 5
) -> List[R]:
    """
    Map function over items using threads.
    
    Args:
        func: Function to apply
        items: List of items
        max_workers: Maximum thread workers
        
    Returns:
        List of results
    """
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(func, items))


def parallel_map_processes(
    func: Callable[[T], R],
    items: List[T],
    max_workers: int = 5
) -> List[R]:
    """
    Map function over items using processes.
    
    Args:
        func: Function to apply
        items: List of items
        max_workers: Maximum process workers
        
    Returns:
        List of results
    """
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(func, items))


async def parallel_filter(
    predicate: Callable[[T], bool],
    items: List[T],
    max_workers: int = 5
) -> List[T]:
    """
    Filter items in parallel (async).
    
    Args:
        predicate: Filter function
        items: List of items
        max_workers: Maximum concurrent workers
        
    Returns:
        Filtered list
    """
    results = await parallel_map(predicate, items, max_workers)
    return [item for item, result in zip(items, results) if result]


def parallel_reduce(
    func: Callable[[R, T], R],
    items: List[T],
    initial: R,
    max_workers: int = 5
) -> R:
    """
    Reduce items in parallel (chunked).
    
    Args:
        func: Reduction function
        items: List of items
        initial: Initial value
        max_workers: Maximum workers
        
    Returns:
        Reduced value
    """
    from utils.collection_utils import chunk_list
    
    chunks = chunk_list(items, len(items) // max_workers + 1)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        chunk_results = list(executor.map(
            lambda chunk: reduce(lambda acc, x: func(acc, x), chunk, initial),
            chunks
        ))
    
    return reduce(func, chunk_results, initial)

