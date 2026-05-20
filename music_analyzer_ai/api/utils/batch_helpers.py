"""
Batch processing helpers
"""

from typing import List, Callable, Any, Optional
import asyncio
import logging

logger = logging.getLogger(__name__)


async def process_batch(
    items: List[Any],
    processor: Callable,
    batch_size: int = 10,
    max_concurrent: int = 5
) -> List[Any]:
    """
    Process items in batches with concurrency control
    
    Args:
        items: List of items to process
        processor: Async function to process each item
        batch_size: Number of items per batch
        max_concurrent: Maximum concurrent operations
    
    Returns:
        List of processed results
    """
    results = []
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_with_semaphore(item):
        async with semaphore:
            try:
                return await processor(item)
            except Exception as e:
                logger.error(f"Error processing item: {e}")
                return None
    
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        batch_results = await asyncio.gather(
            *[process_with_semaphore(item) for item in batch],
            return_exceptions=True
        )
        results.extend([r for r in batch_results if r is not None])
    
    return results


def chunk_list(items: List[Any], chunk_size: int) -> List[List[Any]]:
    """
    Split list into chunks
    
    Args:
        items: List to chunk
        chunk_size: Size of each chunk
    
    Returns:
        List of chunks
    """
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


async def process_parallel(
    items: List[Any],
    processor: Callable,
    max_workers: int = 5
) -> List[Any]:
    """
    Process items in parallel
    
    Args:
        items: List of items to process
        processor: Async function to process each item
        max_workers: Maximum parallel workers
    
    Returns:
        List of processed results
    """
    semaphore = asyncio.Semaphore(max_workers)
    
    async def process_with_limit(item):
        async with semaphore:
            return await processor(item)
    
    results = await asyncio.gather(
        *[process_with_limit(item) for item in items],
        return_exceptions=True
    )
    
    return [r for r in results if not isinstance(r, Exception)]

