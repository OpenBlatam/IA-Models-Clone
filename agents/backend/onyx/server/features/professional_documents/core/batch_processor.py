"""
Batch processing utilities for professional documents.

Functions for processing multiple documents in batches.
"""

import asyncio
from typing import List, TypeVar, Callable, Optional, Any
from .models import ProfessionalDocument

T = TypeVar('T')
R = TypeVar('R')


async def process_batch(
    items: List[T],
    processor: Callable[[T], R],
    batch_size: int = 10,
    max_concurrent: int = 5
) -> List[R]:
    """
    Process items in batches with concurrency control.
    
    Args:
        items: List of items to process
        processor: Async function to process each item
        batch_size: Number of items per batch
        max_concurrent: Maximum concurrent operations
        
    Returns:
        List of processed results
    """
    results = []
    
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        
        # Process batch with concurrency limit
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_with_semaphore(item: T) -> R:
            async with semaphore:
                return await processor(item)
        
        batch_results = await asyncio.gather(
            *[process_with_semaphore(item) for item in batch],
            return_exceptions=True
        )
        
        # Filter out exceptions
        valid_results = [
            result for result in batch_results
            if not isinstance(result, Exception)
        ]
        
        results.extend(valid_results)
    
    return results


async def batch_export_documents(
    documents: List[ProfessionalDocument],
    export_func: Callable[[ProfessionalDocument], Any],
    batch_size: int = 5
) -> List[Any]:
    """
    Export multiple documents in batches.
    
    Args:
        documents: List of documents to export
        export_func: Function to export a single document
        batch_size: Number of documents per batch
        
    Returns:
        List of export results
    """
    return await process_batch(
        documents,
        export_func,
        batch_size=batch_size,
        max_concurrent=3  # Lower concurrency for I/O operations
    )


async def batch_generate_documents(
    requests: List[Any],
    generate_func: Callable[[Any], Any],
    batch_size: int = 3
) -> List[Any]:
    """
    Generate multiple documents in batches.
    
    Args:
        requests: List of generation requests
        generate_func: Function to generate a single document
        batch_size: Number of requests per batch
        
    Returns:
        List of generation results
    """
    return await process_batch(
        requests,
        generate_func,
        batch_size=batch_size,
        max_concurrent=2  # Lower concurrency for AI operations
    )






