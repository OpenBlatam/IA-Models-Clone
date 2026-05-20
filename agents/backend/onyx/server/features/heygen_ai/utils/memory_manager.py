"""
Memory Management Utilities
============================

Helper functions for efficient memory management during batch processing.
This eliminates repetitive memory cleanup code throughout the codebase.

Benefits:
- Consistent memory management patterns
- Prevents memory leaks during batch processing
- Easier to optimize memory usage
- Centralized cleanup logic
"""

import gc
import logging
from typing import List, TypeVar, Callable, Iterator, Optional
import torch

logger = logging.getLogger(__name__)

# Type variable for batch item type
T = TypeVar('T')


def process_in_batches(
    items: List[T],
    batch_size: int,
    processor: Callable[[List[T]], List[T]],
    cleanup_interval: Optional[int] = None,
) -> List[T]:
    """
    Process items in batches with automatic memory cleanup.
    
    This helper eliminates repetitive batch processing code that appears
    in video_renderer.py and other files.
    
    Args:
        items: List of items to process
        batch_size: Number of items per batch
        processor: Function to process a batch of items
        cleanup_interval: Cleanup memory every N batches (None = after each batch)
    
    Returns:
        List of processed items
    
    Example:
        >>> def resize_frame(frame):
        ...     return cv2.resize(frame, (1920, 1080))
        >>> 
        >>> def process_batch(frames):
        ...     return [resize_frame(f) for f in frames]
        >>> 
        >>> processed = process_in_batches(
        ...     video_frames,
        ...     batch_size=32,
        ...     processor=process_batch,
        ...     cleanup_interval=4
        ... )
    """
    if not items:
        return []
    
    processed_items = []
    total_batches = (len(items) + batch_size - 1) // batch_size
    
    for batch_idx in range(0, len(items), batch_size):
        batch = items[batch_idx:batch_idx + batch_size]
        batch_num = (batch_idx // batch_size) + 1
        
        try:
            # Process batch
            batch_results = processor(batch)
            processed_items.extend(batch_results)
            
            # Cleanup memory periodically
            should_cleanup = (
                cleanup_interval is None or
                batch_num % cleanup_interval == 0 or
                batch_num == total_batches
            )
            
            if should_cleanup:
                clear_memory()
                logger.debug(
                    f"Memory cleaned after batch {batch_num}/{total_batches}"
                )
        
        except MemoryError as e:
            logger.error(f"Out of memory processing batch {batch_num}")
            clear_memory()
            raise RuntimeError(
                "Insufficient memory for batch processing. "
                "Try reducing batch size or processing fewer items."
            ) from e
    
    return processed_items


def clear_memory() -> None:
    """
    Clear memory cache and run garbage collection.
    
    This is a more general version of clear_gpu_memory() that works
    for both GPU and CPU operations.
    
    Example:
        >>> # After processing large batch
        >>> clear_memory()
    """
    # Clear GPU cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Run garbage collection
    gc.collect()
    logger.debug("Memory cleared")


def batch_iterator(
    items: List[T],
    batch_size: int,
    cleanup_interval: Optional[int] = None,
) -> Iterator[List[T]]:
    """
    Create an iterator that yields items in batches with memory management.
    
    Args:
        items: List of items to iterate over
        batch_size: Number of items per batch
        cleanup_interval: Cleanup memory every N batches
    
    Yields:
        Batches of items
    
    Example:
        >>> for batch in batch_iterator(frames, batch_size=32):
        ...     processed = process_batch(batch)
        ...     results.extend(processed)
    """
    total_batches = (len(items) + batch_size - 1) // batch_size
    
    for batch_idx in range(0, len(items), batch_size):
        batch = items[batch_idx:batch_idx + batch_size]
        batch_num = (batch_idx // batch_size) + 1
        
        yield batch
        
        # Cleanup periodically
        if cleanup_interval and batch_num % cleanup_interval == 0:
            clear_memory()


def memory_efficient_map(
    items: List[T],
    mapper: Callable[[T], T],
    batch_size: int = 32,
    cleanup_interval: Optional[int] = 4,
) -> List[T]:
    """
    Apply a mapping function to items in batches with memory management.
    
    Args:
        items: List of items to map
        mapper: Function to apply to each item
        batch_size: Number of items per batch
        cleanup_interval: Cleanup memory every N batches
    
    Returns:
        List of mapped items
    
    Example:
        >>> def resize_frame(frame):
        ...     return cv2.resize(frame, (1920, 1080))
        >>> 
        >>> resized = memory_efficient_map(
        ...     frames,
        ...     mapper=resize_frame,
        ...     batch_size=32
        ... )
    """
    def process_batch(batch: List[T]) -> List[T]:
        return [mapper(item) for item in batch]
    
    return process_in_batches(
        items,
        batch_size=batch_size,
        processor=process_batch,
        cleanup_interval=cleanup_interval,
    )








