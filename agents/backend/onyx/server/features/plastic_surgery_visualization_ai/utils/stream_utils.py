"""Streaming utilities."""

from typing import AsyncIterator, Iterator, Any, Callable
import asyncio


async def async_stream(
    items: list,
    delay: float = 0.0
) -> AsyncIterator[Any]:
    """
    Create async stream from list.
    
    Args:
        items: List of items
        delay: Delay between items in seconds
        
    Yields:
        Items from list
    """
    for item in items:
        if delay > 0:
            await asyncio.sleep(delay)
        yield item


async def stream_with_transform(
    stream: AsyncIterator,
    transform: Callable
) -> AsyncIterator[Any]:
    """
    Transform items in stream.
    
    Args:
        stream: Async iterator
        transform: Transform function
        
    Yields:
        Transformed items
    """
    async for item in stream:
        yield transform(item)


async def stream_with_filter(
    stream: AsyncIterator,
    predicate: Callable
) -> AsyncIterator[Any]:
    """
    Filter items in stream.
    
    Args:
        stream: Async iterator
        predicate: Filter function
        
    Yields:
        Filtered items
    """
    async for item in stream:
        if predicate(item):
            yield item


async def batch_stream(
    stream: AsyncIterator,
    batch_size: int = 10
) -> AsyncIterator[list]:
    """
    Batch items from stream.
    
    Args:
        stream: Async iterator
        batch_size: Size of each batch
        
    Yields:
        Batches of items
    """
    batch = []
    async for item in stream:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    
    if batch:
        yield batch


async def merge_streams(*streams: AsyncIterator) -> AsyncIterator[Any]:
    """
    Merge multiple streams.
    
    Args:
        *streams: Async iterators to merge
        
    Yields:
        Items from all streams
    """
    tasks = [stream.__anext__() for stream in streams]
    
    while tasks:
        done, pending = await asyncio.wait(
            tasks,
            return_when=asyncio.FIRST_COMPLETED
        )
        
        for task in done:
            try:
                item = await task
                yield item
                # Get next item from same stream
                stream_index = tasks.index(task)
                tasks[stream_index] = streams[stream_index].__anext__()
            except StopAsyncIteration:
                tasks.remove(task)
        
        tasks = list(pending)


async def stream_to_list(stream: AsyncIterator) -> list:
    """
    Collect all items from stream into list.
    
    Args:
        stream: Async iterator
        
    Returns:
        List of items
    """
    return [item async for item in stream]

