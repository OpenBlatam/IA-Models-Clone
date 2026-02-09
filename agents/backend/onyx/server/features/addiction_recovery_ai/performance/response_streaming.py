"""
Response Streaming Optimizer
Stream large responses efficiently to reduce memory usage and improve latency
"""

import logging
import asyncio
from typing import Any, AsyncIterator, Iterator, Optional, List
from fastapi import Response
from starlette.responses import StreamingResponse

logger = logging.getLogger(__name__)


class StreamingOptimizer:
    """
    Response streaming optimizer
    
    Features:
    - Chunked response streaming
    - Memory-efficient large data handling
    - Progressive response delivery
    - Backpressure handling
    - Adaptive chunk sizing
    """
    
    def __init__(self, default_chunk_size: int = 8192, max_chunk_size: int = 65536):
        self.default_chunk_size = default_chunk_size
        self.max_chunk_size = max_chunk_size
        logger.info(f"✅ Streaming optimizer initialized (chunk_size: {default_chunk_size})")
    
    async def stream_json_array(
        self,
        items: List[Any],
        chunk_size: Optional[int] = None
    ) -> AsyncIterator[bytes]:
        """
        Stream JSON array in chunks
        
        Args:
            items: List of items to stream
            chunk_size: Items per chunk
            
        Yields:
            JSON chunks as bytes
        """
        from performance.serialization_optimizer import get_serializer
        serializer = get_serializer()
        
        chunk_size = chunk_size or 100  # Items per chunk
        
        # Stream opening bracket
        yield b'['
        
        first = True
        for i in range(0, len(items), chunk_size):
            chunk_items = items[i:i + chunk_size]
            
            # Serialize chunk
            chunk_json = serializer.serialize_json(chunk_items)
            
            # Remove brackets from chunk (we add them manually)
            if chunk_json.startswith(b'['):
                chunk_json = chunk_json[1:]
            if chunk_json.endswith(b']'):
                chunk_json = chunk_json[:-1]
            
            # Add comma separator
            if not first:
                yield b','
            else:
                first = False
            
            yield chunk_json
            
            # Yield control to event loop
            await asyncio.sleep(0)
        
        # Stream closing bracket
        yield b']'
    
    async def stream_large_data(
        self,
        data_generator: Iterator[Any],
        serializer: Optional[Any] = None
    ) -> AsyncIterator[bytes]:
        """
        Stream large data from generator
        
        Args:
            data_generator: Iterator yielding data chunks
            serializer: Optional serializer (uses default if None)
            
        Yields:
            Serialized chunks as bytes
        """
        if serializer is None:
            from performance.serialization_optimizer import get_serializer
            serializer = get_serializer()
        
        for item in data_generator:
            chunk = serializer.serialize_json(item)
            yield chunk
            await asyncio.sleep(0)  # Yield control
    
    def create_streaming_response(
        self,
        stream: AsyncIterator[bytes],
        media_type: str = "application/json",
        headers: Optional[dict] = None
    ) -> StreamingResponse:
        """
        Create streaming response
        
        Args:
            stream: Async iterator of bytes
            media_type: Response media type
            headers: Additional headers
            
        Returns:
            StreamingResponse
        """
        response_headers = {
            "Transfer-Encoding": "chunked",
            "X-Streaming": "true",
            **(headers or {})
        }
        
        return StreamingResponse(
            content=stream,
            media_type=media_type,
            headers=response_headers
        )
    
    async def stream_with_backpressure(
        self,
        stream: AsyncIterator[bytes],
        max_queue_size: int = 10
    ) -> AsyncIterator[bytes]:
        """
        Stream with backpressure handling
        
        Args:
            stream: Async iterator of bytes
            max_queue_size: Maximum queue size before backpressure
            
        Yields:
            Bytes chunks
        """
        queue = asyncio.Queue(maxsize=max_queue_size)
        
        async def producer():
            try:
                async for chunk in stream:
                    await queue.put(chunk)
                await queue.put(None)  # Sentinel
            except Exception as e:
                await queue.put(e)
        
        # Start producer
        producer_task = asyncio.create_task(producer())
        
        try:
            while True:
                item = await queue.get()
                
                if item is None:
                    break
                elif isinstance(item, Exception):
                    raise item
                
                yield item
                queue.task_done()
        finally:
            producer_task.cancel()
            try:
                await producer_task
            except asyncio.CancelledError:
                pass


# Global optimizer instance
_streaming_optimizer: Optional[StreamingOptimizer] = None


def get_streaming_optimizer() -> StreamingOptimizer:
    """Get global streaming optimizer instance"""
    global _streaming_optimizer
    if _streaming_optimizer is None:
        _streaming_optimizer = StreamingOptimizer()
    return _streaming_optimizer















