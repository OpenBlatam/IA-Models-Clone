"""
Stream Processor
================

Real-time stream processing.
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional, Callable, AsyncIterator
from collections import deque

logger = logging.getLogger(__name__)


class StreamProcessor:
    """Real-time stream processor."""
    
    def __init__(self, buffer_size: int = 1000):
        self.buffer_size = buffer_size
        self._streams: Dict[str, deque] = {}
        self._processors: Dict[str, List[Callable]] = {}
        self._running = False
    
    def register_stream(self, stream_id: str):
        """Register stream."""
        self._streams[stream_id] = deque(maxlen=self.buffer_size)
        self._processors[stream_id] = []
        logger.info(f"Registered stream: {stream_id}")
    
    def add_processor(self, stream_id: str, processor: Callable):
        """Add processor to stream."""
        if stream_id not in self._processors:
            self._processors[stream_id] = []
        
        self._processors[stream_id].append(processor)
        logger.info(f"Added processor to stream: {stream_id}")
    
    async def process_item(self, stream_id: str, item: Any):
        """Process single item."""
        if stream_id not in self._streams:
            self.register_stream(stream_id)
        
        # Add to stream buffer
        self._streams[stream_id].append(item)
        
        # Process with all processors
        for processor in self._processors.get(stream_id, []):
            try:
                if asyncio.iscoroutinefunction(processor):
                    await processor(item)
                else:
                    processor(item)
            except Exception as e:
                logger.error(f"Processor failed for {stream_id}: {e}")
    
    async def process_stream(self, stream_id: str, items: AsyncIterator[Any]):
        """Process stream of items."""
        async for item in items:
            await self.process_item(stream_id, item)
    
    def get_stream_buffer(self, stream_id: str, limit: int = 100) -> List[Any]:
        """Get stream buffer."""
        if stream_id not in self._streams:
            return []
        
        buffer = list(self._streams[stream_id])
        return buffer[-limit:]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processor statistics."""
        return {
            "streams": len(self._streams),
            "stream_info": {
                stream_id: {
                    "buffer_size": len(buffer),
                    "processors": len(self._processors.get(stream_id, []))
                }
                for stream_id, buffer in self._streams.items()
            }
        }















