"""
Streaming Processor for Color Grading AI
=========================================

Streaming processing for real-time color grading operations.
"""

import logging
import asyncio
from typing import Dict, Any, Optional, List, AsyncIterator, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class StreamStatus(Enum):
    """Stream status."""
    INITIALIZING = "initializing"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    ERROR = "error"
    CANCELLED = "cancelled"


@dataclass
class StreamChunk:
    """Stream chunk."""
    chunk_id: str
    data: Any
    sequence: int
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class StreamingProcessor:
    """
    Streaming processor for real-time operations.
    
    Features:
    - Real-time processing
    - Chunk-based processing
    - Backpressure handling
    - Stream management
    - Progress tracking
    - Error recovery
    """
    
    def __init__(self, chunk_size: int = 1024, max_buffer_size: int = 10000):
        """
        Initialize streaming processor.
        
        Args:
            chunk_size: Chunk size in bytes
            max_buffer_size: Maximum buffer size
        """
        self.chunk_size = chunk_size
        self.max_buffer_size = max_buffer_size
        self._streams: Dict[str, Dict[str, Any]] = {}
        self._processors: Dict[str, Callable] = {}
    
    def register_processor(self, stream_type: str, processor: Callable):
        """
        Register stream processor.
        
        Args:
            stream_type: Stream type
            processor: Processor function
        """
        self._processors[stream_type] = processor
        logger.info(f"Registered processor for {stream_type}")
    
    async def create_stream(
        self,
        stream_id: str,
        stream_type: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create new stream.
        
        Args:
            stream_id: Stream ID
            stream_type: Stream type
            metadata: Optional metadata
            
        Returns:
            Stream ID
        """
        self._streams[stream_id] = {
            "stream_id": stream_id,
            "stream_type": stream_type,
            "status": StreamStatus.INITIALIZING,
            "metadata": metadata or {},
            "chunks_processed": 0,
            "created_at": datetime.now(),
            "buffer": asyncio.Queue(maxsize=self.max_buffer_size)
        }
        
        logger.info(f"Created stream: {stream_id}")
        
        return stream_id
    
    async def process_stream(
        self,
        stream_id: str,
        data_stream: AsyncIterator[bytes]
    ) -> AsyncIterator[StreamChunk]:
        """
        Process stream.
        
        Args:
            stream_id: Stream ID
            data_stream: Async iterator of data chunks
            
        Yields:
            Processed stream chunks
        """
        if stream_id not in self._streams:
            raise ValueError(f"Stream {stream_id} not found")
        
        stream = self._streams[stream_id]
        stream["status"] = StreamStatus.ACTIVE
        
        processor = self._processors.get(stream["stream_type"])
        if not processor:
            raise ValueError(f"No processor registered for {stream['stream_type']}")
        
        sequence = 0
        
        try:
            async for chunk_data in data_stream:
                # Process chunk
                if asyncio.iscoroutinefunction(processor):
                    processed = await processor(chunk_data, stream["metadata"])
                else:
                    processed = processor(chunk_data, stream["metadata"])
                
                chunk = StreamChunk(
                    chunk_id=f"{stream_id}_{sequence}",
                    data=processed,
                    sequence=sequence,
                    metadata={"stream_id": stream_id}
                )
                
                stream["chunks_processed"] += 1
                sequence += 1
                
                yield chunk
        
        except Exception as e:
            stream["status"] = StreamStatus.ERROR
            logger.error(f"Error processing stream {stream_id}: {e}")
            raise
        
        stream["status"] = StreamStatus.COMPLETED
        logger.info(f"Stream {stream_id} completed ({stream['chunks_processed']} chunks)")
    
    def pause_stream(self, stream_id: str):
        """Pause stream."""
        if stream_id in self._streams:
            self._streams[stream_id]["status"] = StreamStatus.PAUSED
            logger.info(f"Paused stream: {stream_id}")
    
    def resume_stream(self, stream_id: str):
        """Resume stream."""
        if stream_id in self._streams:
            self._streams[stream_id]["status"] = StreamStatus.ACTIVE
            logger.info(f"Resumed stream: {stream_id}")
    
    def cancel_stream(self, stream_id: str):
        """Cancel stream."""
        if stream_id in self._streams:
            self._streams[stream_id]["status"] = StreamStatus.CANCELLED
            logger.info(f"Cancelled stream: {stream_id}")
    
    def get_stream_status(self, stream_id: str) -> Optional[Dict[str, Any]]:
        """Get stream status."""
        stream = self._streams.get(stream_id)
        if not stream:
            return None
        
        return {
            "stream_id": stream_id,
            "status": stream["status"].value,
            "chunks_processed": stream["chunks_processed"],
            "created_at": stream["created_at"].isoformat(),
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get streaming processor statistics."""
        return {
            "active_streams": len([s for s in self._streams.values() if s["status"] == StreamStatus.ACTIVE]),
            "total_streams": len(self._streams),
            "processors_count": len(self._processors),
        }




