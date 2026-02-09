"""
Data Stream

Utilities for creating and managing data streams.
"""

import logging
from typing import Iterator, Any, Optional, Callable
from queue import Queue
import threading

logger = logging.getLogger(__name__)


class DataStream:
    """Data stream manager."""
    
    def __init__(
        self,
        source: Optional[Iterator[Any]] = None,
        buffer_size: int = 100
    ):
        """
        Initialize data stream.
        
        Args:
            source: Source iterator
            buffer_size: Buffer size
        """
        self.source = source
        self.buffer_size = buffer_size
        self.queue = Queue(maxsize=buffer_size)
        self.streaming = False
        self.thread = None
    
    def start(self) -> None:
        """Start streaming."""
        if self.streaming:
            return
        
        if self.source is None:
            raise ValueError("No source provided")
        
        self.streaming = True
        self.thread = threading.Thread(target=self._stream_worker, daemon=True)
        self.thread.start()
        logger.info("Data stream started")
    
    def stop(self) -> None:
        """Stop streaming."""
        self.streaming = False
        if self.thread:
            self.thread.join(timeout=2.0)
        logger.info("Data stream stopped")
    
    def _stream_worker(self) -> None:
        """Worker thread for streaming."""
        try:
            for item in self.source:
                if not self.streaming:
                    break
                
                try:
                    self.queue.put(item, timeout=1.0)
                except:
                    if not self.streaming:
                        break
        finally:
            self.streaming = False
    
    def __iter__(self) -> Iterator[Any]:
        """Iterate over stream."""
        if not self.streaming:
            self.start()
        
        while self.streaming or not self.queue.empty():
            try:
                item = self.queue.get(timeout=0.1)
                yield item
            except:
                if not self.streaming:
                    break


def create_data_stream(
    source: Iterator[Any],
    buffer_size: int = 100
) -> DataStream:
    """
    Create data stream.
    
    Args:
        source: Source iterator
        buffer_size: Buffer size
        
    Returns:
        DataStream instance
    """
    return DataStream(source, buffer_size)



