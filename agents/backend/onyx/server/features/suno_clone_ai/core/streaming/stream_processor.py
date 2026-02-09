"""
Stream Processor

Utilities for processing data streams.
"""

import logging
from typing import Iterator, Callable, Any, Optional
from queue import Queue
import threading

logger = logging.getLogger(__name__)


class StreamProcessor:
    """Process data streams."""
    
    def __init__(
        self,
        process_fn: Callable,
        buffer_size: int = 100
    ):
        """
        Initialize stream processor.
        
        Args:
            process_fn: Processing function
            buffer_size: Buffer size
        """
        self.process_fn = process_fn
        self.buffer_size = buffer_size
        self.queue = Queue(maxsize=buffer_size)
        self.processing = False
        self.thread = None
    
    def process(
        self,
        stream: Iterator[Any]
    ) -> Iterator[Any]:
        """
        Process stream.
        
        Args:
            stream: Input stream
            
        Returns:
            Processed stream
        """
        for item in stream:
            try:
                processed = self.process_fn(item)
                yield processed
            except Exception as e:
                logger.error(f"Stream processing error: {e}")
                continue
    
    def process_async(
        self,
        stream: Iterator[Any]
    ) -> Iterator[Any]:
        """
        Process stream asynchronously.
        
        Args:
            stream: Input stream
            
        Returns:
            Processed stream
        """
        self.processing = True
        self.thread = threading.Thread(target=self._process_worker, args=(stream,), daemon=True)
        self.thread.start()
        
        while self.processing or not self.queue.empty():
            try:
                item = self.queue.get(timeout=0.1)
                yield item
            except:
                continue
    
    def _process_worker(self, stream: Iterator[Any]) -> None:
        """Worker thread for async processing."""
        try:
            for item in stream:
                if not self.processing:
                    break
                
                try:
                    processed = self.process_fn(item)
                    self.queue.put(processed)
                except Exception as e:
                    logger.error(f"Processing error: {e}")
        finally:
            self.processing = False
    
    def stop(self) -> None:
        """Stop processing."""
        self.processing = False
        if self.thread:
            self.thread.join(timeout=2.0)


def process_stream(
    stream: Iterator[Any],
    process_fn: Callable
) -> Iterator[Any]:
    """
    Process stream with function.
    
    Args:
        stream: Input stream
        process_fn: Processing function
        
    Returns:
        Processed stream
    """
    processor = StreamProcessor(process_fn)
    return processor.process(stream)


def create_stream_pipeline(
    *processors: Callable
) -> Callable:
    """
    Create stream processing pipeline.
    
    Args:
        *processors: Processing functions
        
    Returns:
        Pipeline function
    """
    def pipeline(stream: Iterator[Any]) -> Iterator[Any]:
        current_stream = stream
        for processor in processors:
            current_stream = process_stream(current_stream, processor)
        return current_stream
    
    return pipeline



