"""
Streaming Processor
===================

Streaming processing for real-time upscaling.
"""

import logging
import asyncio
from typing import AsyncIterator, Optional, Callable, Any
from PIL import Image
import time

logger = logging.getLogger(__name__)


class StreamingProcessor:
    """
    Streaming processor for real-time upscaling.
    
    Features:
    - Async streaming
    - Progress callbacks
    - Error handling
    - Backpressure management
    """
    
    def __init__(
        self,
        upscale_func: Callable,
        max_queue_size: int = 10,
        timeout: float = 30.0
    ):
        """
        Initialize streaming processor.
        
        Args:
            upscale_func: Function to upscale images
            max_queue_size: Maximum queue size
            timeout: Operation timeout in seconds
        """
        self.upscale_func = upscale_func
        self.max_queue_size = max_queue_size
        self.timeout = timeout
        self.queue = asyncio.Queue(maxsize=max_queue_size)
        self.processing = False
    
    async def process_stream(
        self,
        image_stream: AsyncIterator[Image.Image],
        scale_factor: float,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> AsyncIterator[Image.Image]:
        """
        Process stream of images.
        
        Args:
            image_stream: Async iterator of images
            scale_factor: Scale factor
            progress_callback: Progress callback
            
        Yields:
            Upscaled images
        """
        self.processing = True
        processed_count = 0
        total_count = 0
        
        try:
            # Start producer task
            producer_task = asyncio.create_task(
                self._producer(image_stream)
            )
            
            # Process images
            while self.processing or not self.queue.empty():
                try:
                    # Get image from queue with timeout
                    image = await asyncio.wait_for(
                        self.queue.get(),
                        timeout=1.0
                    )
                    
                    if image is None:  # End marker
                        break
                    
                    total_count += 1
                    
                    # Upscale
                    try:
                        upscaled = await asyncio.wait_for(
                            self._upscale_image(image, scale_factor),
                            timeout=self.timeout
                        )
                        
                        processed_count += 1
                        
                        if progress_callback:
                            progress_callback(processed_count, total_count)
                        
                        yield upscaled
                        
                    except asyncio.TimeoutError:
                        logger.error(f"Timeout upscaling image {total_count}")
                        continue
                    except Exception as e:
                        logger.error(f"Error upscaling image {total_count}: {e}")
                        continue
                    
                except asyncio.TimeoutError:
                    # Check if producer is done
                    if producer_task.done():
                        break
                    continue
            
            # Wait for producer to finish
            await producer_task
            
        finally:
            self.processing = False
    
    async def _producer(
        self,
        image_stream: AsyncIterator[Image.Image]
    ) -> None:
        """Produce images from stream."""
        try:
            async for image in image_stream:
                await self.queue.put(image)
        except Exception as e:
            logger.error(f"Error in producer: {e}")
        finally:
            # Put end marker
            await self.queue.put(None)
    
    async def _upscale_image(
        self,
        image: Image.Image,
        scale_factor: float
    ) -> Image.Image:
        """Upscale single image."""
        if asyncio.iscoroutinefunction(self.upscale_func):
            return await self.upscale_func(image, scale_factor)
        else:
            # Run in executor
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                self.upscale_func,
                image,
                scale_factor
            )
    
    def stop(self) -> None:
        """Stop processing."""
        self.processing = False


