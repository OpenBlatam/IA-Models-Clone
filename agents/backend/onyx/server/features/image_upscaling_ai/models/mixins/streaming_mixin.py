"""
Streaming Mixin

Contains streaming and real-time processing functionality.
"""

import logging
import asyncio
from typing import Union, Dict, Any, Optional, Callable, AsyncGenerator
from pathlib import Path
from PIL import Image
from io import BytesIO

logger = logging.getLogger(__name__)


class StreamingMixin:
    """
    Mixin providing streaming and real-time processing functionality.
    
    This mixin contains:
    - Streaming upscaling
    - Progressive processing
    - Real-time updates
    - Chunk-based processing
    - Async generators
    """
    
    async def stream_upscale(
        self,
        image: Union[Image.Image, str, Path],
        scale_factor: float,
        method: str = "lanczos",
        chunk_size: int = 256
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream upscaling process with progress updates.
        
        Args:
            image: Input image
            scale_factor: Scale factor
            method: Upscaling method
            chunk_size: Size of processing chunks
            
        Yields:
            Progress updates during processing
        """
        if isinstance(image, (str, Path)):
            img = Image.open(image).convert("RGB")
        else:
            img = image.convert("RGB")
        
        width, height = img.size
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        
        # Yield initial progress
        yield {
            "stage": "initialized",
            "progress": 0.0,
            "message": "Starting upscaling...",
        }
        
        # Process in chunks (simulated for demonstration)
        # In real implementation, this would process image regions
        total_chunks = (new_width // chunk_size) * (new_height // chunk_size)
        processed_chunks = 0
        
        # Simulate chunk processing
        for y in range(0, new_height, chunk_size):
            for x in range(0, new_width, chunk_size):
                await asyncio.sleep(0.01)  # Simulate processing
                processed_chunks += 1
                progress = processed_chunks / total_chunks if total_chunks > 0 else 1.0
                
                yield {
                    "stage": "processing",
                    "progress": progress,
                    "message": f"Processing chunk {processed_chunks}/{total_chunks}",
                    "chunk": {"x": x, "y": y},
                }
        
        # Final upscaling
        yield {
            "stage": "finalizing",
            "progress": 0.95,
            "message": "Finalizing upscaling...",
        }
        
        # Perform actual upscaling
        result = self.upscale(img, scale_factor, method, return_metrics=False)
        
        yield {
            "stage": "completed",
            "progress": 1.0,
            "message": "Upscaling completed",
            "result": result,
        }
    
    async def progressive_upscale(
        self,
        image: Union[Image.Image, str, Path],
        target_scale: float,
        steps: int = 3,
        method: str = "lanczos",
        callback: Optional[Callable[[Image.Image, float], None]] = None
    ) -> Image.Image:
        """
        Progressive upscaling in multiple steps.
        
        Args:
            image: Input image
            target_scale: Target scale factor
            steps: Number of progressive steps
            method: Upscaling method
            callback: Optional callback for each step
            
        Returns:
            Final upscaled image
        """
        if isinstance(image, (str, Path)):
            current_image = Image.open(image).convert("RGB")
        else:
            current_image = image.convert("RGB")
        
        scale_per_step = target_scale ** (1.0 / steps)
        current_scale = 1.0
        
        for step in range(steps):
            current_scale *= scale_per_step
            current_image = self.upscale(
                current_image, scale_per_step, method, return_metrics=False
            )
            
            if callback:
                callback(current_image, current_scale)
            
            await asyncio.sleep(0.1)  # Allow other tasks
        
        return current_image
    
    def process_image_stream(
        self,
        image_stream: BytesIO,
        scale_factor: float,
        method: str = "lanczos"
    ) -> BytesIO:
        """
        Process image from stream.
        
        Args:
            image_stream: Image data stream
            scale_factor: Scale factor
            method: Upscaling method
            
        Returns:
            Processed image stream
        """
        image = Image.open(image_stream).convert("RGB")
        result = self.upscale(image, scale_factor, method, return_metrics=False)
        
        output_stream = BytesIO()
        result.save(output_stream, format='PNG')
        output_stream.seek(0)
        
        return output_stream
    
    async def batch_stream_upscale(
        self,
        image_paths: List[Union[str, Path]],
        scale_factor: float,
        method: str = "lanczos",
        max_concurrent: int = 3
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream batch upscaling with progress.
        
        Args:
            image_paths: List of image paths
            scale_factor: Scale factor
            method: Upscaling method
            max_concurrent: Maximum concurrent operations
            
        Yields:
            Progress updates for each image
        """
        total = len(image_paths)
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_one(index: int, path: Union[str, Path]):
            async with semaphore:
                try:
                    result = self.upscale(path, scale_factor, method, return_metrics=False)
                    return {
                        "index": index,
                        "path": str(path),
                        "success": True,
                        "result": result,
                    }
                except Exception as e:
                    return {
                        "index": index,
                        "path": str(path),
                        "success": False,
                        "error": str(e),
                    }
        
        tasks = [process_one(i, path) for i, path in enumerate(image_paths)]
        
        completed = 0
        for coro in asyncio.as_completed(tasks):
            result = await coro
            completed += 1
            
            yield {
                "progress": completed / total,
                "completed": completed,
                "total": total,
                "result": result,
            }


