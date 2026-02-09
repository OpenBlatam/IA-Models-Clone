"""
Batch Processing with Parallel Execution
=========================================

Parallel batch processing for image upscaling.
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional, Callable, Union
from pathlib import Path
from PIL import Image
import time

logger = logging.getLogger(__name__)


class BatchProcessor:
    """
    Parallel batch processor for image upscaling.
    
    Features:
    - Parallel processing
    - Progress tracking
    - Error handling
    - Resource management
    """
    
    def __init__(
        self,
        max_workers: int = 4,
        progress_callback: Optional[Callable] = None
    ):
        """
        Initialize batch processor.
        
        Args:
            max_workers: Maximum parallel workers
            progress_callback: Optional callback for progress updates
        """
        self.max_workers = max_workers
        self.progress_callback = progress_callback
        self.semaphore = asyncio.Semaphore(max_workers)
    
    async def process_batch(
        self,
        images: List[Union[Image.Image, str, Path]],
        process_func: Callable,
        *args,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Process batch of images in parallel.
        
        Args:
            images: List of images to process
            process_func: Async function to process each image
            *args: Additional positional arguments for process_func
            **kwargs: Additional keyword arguments for process_func
            
        Returns:
            List of results
        """
        total = len(images)
        results = []
        completed = 0
        start_time = time.time()
        
        async def process_single(
            index: int,
            image: Union[Image.Image, str, Path]
        ) -> Dict[str, Any]:
            """Process a single image with semaphore control."""
            async with self.semaphore:
                try:
                    result = await process_func(image, *args, **kwargs)
                    result["batch_index"] = index
                    result["success"] = True
                    return result
                except Exception as e:
                    logger.error(f"Error processing image {index}: {e}")
                    return {
                        "batch_index": index,
                        "success": False,
                        "error": str(e)
                    }
                finally:
                    nonlocal completed
                    completed += 1
                    if self.progress_callback:
                        progress = completed / total
                        elapsed = time.time() - start_time
                        eta = (elapsed / completed * (total - completed)) if completed > 0 else 0
                        self.progress_callback(completed, total, progress, eta)
        
        # Create tasks for all images
        tasks = [
            process_single(i, img)
            for i, img in enumerate(images)
        ]
        
        # Execute in parallel
        results = await asyncio.gather(*tasks)
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get batch processor statistics."""
        return {
            "max_workers": self.max_workers,
            "current_workers": self.max_workers - self.semaphore._value
        }


