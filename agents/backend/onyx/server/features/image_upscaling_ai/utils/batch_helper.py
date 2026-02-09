"""
Batch Helper
============

Helper functions for batch processing.
"""

import logging
import asyncio
from typing import List, Callable, Any, Optional, Dict
from pathlib import Path
from PIL import Image

logger = logging.getLogger(__name__)


class BatchHelper:
    """
    Helper for batch processing operations.
    
    Features:
    - Batch preparation
    - Progress tracking
    - Error handling
    - Result aggregation
    """
    
    @staticmethod
    def prepare_images(
        image_paths: List[Union[str, Path]],
        validate: bool = True
    ) -> List[Image.Image]:
        """
        Prepare images from paths.
        
        Args:
            image_paths: List of image paths
            validate: Validate images before returning
            
        Returns:
            List of PIL Images
        """
        images = []
        
        for path in image_paths:
            try:
                img_path = Path(path)
                if not img_path.exists():
                    logger.warning(f"Image not found: {path}")
                    continue
                
                img = Image.open(img_path).convert("RGB")
                
                if validate:
                    # Basic validation
                    if img.size[0] < 32 or img.size[1] < 32:
                        logger.warning(f"Image too small: {path}")
                        continue
                
                images.append(img)
                
            except Exception as e:
                logger.error(f"Error loading image {path}: {e}")
                continue
        
        return images
    
    @staticmethod
    async def process_batch_with_progress(
        items: List[Any],
        process_func: Callable,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        max_concurrent: int = 2
    ) -> List[Dict[str, Any]]:
        """
        Process batch with progress tracking.
        
        Args:
            items: List of items to process
            process_func: Async function to process items
            progress_callback: Progress callback
            max_concurrent: Maximum concurrent operations
            
        Returns:
            List of results
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        results = []
        
        async def process_with_semaphore(item, index):
            async with semaphore:
                try:
                    result = await process_func(item)
                    if progress_callback:
                        progress_callback(index + 1, len(items))
                    return {"success": True, "result": result, "index": index}
                except Exception as e:
                    logger.error(f"Error processing item {index}: {e}")
                    if progress_callback:
                        progress_callback(index + 1, len(items))
                    return {"success": False, "error": str(e), "index": index}
        
        tasks = [
            process_with_semaphore(item, idx)
            for idx, item in enumerate(items)
        ]
        
        results = await asyncio.gather(*tasks)
        return results
    
    @staticmethod
    def aggregate_results(
        results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Aggregate batch results.
        
        Args:
            results: List of result dictionaries
            
        Returns:
            Aggregated statistics
        """
        total = len(results)
        successful = sum(1 for r in results if r.get("success", False))
        failed = total - successful
        
        processing_times = [
            r.get("processing_time", 0.0)
            for r in results
            if r.get("success", False) and "processing_time" in r
        ]
        
        avg_time = (
            sum(processing_times) / len(processing_times)
            if processing_times else 0.0
        )
        
        return {
            "total": total,
            "successful": successful,
            "failed": failed,
            "success_rate": successful / total if total > 0 else 0.0,
            "avg_processing_time": avg_time,
            "total_processing_time": sum(processing_times)
        }


