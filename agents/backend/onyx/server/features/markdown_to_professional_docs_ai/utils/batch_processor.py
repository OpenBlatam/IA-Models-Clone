"""Advanced batch processing utilities"""
from typing import List, Dict, Any, Optional, Callable
import asyncio
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class AdvancedBatchProcessor:
    """Advanced batch processing with progress tracking"""
    
    def __init__(self, max_concurrent: int = 5):
        """
        Initialize batch processor
        
        Args:
            max_concurrent: Maximum concurrent operations
        """
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_batch(
        self,
        items: List[Any],
        processor: Callable,
        progress_callback: Optional[Callable] = None,
        error_handler: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Process batch of items with progress tracking
        
        Args:
            items: List of items to process
            processor: Async function to process each item
            progress_callback: Optional callback for progress updates
            error_handler: Optional error handler
            
        Returns:
            Batch processing results
        """
        results = {
            "total": len(items),
            "successful": 0,
            "failed": 0,
            "results": [],
            "errors": [],
            "start_time": datetime.now().isoformat()
        }
        
        async def process_item(item, index):
            async with self.semaphore:
                try:
                    result = await processor(item, index)
                    results["successful"] += 1
                    results["results"].append({
                        "index": index,
                        "item": str(item),
                        "result": result,
                        "status": "success"
                    })
                    
                    if progress_callback:
                        await progress_callback(index + 1, len(items), result)
                    
                    return result
                except Exception as e:
                    results["failed"] += 1
                    error_info = {
                        "index": index,
                        "item": str(item),
                        "error": str(e),
                        "status": "failed"
                    }
                    results["errors"].append(error_info)
                    results["results"].append(error_info)
                    
                    if error_handler:
                        await error_handler(item, index, e)
                    
                    logger.error(f"Error processing item {index}: {e}")
                    return None
        
        # Process all items
        tasks = [process_item(item, idx) for idx, item in enumerate(items)]
        await asyncio.gather(*tasks)
        
        results["end_time"] = datetime.now().isoformat()
        results["duration_seconds"] = (
            datetime.fromisoformat(results["end_time"]) - 
            datetime.fromisoformat(results["start_time"])
        ).total_seconds()
        
        return results
    
    async def process_with_retry(
        self,
        items: List[Any],
        processor: Callable,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ) -> Dict[str, Any]:
        """
        Process batch with retry logic
        
        Args:
            items: List of items
            processor: Processor function
            max_retries: Maximum retry attempts
            retry_delay: Delay between retries in seconds
            
        Returns:
            Processing results
        """
        async def processor_with_retry(item, index):
            last_error = None
            
            for attempt in range(max_retries):
                try:
                    return await processor(item, index)
                except Exception as e:
                    last_error = e
                    if attempt < max_retries - 1:
                        await asyncio.sleep(retry_delay * (attempt + 1))
                    else:
                        raise last_error
            
            raise last_error
        
        return await self.process_batch(items, processor_with_retry)


# Global batch processor
_batch_processor: Optional[AdvancedBatchProcessor] = None


def get_batch_processor(max_concurrent: int = 5) -> AdvancedBatchProcessor:
    """Get global batch processor"""
    global _batch_processor
    if _batch_processor is None:
        _batch_processor = AdvancedBatchProcessor(max_concurrent)
    return _batch_processor

