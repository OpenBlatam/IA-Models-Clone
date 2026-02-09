"""Batch processing utilities."""
from typing import List, Any, Dict, Callable, Awaitable, Optional
from asyncio import gather, Semaphore


async def process_batch(
    items: List[Any],
    processor: Callable[[Any], Awaitable[Any]],
    max_concurrent: int = 10,
    batch_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Process a batch of items concurrently with rate limiting.
    
    Args:
        items: List of items to process
        processor: Async function to process each item
        max_concurrent: Maximum concurrent operations
        batch_id: Optional batch identifier
    
    Returns:
        Dictionary with results and statistics
    """
    semaphore = Semaphore(max_concurrent)
    results = []
    errors = []
    
    async def process_item(item):
        async with semaphore:
            try:
                result = await processor(item)
                return {"success": True, "result": result, "item": item}
            except Exception as e:
                return {"success": False, "error": str(e), "item": item}
    
    # Process all items
    task_results = await gather(*[process_item(item) for item in items])
    
    # Separate successes and errors
    for result in task_results:
        if result["success"]:
            results.append(result["result"])
        else:
            errors.append({
                "item": result["item"],
                "error": result["error"]
            })
    
    return {
        "batch_id": batch_id,
        "total": len(items),
        "successful": len(results),
        "failed": len(errors),
        "results": results,
        "errors": errors,
        "success_rate": len(results) / len(items) if items else 0
    }


class BatchProcessor:
    """Batch processing manager."""
    
    def __init__(self, max_concurrent: int = 10):
        self.max_concurrent = max_concurrent
        self.active_batches: Dict[str, Dict[str, Any]] = {}
    
    async def create_batch(
        self,
        items: List[Any],
        processor: Callable[[Any], Awaitable[Any]],
        batch_id: Optional[str] = None
    ) -> str:
        """Create and start a batch job."""
        from uuid import uuid4
        
        if batch_id is None:
            batch_id = str(uuid4())
        
        # Store batch info
        self.active_batches[batch_id] = {
            "status": "processing",
            "total": len(items),
            "completed": 0
        }
        
        # Process batch in background
        async def process_and_update():
            result = await process_batch(items, processor, self.max_concurrent, batch_id)
            self.active_batches[batch_id].update({
                "status": "completed",
                "result": result
            })
        
        import asyncio
        asyncio.create_task(process_and_update())
        
        return batch_id
    
    def get_batch_status(self, batch_id: str) -> Optional[Dict[str, Any]]:
        """Get batch processing status."""
        return self.active_batches.get(batch_id)

