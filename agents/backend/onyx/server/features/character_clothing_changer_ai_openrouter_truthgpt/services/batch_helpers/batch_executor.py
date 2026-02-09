"""
Batch Executor
==============
Executes batch operations with concurrency control
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable, Awaitable

from .batch_item_processor import BatchItemProcessor
from .batch_validator import BatchValidator

logger = logging.getLogger(__name__)


class BatchExecutor:
    """
    Executes batch operations with concurrency control.
    """
    
    def __init__(
        self,
        item_processor: BatchItemProcessor,
        max_concurrent: int = 5
    ):
        """
        Initialize batch executor.
        
        Args:
            item_processor: Batch item processor instance
            max_concurrent: Maximum concurrent operations
        """
        self.item_processor = item_processor
        self.max_concurrent = BatchValidator.validate_max_concurrent(max_concurrent)
        self.semaphore = asyncio.Semaphore(self.max_concurrent)
    
    async def execute_batch(
        self,
        items: List[Dict[str, Any]],
        operation_type: str,
        progress_callback: Optional[Callable[[int, int], Awaitable[None]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute batch processing with concurrency control.
        
        Args:
            items: List of items to process
            operation_type: Type of operation ("clothing_change" or "face_swap")
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of processing results
        """
        results = []
        total_items = len(items)
        
        async def process_item(item: Dict[str, Any], item_id: str, index: int):
            """Process a single item with semaphore control"""
            async with self.semaphore:
                started_at = datetime.now()
                try:
                    if operation_type == "clothing_change":
                        result = await self.item_processor.process_clothing_change_item(
                            item, item_id
                        )
                    else:  # face_swap
                        result = await self.item_processor.process_face_swap_item(
                            item, item_id
                        )
                    
                    completed_at = datetime.now()
                    item_result = self.item_processor.build_item_result(
                        item_id, result, started_at, completed_at
                    )
                    
                    if progress_callback:
                        await progress_callback(index + 1, total_items)
                    
                    return item_result
                except Exception as e:
                    completed_at = datetime.now()
                    logger.error(f"Unexpected error processing item {item_id}: {e}", exc_info=True)
                    error_result = self.item_processor.build_item_error(
                        item_id, str(e), started_at, completed_at
                    )
                    
                    if progress_callback:
                        await progress_callback(index + 1, total_items)
                    
                    return error_result
        
        # Create tasks for all items
        tasks = []
        for i, item in enumerate(items):
            item_id = item.get("item_id") or f"item_{i}"
            task = process_item(item, item_id, i)
            tasks.append(task)
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions that weren't caught
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Exception in batch item {i}: {result}", exc_info=True)
                error_result = self.item_processor.build_item_error(
                    f"item_{i}", str(result)
                )
                processed_results.append(error_result)
            else:
                processed_results.append(result)
        
        return processed_results

