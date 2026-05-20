"""
Batch Processor Base Utility
=============================

Base utility for batch processing operations.
"""

import logging
from typing import List, Dict, Any, Callable, Optional

logger = logging.getLogger(__name__)


class BatchProcessorBase:
    """Base class for batch processing operations."""
    
    @staticmethod
    def process_item(
        item: Dict[str, Any],
        index: int,
        total: int,
        process_fn: Callable,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process a single batch item.
        
        Args:
            item: Item to process
            index: Item index (0-based)
            total: Total number of items
            process_fn: Function to process the item
            **kwargs: Additional arguments for process_fn
            
        Returns:
            Result dictionary with batch metadata
        """
        logger.info(f"Processing batch item {index+1}/{total}")
        
        try:
            result = process_fn(
                image=item["image"],
                clothing_description=item["clothing_description"],
                character_name=item.get("character_name"),
                **kwargs
            )
            result["batch_index"] = index
            result["batch_total"] = total
            result["success"] = True
            return result
        except Exception as e:
            logger.error(f"Error processing batch item {index+1}: {e}")
            return {
                "batch_index": index,
                "batch_total": total,
                "error": str(e),
                "success": False,
            }
    
    @staticmethod
    def process_batch(
        items: List[Dict[str, Any]],
        process_fn: Callable,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Process a batch of items.
        
        Args:
            items: List of items to process
            process_fn: Function to process each item
            progress_callback: Optional callback(current, total)
            **kwargs: Additional arguments for process_fn
            
        Returns:
            List of result dictionaries
        """
        results = []
        total = len(items)
        
        for i, item in enumerate(items):
            # Call progress callback if provided
            if progress_callback:
                progress_callback(i + 1, total)
            
            # Process item
            result = BatchProcessorBase.process_item(
                item=item,
                index=i,
                total=total,
                process_fn=process_fn,
                **kwargs
            )
            results.append(result)
        
        # Final progress callback
        if progress_callback:
            progress_callback(total, total)
        
        return results

