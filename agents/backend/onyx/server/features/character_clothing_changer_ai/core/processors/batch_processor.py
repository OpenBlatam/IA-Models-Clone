"""
Batch Processor
===============

Handles batch processing of clothing changes.
"""

import logging
from typing import List, Dict, Any, Callable, Optional

from ..utils.batch_processor_base import BatchProcessorBase

logger = logging.getLogger(__name__)


class BatchProcessor:
    """Processes multiple clothing changes in batch."""
    
    def __init__(self, change_clothing_fn: Callable):
        """
        Initialize batch processor.
        
        Args:
            change_clothing_fn: Function to call for each item
        """
        self.change_clothing_fn = change_clothing_fn
    
    def process_batch(
        self,
        image_clothing_pairs: List[Dict[str, Any]],
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Process batch of clothing changes.
        
        Args:
            image_clothing_pairs: List of dicts with 'image' and 'clothing_description' keys
            **kwargs: Additional arguments passed to change_clothing_fn
            
        Returns:
            List of result dicts
        """
        return BatchProcessorBase.process_batch(
            items=image_clothing_pairs,
            process_fn=self.change_clothing_fn,
            progress_callback=None,
            **kwargs
        )
    
    def process_batch_with_progress(
        self,
        image_clothing_pairs: List[Dict[str, Any]],
        progress_callback: Optional[Callable[[int, int], None]] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Process batch with progress callback.
        
        Args:
            image_clothing_pairs: List of dicts with 'image' and 'clothing_description' keys
            progress_callback: Optional callback(current, total)
            **kwargs: Additional arguments passed to change_clothing_fn
            
        Returns:
            List of result dicts
        """
        return BatchProcessorBase.process_batch(
            items=image_clothing_pairs,
            process_fn=self.change_clothing_fn,
            progress_callback=progress_callback,
            **kwargs
        )

