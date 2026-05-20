"""
Batch Processing Utilities
Advanced batch processing utilities
"""

import torch
from typing import List, Callable, Any, Optional
from functools import partial
import logging

logger = logging.getLogger(__name__)


class BatchProcessor:
    """
    Advanced batch processing
    """
    
    def __init__(
        self,
        batch_size: int = 32,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize batch processor
        
        Args:
            batch_size: Batch size
            device: Device for processing
        """
        self.batch_size = batch_size
        self.device = device or torch.device('cpu')
    
    def process_batches(
        self,
        items: List[Any],
        process_fn: Callable,
        collate_fn: Optional[Callable] = None,
    ) -> List[Any]:
        """
        Process items in batches
        
        Args:
            items: List of items to process
            process_fn: Function to process batch
            collate_fn: Function to collate items (optional)
            
        Returns:
            List of processed results
        """
        results = []
        
        for i in range(0, len(items), self.batch_size):
            batch = items[i:i + self.batch_size]
            
            if collate_fn:
                batch = collate_fn(batch)
            
            if isinstance(batch, torch.Tensor):
                batch = batch.to(self.device)
            
            batch_results = process_fn(batch)
            results.extend(batch_results if isinstance(batch_results, list) else [batch_results])
        
        return results
    
    def process_with_progress(
        self,
        items: List[Any],
        process_fn: Callable,
        collate_fn: Optional[Callable] = None,
        show_progress: bool = True,
    ) -> List[Any]:
        """
        Process items with progress bar
        
        Args:
            items: List of items to process
            process_fn: Function to process batch
            collate_fn: Function to collate items (optional)
            show_progress: Show progress bar
            
        Returns:
            List of processed results
        """
        try:
            from tqdm import tqdm
            use_tqdm = True
        except ImportError:
            use_tqdm = False
            logger.warning("tqdm not available, progress bar disabled")
        
        results = []
        num_batches = (len(items) + self.batch_size - 1) // self.batch_size
        
        iterator = range(0, len(items), self.batch_size)
        if show_progress and use_tqdm:
            iterator = tqdm(iterator, desc="Processing batches", total=num_batches)
        
        for i in iterator:
            batch = items[i:i + self.batch_size]
            
            if collate_fn:
                batch = collate_fn(batch)
            
            if isinstance(batch, torch.Tensor):
                batch = batch.to(self.device)
            
            batch_results = process_fn(batch)
            results.extend(batch_results if isinstance(batch_results, list) else [batch_results])
        
        return results



