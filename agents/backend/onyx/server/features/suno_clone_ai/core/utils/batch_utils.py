"""
Batch Utilities

Utilities for batch processing and batching operations.
"""

import logging
import torch
from typing import List, Any, Optional, Callable
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


class BatchProcessor:
    """Process batches efficiently."""
    
    def __init__(
        self,
        batch_size: int = 32,
        collate_fn: Optional[Callable] = None
    ):
        """
        Initialize batch processor.
        
        Args:
            batch_size: Batch size
            collate_fn: Custom collate function
        """
        self.batch_size = batch_size
        self.collate_fn = collate_fn or self._default_collate
    
    @staticmethod
    def _default_collate(batch: List[Any]) -> Any:
        """
        Default collate function.
        
        Args:
            batch: List of samples
            
        Returns:
            Batched data
        """
        if isinstance(batch[0], torch.Tensor):
            return torch.stack(batch)
        elif isinstance(batch[0], dict):
            return {key: torch.stack([item[key] for item in batch]) for key in batch[0].keys()}
        else:
            return batch
    
    def create_dataloader(
        self,
        dataset: Dataset,
        shuffle: bool = True,
        num_workers: int = 0,
        pin_memory: bool = False
    ) -> DataLoader:
        """
        Create data loader.
        
        Args:
            dataset: Dataset
            shuffle: Whether to shuffle
            num_workers: Number of worker processes
            pin_memory: Pin memory for faster GPU transfer
            
        Returns:
            DataLoader instance
        """
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=self.collate_fn
        )
    
    def process_batches(
        self,
        data: List[Any],
        process_fn: Callable
    ) -> List[Any]:
        """
        Process data in batches.
        
        Args:
            data: List of data items
            process_fn: Processing function
            
        Returns:
            List of processed items
        """
        results = []
        
        for i in range(0, len(data), self.batch_size):
            batch = data[i:i + self.batch_size]
            batch_results = process_fn(batch)
            
            if isinstance(batch_results, list):
                results.extend(batch_results)
            else:
                results.append(batch_results)
        
        return results


def create_batch_processor(
    batch_size: int = 32,
    **kwargs
) -> BatchProcessor:
    """Create batch processor."""
    return BatchProcessor(batch_size, **kwargs)


def process_in_batches(
    data: List[Any],
    process_fn: Callable,
    batch_size: int = 32
) -> List[Any]:
    """Process data in batches."""
    processor = BatchProcessor(batch_size)
    return processor.process_batches(data, process_fn)



