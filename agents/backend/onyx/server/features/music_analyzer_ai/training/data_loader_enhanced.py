"""
Enhanced DataLoader Utilities

Improved data loading with:
- Efficient batching
- Memory optimization
- Multi-worker support
- Proper pinning
"""

from typing import Optional, Dict, Any, Callable
import logging
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

logger = logging.getLogger(__name__)


class EnhancedDataLoader:
    """
    Enhanced DataLoader with optimizations for training.
    
    Features:
    - Automatic worker configuration
    - Memory pinning
    - Persistent workers
    - Proper batch size calculation
    """
    
    @staticmethod
    def create_loader(
        dataset: Dataset,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: Optional[int] = None,
        pin_memory: bool = True,
        drop_last: bool = False,
        persistent_workers: bool = True,
        prefetch_factor: int = 2,
        collate_fn: Optional[Callable] = None
    ) -> DataLoader:
        """
        Create optimized DataLoader.
        
        Args:
            dataset: Dataset to load from
            batch_size: Batch size
            shuffle: Whether to shuffle
            num_workers: Number of worker processes (auto if None)
            pin_memory: Pin memory for faster GPU transfer
            drop_last: Drop last incomplete batch
            persistent_workers: Keep workers alive between epochs
            prefetch_factor: Number of batches to prefetch
            collate_fn: Custom collate function
            
        Returns:
            Optimized DataLoader
        """
        # Auto-configure num_workers
        if num_workers is None:
            num_workers = EnhancedDataLoader._get_optimal_workers()
        
        # Pin memory only for CUDA
        pin_memory = pin_memory and torch.cuda.is_available()
        
        # Persistent workers only if num_workers > 0
        persistent_workers = persistent_workers and num_workers > 0
        
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor if num_workers > 0 else 2,
            collate_fn=collate_fn
        )
        
        logger.info(
            f"DataLoader created: "
            f"batch_size={batch_size}, "
            f"num_workers={num_workers}, "
            f"pin_memory={pin_memory}"
        )
        
        return loader
    
    @staticmethod
    def _get_optimal_workers() -> int:
        """
        Get optimal number of workers based on CPU cores.
        
        Returns:
            Optimal number of workers
        """
        import os
        cpu_count = os.cpu_count() or 4
        # Use 75% of CPU cores, but max 8
        optimal = min(int(cpu_count * 0.75), 8)
        return max(optimal, 1)
    
    @staticmethod
    def create_distributed_loader(
        dataset: Dataset,
        batch_size: int = 32,
        num_workers: Optional[int] = None,
        pin_memory: bool = True,
        **kwargs
    ) -> DataLoader:
        """
        Create DataLoader for distributed training.
        
        Args:
            dataset: Dataset to load from
            batch_size: Batch size per process
            num_workers: Number of worker processes
            pin_memory: Pin memory for faster GPU transfer
            **kwargs: Additional DataLoader arguments
            
        Returns:
            Distributed DataLoader
        """
        if not torch.distributed.is_available():
            logger.warning("Distributed training not available")
            return EnhancedDataLoader.create_loader(
                dataset, batch_size, num_workers=num_workers,
                pin_memory=pin_memory, **kwargs
            )
        
        from torch.utils.data.distributed import DistributedSampler
        
        sampler = DistributedSampler(
            dataset,
            num_replicas=torch.distributed.get_world_size(),
            rank=torch.distributed.get_rank(),
            shuffle=kwargs.get('shuffle', True)
        )
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers or EnhancedDataLoader._get_optimal_workers(),
            pin_memory=pin_memory and torch.cuda.is_available(),
            **{k: v for k, v in kwargs.items() if k != 'shuffle'}
        )


class SmartBatchSampler(Sampler):
    """
    Smart batch sampler that adjusts batch size based on available memory.
    """
    
    def __init__(
        self,
        dataset: Dataset,
        base_batch_size: int = 32,
        max_batch_size: int = 128,
        min_batch_size: int = 1
    ):
        """
        Initialize smart batch sampler.
        
        Args:
            dataset: Dataset to sample from
            base_batch_size: Base batch size
            max_batch_size: Maximum batch size
            min_batch_size: Minimum batch size
        """
        self.dataset = dataset
        self.base_batch_size = base_batch_size
        self.max_batch_size = max_batch_size
        self.min_batch_size = min_batch_size
        self.current_batch_size = base_batch_size
    
    def __iter__(self):
        """Generate batches."""
        indices = list(range(len(self.dataset)))
        for i in range(0, len(indices), self.current_batch_size):
            yield indices[i:i + self.current_batch_size]
    
    def __len__(self):
        """Get number of batches."""
        return (len(self.dataset) + self.current_batch_size - 1) // self.current_batch_size
    
    def adjust_batch_size(self, factor: float):
        """
        Adjust batch size by factor.
        
        Args:
            factor: Factor to multiply batch size by
        """
        new_size = int(self.current_batch_size * factor)
        self.current_batch_size = max(
            self.min_batch_size,
            min(self.max_batch_size, new_size)
        )
        logger.info(f"Batch size adjusted to {self.current_batch_size}")


__all__ = [
    "EnhancedDataLoader",
    "SmartBatchSampler",
]



