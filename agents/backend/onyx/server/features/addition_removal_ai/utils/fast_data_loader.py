"""
Fast Data Loading Utilities
"""

import torch
from torch.utils.data import DataLoader, Dataset
from typing import Optional, Callable
import logging

logger = logging.getLogger(__name__)


def create_fast_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: Optional[int] = None,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
    persistent_workers: bool = True
) -> DataLoader:
    """
    Create optimized DataLoader for fast training
    
    Args:
        dataset: Dataset
        batch_size: Batch size
        shuffle: Shuffle data
        num_workers: Number of worker processes (auto if None)
        pin_memory: Pin memory for faster GPU transfer
        prefetch_factor: Prefetch factor
        persistent_workers: Keep workers alive between epochs
        
    Returns:
        Optimized DataLoader
    """
    if num_workers is None:
        num_workers = min(4, torch.get_num_threads())
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        drop_last=False
    )


class FastDataset(Dataset):
    """Fast dataset with caching"""
    
    def __init__(self, data, transform: Optional[Callable] = None, cache: bool = True):
        """
        Initialize fast dataset
        
        Args:
            data: Data list or array
            transform: Optional transform function
            cache: Cache transformed data
        """
        self.data = data
        self.transform = transform
        self.cache = cache
        self._cache = {} if cache else None
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if self._cache is not None and idx in self._cache:
            return self._cache[idx]
        
        item = self.data[idx]
        
        if self.transform:
            item = self.transform(item)
        
        if self._cache is not None:
            self._cache[idx] = item
        
        return item


def optimize_dataloader(dataloader: DataLoader) -> DataLoader:
    """
    Optimize existing DataLoader settings
    
    Args:
        dataloader: DataLoader to optimize
        
    Returns:
        Optimized DataLoader
    """
    # Create new DataLoader with optimized settings
    return DataLoader(
        dataloader.dataset,
        batch_size=dataloader.batch_size,
        shuffle=dataloader.sampler is not None,
        num_workers=min(4, torch.get_num_threads()) if dataloader.num_workers == 0 else dataloader.num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        prefetch_factor=2,
        persistent_workers=True if dataloader.num_workers > 0 else False
    )

