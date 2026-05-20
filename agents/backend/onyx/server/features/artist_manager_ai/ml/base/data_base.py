"""
Base Data Interface
===================

Abstract base classes for datasets and dataloaders.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional
import torch
from torch.utils.data import Dataset, DataLoader

class BaseDataset(Dataset, ABC):
    """
    Abstract base class for datasets.
    
    All datasets should implement:
    - __len__(): Dataset size
    - __getitem__(): Get item by index
    """
    
    @abstractmethod
    def __len__(self) -> int:
        """Return dataset size."""
        pass
    
    @abstractmethod
    def __getitem__(self, idx: int) -> tuple:
        """
        Get item by index.
        
        Args:
            idx: Index
        
        Returns:
            (features, target) tuple
        """
        pass


class BaseDataLoader:
    """
    Base class for dataloader creation.
    
    Provides factory methods for creating dataloaders.
    """
    
    @staticmethod
    def create_dataloader(
        dataset: Dataset,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
        pin_memory: bool = False
    ) -> DataLoader:
        """
        Create DataLoader.
        
        Args:
            dataset: PyTorch dataset
            batch_size: Batch size
            shuffle: Whether to shuffle
            num_workers: Number of workers
            pin_memory: Whether to pin memory
        
        Returns:
            DataLoader
        """
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory
        )




