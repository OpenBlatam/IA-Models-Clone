"""
Data Factory
============

Factory for creating datasets and dataloaders.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from torch.utils.data import DataLoader

from ..data import EventDataset, RoutineDataset, create_dataloaders, FeatureExtractor
from ..base import BaseDataset

logger = logging.getLogger(__name__)


class DataFactory:
    """
    Factory for creating datasets and dataloaders.
    
    Supports:
    - Event datasets
    - Routine datasets
    - Custom dataloaders
    - Data preprocessing
    """
    
    def __init__(self):
        """Initialize factory."""
        self._logger = logger
        self.feature_extractor = FeatureExtractor()
    
    def create_dataset(
        self,
        dataset_type: str,
        data: List[Dict[str, Any]],
        config: Optional[Dict[str, Any]] = None
    ) -> BaseDataset:
        """
        Create dataset.
        
        Args:
            dataset_type: Type of dataset (event, routine)
            data: Data list
            config: Dataset configuration
        
        Returns:
            Dataset instance
        """
        if dataset_type == "event":
            return EventDataset(data, self.feature_extractor)
        elif dataset_type == "routine":
            return RoutineDataset(data, self.feature_extractor)
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    def create_dataloaders(
        self,
        dataset: BaseDataset,
        config: Optional[Dict[str, Any]] = None
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create train/val/test dataloaders.
        
        Args:
            dataset: Dataset
            config: Dataloader configuration
        
        Returns:
            (train_loader, val_loader, test_loader)
        """
        if config is None:
            config = self._get_default_config()
        
        return create_dataloaders(
            dataset,
            batch_size=config.get("batch_size", 32),
            train_ratio=config.get("train_ratio", 0.8),
            val_ratio=config.get("val_ratio", 0.1),
            test_ratio=config.get("test_ratio", 0.1),
            num_workers=config.get("num_workers", 0),
            pin_memory=config.get("pin_memory", False)
        )
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default dataloader configuration."""
        return {
            "batch_size": 32,
            "train_ratio": 0.8,
            "val_ratio": 0.1,
            "test_ratio": 0.1,
            "num_workers": 0,
            "pin_memory": False
        }




