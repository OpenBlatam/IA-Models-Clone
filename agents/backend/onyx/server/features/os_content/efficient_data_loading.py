from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torch.utils.data.distributed import DistributedSampler
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass
import logging
import os
import pickle
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time
        from sklearn.model_selection import train_test_split
from typing import Any, List, Dict, Optional
"""
Efficient Data Loading and Cross-Validation System
==================================================

This module provides efficient data loading using PyTorch's DataLoader with:
- Proper train/validation/test splits
- Cross-validation support
- Memory-efficient data processing
- Multi-worker data loading
- Caching and prefetching
"""


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DataConfig:
    """Configuration for data loading and processing."""
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    shuffle: bool = True
    drop_last: bool = False
    prefetch_factor: int = 2
    persistent_workers: bool = True
    timeout: int = 60
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    random_seed: int = 42
    cache_data: bool = True
    cache_dir: str = "./data_cache"
    normalize: bool = True
    augment: bool = False
    max_samples: Optional[int] = None

class EfficientDataset(Dataset):
    """Base dataset class with efficient data loading capabilities."""
    
    def __init__(self, data: Union[np.ndarray, pd.DataFrame, List], 
                 targets: Optional[Union[np.ndarray, List]] = None,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 cache_data: bool = True):
        """
        Initialize dataset with efficient caching and processing.
        
        Args:
            data: Input data (numpy array, pandas DataFrame, or list)
            targets: Target labels (optional)
            transform: Data transformation function
            target_transform: Target transformation function
            cache_data: Whether to cache processed data
        """
        self.data = self._process_data(data)
        self.targets = targets
        self.transform = transform
        self.target_transform = target_transform
        self.cache_data = cache_data
        
        if cache_data:
            self._cache_processed_data()
    
    def _process_data(self, data: Union[np.ndarray, pd.DataFrame, List]) -> torch.Tensor:
        """Process and convert data to torch tensor."""
        if isinstance(data, pd.DataFrame):
            data = data.values
        elif isinstance(data, list):
            data = np.array(data)
        
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float32)
        
        return data
    
    def _cache_processed_data(self) -> Any:
        """Cache processed data for faster loading."""
        if hasattr(self, '_cached_data'):
            return
        
        self._cached_data = {}
        for idx in range(len(self.data)):
            if idx % 1000 == 0:  # Progress indicator
                logger.info(f"Caching data: {idx}/{len(self.data)}")
            
            sample = self.data[idx]
            if self.transform:
                sample = self.transform(sample)
            self._cached_data[idx] = sample
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.cache_data and idx in self._cached_data:
            sample = self._cached_data[idx]
        else:
            sample = self.data[idx]
            if self.transform:
                sample = self.transform(sample)
        
        if self.targets is not None:
            target = self.targets[idx]
            if self.target_transform:
                target = self.target_transform(target)
            return sample, target
        
        return sample, None

class DataSplitter:
    """Handles data splitting for train/validation/test sets."""
    
    def __init__(self, train_split: float = 0.7, val_split: float = 0.15, 
                 test_split: float = 0.15, random_seed: int = 42):
        """
        Initialize data splitter.
        
        Args:
            train_split: Proportion of data for training
            val_split: Proportion of data for validation
            test_split: Proportion of data for testing
            random_seed: Random seed for reproducibility
        """
        assert abs(train_split + val_split + test_split - 1.0) < 1e-6, "Splits must sum to 1.0"
        
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.random_seed = random_seed
        
        # Set random seed
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
    
    def split_dataset(self, dataset: Dataset) -> Tuple[Dataset, Dataset, Dataset]:
        """
        Split dataset into train/validation/test sets.
        
        Args:
            dataset: Input dataset
            
        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        total_size = len(dataset)
        train_size = int(self.train_split * total_size)
        val_size = int(self.val_split * total_size)
        test_size = total_size - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(self.random_seed)
        )
        
        logger.info(f"Dataset split: Train={len(train_dataset)}, "
                   f"Val={len(val_dataset)}, Test={len(test_dataset)}")
        
        return train_dataset, val_dataset, test_dataset
    
    def stratified_split(self, dataset: Dataset, targets: List) -> Tuple[Dataset, Dataset, Dataset]:
        """
        Perform stratified split based on target distribution.
        
        Args:
            dataset: Input dataset
            targets: Target labels for stratification
            
        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        
        indices = list(range(len(dataset)))
        
        # First split: train vs (val + test)
        train_indices, temp_indices = train_test_split(
            indices, train_size=self.train_split, 
            stratify=targets, random_state=self.random_seed
        )
        
        # Second split: val vs test
        val_test_ratio = self.val_split / (self.val_split + self.test_split)
        val_indices, test_indices = train_test_split(
            temp_indices, train_size=val_test_ratio,
            stratify=[targets[i] for i in temp_indices], 
            random_state=self.random_seed
        )
        
        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)
        test_dataset = Subset(dataset, test_indices)
        
        logger.info(f"Stratified split: Train={len(train_dataset)}, "
                   f"Val={len(val_dataset)}, Test={len(test_dataset)}")
        
        return train_dataset, val_dataset, test_dataset

class CrossValidator:
    """Handles cross-validation for model evaluation."""
    
    def __init__(self, n_splits: int = 5, shuffle: bool = True, random_seed: int = 42):
        """
        Initialize cross-validator.
        
        Args:
            n_splits: Number of folds for cross-validation
            shuffle: Whether to shuffle data before splitting
            random_seed: Random seed for reproducibility
        """
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_seed = random_seed
    
    def k_fold_split(self, dataset: Dataset) -> List[Tuple[Dataset, Dataset]]:
        """
        Perform k-fold cross-validation split.
        
        Args:
            dataset: Input dataset
            
        Returns:
            List of (train_fold, val_fold) tuples
        """
        kfold = KFold(n_splits=self.n_splits, shuffle=self.shuffle, 
                     random_state=self.random_seed)
        
        indices = list(range(len(dataset)))
        folds = []
        
        for train_idx, val_idx in kfold.split(indices):
            train_fold = Subset(dataset, train_idx)
            val_fold = Subset(dataset, val_idx)
            folds.append((train_fold, val_fold))
        
        logger.info(f"Created {self.n_splits}-fold cross-validation splits")
        return folds
    
    def stratified_k_fold_split(self, dataset: Dataset, targets: List) -> List[Tuple[Dataset, Dataset]]:
        """
        Perform stratified k-fold cross-validation split.
        
        Args:
            dataset: Input dataset
            targets: Target labels for stratification
            
        Returns:
            List of (train_fold, val_fold) tuples
        """
        skfold = StratifiedKFold(n_splits=self.n_splits, shuffle=self.shuffle, 
                                random_state=self.random_seed)
        
        indices = list(range(len(dataset)))
        folds = []
        
        for train_idx, val_idx in skfold.split(indices, targets):
            train_fold = Subset(dataset, train_idx)
            val_fold = Subset(dataset, val_idx)
            folds.append((train_fold, val_fold))
        
        logger.info(f"Created stratified {self.n_splits}-fold cross-validation splits")
        return folds

class DataLoaderFactory:
    """Factory for creating efficient DataLoaders."""
    
    def __init__(self, config: DataConfig):
        """
        Initialize DataLoader factory.
        
        Args:
            config: Data configuration
        """
        self.config = config
    
    def create_dataloader(self, dataset: Dataset, is_train: bool = True, 
                         distributed: bool = False) -> DataLoader:
        """
        Create efficient DataLoader.
        
        Args:
            dataset: Input dataset
            is_train: Whether this is for training
            distributed: Whether to use distributed sampling
            
        Returns:
            Configured DataLoader
        """
        # Determine sampler
        if distributed:
            sampler = DistributedSampler(
                dataset, shuffle=is_train and self.config.shuffle
            )
            shuffle = False  # DistributedSampler handles shuffling
        else:
            sampler = None
            shuffle = is_train and self.config.shuffle
        
        # Create DataLoader
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            drop_last=self.config.drop_last,
            prefetch_factor=self.config.prefetch_factor,
            persistent_workers=self.config.persistent_workers,
            timeout=self.config.timeout
        )
        
        return dataloader
    
    def create_dataloaders(self, train_dataset: Dataset, val_dataset: Dataset, 
                          test_dataset: Optional[Dataset] = None,
                          distributed: bool = False) -> Dict[str, DataLoader]:
        """
        Create DataLoaders for all splits.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            test_dataset: Test dataset (optional)
            distributed: Whether to use distributed sampling
            
        Returns:
            Dictionary of DataLoaders
        """
        dataloaders = {
            'train': self.create_dataloader(train_dataset, is_train=True, distributed=distributed),
            'val': self.create_dataloader(val_dataset, is_train=False, distributed=distributed)
        }
        
        if test_dataset is not None:
            dataloaders['test'] = self.create_dataloader(test_dataset, is_train=False, distributed=distributed)
        
        return dataloaders

class DataPreprocessor:
    """Handles data preprocessing and normalization."""
    
    def __init__(self, normalize: bool = True, scaler_type: str = 'standard'):
        """
        Initialize data preprocessor.
        
        Args:
            normalize: Whether to normalize data
            scaler_type: Type of scaler ('standard' or 'minmax')
        """
        self.normalize = normalize
        self.scaler_type = scaler_type
        self.scaler = None
        self.is_fitted = False
    
    def fit(self, data: Union[np.ndarray, torch.Tensor]):
        """Fit the scaler to the data."""
        if not self.normalize:
            return
        
        if isinstance(data, torch.Tensor):
            data = data.numpy()
        
        if self.scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif self.scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaler type: {self.scaler_type}")
        
        self.scaler.fit(data)
        self.is_fitted = True
        logger.info(f"Fitted {self.scaler_type} scaler")
    
    def transform(self, data: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Transform data using fitted scaler."""
        if not self.normalize or not self.is_fitted:
            if isinstance(data, np.ndarray):
                return torch.tensor(data, dtype=torch.float32)
            return data
        
        if isinstance(data, torch.Tensor):
            data_np = data.numpy()
        else:
            data_np = data
        
        transformed_data = self.scaler.transform(data_np)
        return torch.tensor(transformed_data, dtype=torch.float32)
    
    def save_scaler(self, filepath: str):
        """Save fitted scaler to file."""
        if self.scaler is not None:
            with open(filepath, 'wb') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                pickle.dump(self.scaler, f)
            logger.info(f"Saved scaler to {filepath}")
    
    def load_scaler(self, filepath: str):
        """Load scaler from file."""
        with open(filepath, 'rb') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            self.scaler = pickle.load(f)
        self.is_fitted = True
        logger.info(f"Loaded scaler from {filepath}")

class DataManager:
    """Comprehensive data management system."""
    
    def __init__(self, config: DataConfig):
        """
        Initialize data manager.
        
        Args:
            config: Data configuration
        """
        self.config = config
        self.splitter = DataSplitter(
            train_split=config.train_split,
            val_split=config.val_split,
            test_split=config.test_split,
            random_seed=config.random_seed
        )
        self.cross_validator = CrossValidator(
            n_splits=5, shuffle=True, random_seed=config.random_seed
        )
        self.loader_factory = DataLoaderFactory(config)
        self.preprocessor = DataPreprocessor(
            normalize=config.normalize
        )
        
        # Create cache directory
        if config.cache_data:
            os.makedirs(config.cache_dir, exist_ok=True)
    
    def prepare_dataset(self, data: Union[np.ndarray, pd.DataFrame, List],
                       targets: Optional[Union[np.ndarray, List]] = None,
                       transform: Optional[Callable] = None,
                       target_transform: Optional[Callable] = None) -> EfficientDataset:
        """
        Prepare dataset with preprocessing and caching.
        
        Args:
            data: Input data
            targets: Target labels
            transform: Data transformation
            target_transform: Target transformation
            
        Returns:
            Prepared dataset
        """
        # Limit samples if specified
        if self.config.max_samples and len(data) > self.config.max_samples:
            indices = np.random.choice(len(data), self.config.max_samples, replace=False)
            if isinstance(data, pd.DataFrame):
                data = data.iloc[indices]
            else:
                data = data[indices]
            
            if targets is not None:
                targets = targets[indices]
        
        # Create dataset
        dataset = EfficientDataset(
            data=data,
            targets=targets,
            transform=transform,
            target_transform=target_transform,
            cache_data=self.config.cache_data
        )
        
        # Fit preprocessor if targets are available
        if targets is not None:
            self.preprocessor.fit(data)
        
        return dataset
    
    def create_splits(self, dataset: EfficientDataset, 
                     targets: Optional[List] = None,
                     stratified: bool = False) -> Tuple[Dataset, Dataset, Dataset]:
        """
        Create train/validation/test splits.
        
        Args:
            dataset: Input dataset
            targets: Target labels for stratification
            stratified: Whether to use stratified splitting
            
        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        if stratified and targets is not None:
            return self.splitter.stratified_split(dataset, targets)
        else:
            return self.splitter.split_dataset(dataset)
    
    def create_dataloaders(self, train_dataset: Dataset, val_dataset: Dataset,
                          test_dataset: Optional[Dataset] = None,
                          distributed: bool = False) -> Dict[str, DataLoader]:
        """
        Create DataLoaders for all splits.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            test_dataset: Test dataset
            distributed: Whether to use distributed training
            
        Returns:
            Dictionary of DataLoaders
        """
        return self.loader_factory.create_dataloaders(
            train_dataset, val_dataset, test_dataset, distributed
        )
    
    def create_cross_validation_folds(self, dataset: EfficientDataset,
                                    targets: Optional[List] = None,
                                    stratified: bool = False) -> List[Tuple[DataLoader, DataLoader]]:
        """
        Create cross-validation folds with DataLoaders.
        
        Args:
            dataset: Input dataset
            targets: Target labels for stratification
            stratified: Whether to use stratified splitting
            
        Returns:
            List of (train_loader, val_loader) tuples
        """
        if stratified and targets is not None:
            folds = self.cross_validator.stratified_k_fold_split(dataset, targets)
        else:
            folds = self.cross_validator.k_fold_split(dataset)
        
        dataloaders = []
        for train_fold, val_fold in folds:
            train_loader = self.loader_factory.create_dataloader(train_fold, is_train=True)
            val_loader = self.loader_factory.create_dataloader(val_fold, is_train=False)
            dataloaders.append((train_loader, val_loader))
        
        return dataloaders
    
    def save_preprocessor(self, filepath: str):
        """Save preprocessor to file."""
        self.preprocessor.save_scaler(filepath)
    
    def load_preprocessor(self, filepath: str):
        """Load preprocessor from file."""
        self.preprocessor.load_scaler(filepath)

# Utility functions for data loading
def create_efficient_dataloader(dataset: Dataset, config: DataConfig,
                               is_train: bool = True) -> DataLoader:
    """Create efficient DataLoader with given configuration."""
    factory = DataLoaderFactory(config)
    return factory.create_dataloader(dataset, is_train=is_train)

def split_dataset_efficiently(dataset: Dataset, config: DataConfig,
                            targets: Optional[List] = None,
                            stratified: bool = False) -> Tuple[Dataset, Dataset, Dataset]:
    """Split dataset efficiently with given configuration."""
    splitter = DataSplitter(
        train_split=config.train_split,
        val_split=config.val_split,
        test_split=config.test_split,
        random_seed=config.random_seed
    )
    
    if stratified and targets is not None:
        return splitter.stratified_split(dataset, targets)
    else:
        return splitter.split_dataset(dataset)

def create_cross_validation_dataloaders(dataset: Dataset, config: DataConfig,
                                      n_splits: int = 5,
                                      targets: Optional[List] = None,
                                      stratified: bool = False) -> List[Tuple[DataLoader, DataLoader]]:
    """Create cross-validation DataLoaders."""
    cross_validator = CrossValidator(n_splits=n_splits, random_seed=config.random_seed)
    factory = DataLoaderFactory(config)
    
    if stratified and targets is not None:
        folds = cross_validator.stratified_k_fold_split(dataset, targets)
    else:
        folds = cross_validator.k_fold_split(dataset)
    
    dataloaders = []
    for train_fold, val_fold in folds:
        train_loader = factory.create_dataloader(train_fold, is_train=True)
        val_loader = factory.create_dataloader(val_fold, is_train=False)
        dataloaders.append((train_loader, val_loader))
    
    return dataloaders

# Example usage and testing
if __name__ == "__main__":
    # Example configuration
    config = DataConfig(
        batch_size=64,
        num_workers=4,
        train_split=0.7,
        val_split=0.15,
        test_split=0.15,
        cache_data=True
    )
    
    # Create sample data
    sample_data = np.random.randn(1000, 10)
    sample_targets = np.random.randint(0, 3, 1000)
    
    # Initialize data manager
    data_manager = DataManager(config)
    
    # Prepare dataset
    dataset = data_manager.prepare_dataset(sample_data, sample_targets)
    
    # Create splits
    train_dataset, val_dataset, test_dataset = data_manager.create_splits(
        dataset, sample_targets, stratified=True
    )
    
    # Create DataLoaders
    dataloaders = data_manager.create_dataloaders(
        train_dataset, val_dataset, test_dataset
    )
    
    # Test DataLoaders
    for split_name, dataloader in dataloaders.items():
        print(f"{split_name} dataset size: {len(dataloader.dataset)}")
        print(f"{split_name} batches: {len(dataloader)}")
        
        # Test batch loading
        for batch_idx, (data, targets) in enumerate(dataloader):
            print(f"{split_name} batch {batch_idx}: data shape {data.shape}, targets shape {targets.shape}")
            break
    
    # Test cross-validation
    cv_dataloaders = data_manager.create_cross_validation_folds(
        dataset, sample_targets, stratified=True
    )
    
    print(f"Cross-validation folds: {len(cv_dataloaders)}")
    for fold_idx, (train_loader, val_loader) in enumerate(cv_dataloaders):
        print(f"Fold {fold_idx}: train batches {len(train_loader)}, val batches {len(val_loader)}") 