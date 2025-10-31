"""
Optimized Data Splitting System for Video-OpusClip

Comprehensive train/validation/test splits and cross-validation utilities
for video datasets with stratification, temporal considerations, and performance optimization.
"""

import torch
from torch.utils.data import Dataset, Subset, DataLoader
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, KFold, 
    GroupKFold, TimeSeriesSplit, ShuffleSplit
)
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import json
import pickle
import structlog
from dataclasses import dataclass
import time
from collections import defaultdict
import random

logger = structlog.get_logger()

# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class SplitConfig:
    """Configuration for data splitting."""
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    random_state: int = 42
    stratify: bool = True
    temporal_split: bool = False
    group_by: Optional[str] = None
    min_samples_per_class: int = 5
    ensure_balance: bool = True

@dataclass
class CrossValidationConfig:
    """Configuration for cross-validation."""
    n_splits: int = 5
    shuffle: bool = True
    random_state: int = 42
    stratify: bool = True
    temporal_cv: bool = False
    group_by: Optional[str] = None

@dataclass
class DatasetSplits:
    """Container for dataset splits."""
    train_indices: List[int]
    val_indices: List[int]
    test_indices: List[int]
    metadata: Dict[str, Any]
    split_config: SplitConfig

@dataclass
class CrossValidationSplits:
    """Container for cross-validation splits."""
    fold_splits: List[Tuple[List[int], List[int]]]  # (train_indices, val_indices)
    metadata: Dict[str, Any]
    cv_config: CrossValidationConfig

# =============================================================================
# OPTIMIZED DATA SPLITTER
# =============================================================================

class OptimizedDataSplitter:
    """High-performance data splitting with stratification and temporal considerations."""
    
    def __init__(
        self,
        dataset: Dataset,
        split_config: Optional[SplitConfig] = None,
        cv_config: Optional[CrossValidationConfig] = None,
        cache_dir: Optional[str] = None,
        enable_caching: bool = True
    ):
        self.dataset = dataset
        self.split_config = split_config or SplitConfig()
        self.cv_config = cv_config or CrossValidationConfig()
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.enable_caching = enable_caching
        
        # Setup cache
        if self.enable_caching and self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract dataset metadata
        self.metadata = self._extract_dataset_metadata()
        
        logger.info(f"Initialized data splitter for {len(dataset)} samples")
    
    def _extract_dataset_metadata(self) -> Dict[str, Any]:
        """Extract metadata from dataset for splitting strategies."""
        metadata = {
            'total_samples': len(self.dataset),
            'labels': [],
            'groups': [],
            'timestamps': [],
            'sample_weights': []
        }
        
        # Extract labels and metadata from first few samples
        sample_indices = min(1000, len(self.dataset))
        for i in range(sample_indices):
            try:
                sample = self.dataset[i]
                
                # Extract labels
                if hasattr(sample, 'labels') and sample.labels:
                    if isinstance(sample.labels, dict):
                        for key, value in sample.labels.items():
                            if key not in metadata['labels']:
                                metadata['labels'].append(key)
                    else:
                        metadata['labels'].append(sample.labels)
                
                # Extract groups
                if hasattr(sample, 'metadata') and sample.metadata:
                    if 'group' in sample.metadata:
                        metadata['groups'].append(sample.metadata['group'])
                    if 'timestamp' in sample.metadata:
                        metadata['timestamps'].append(sample.metadata['timestamp'])
                
                # Extract sample weights
                if hasattr(sample, 'weight'):
                    metadata['sample_weights'].append(sample.weight)
                    
            except Exception as e:
                logger.warning(f"Failed to extract metadata from sample {i}: {e}")
                continue
        
        # Remove duplicates
        metadata['labels'] = list(set(metadata['labels']))
        metadata['groups'] = list(set(metadata['groups']))
        metadata['timestamps'] = list(set(metadata['timestamps']))
        
        return metadata
    
    def _get_stratification_labels(self) -> Optional[np.ndarray]:
        """Get labels for stratification."""
        if not self.split_config.stratify:
            return None
        
        labels = []
        for i in range(len(self.dataset)):
            try:
                sample = self.dataset[i]
                if hasattr(sample, 'labels') and sample.labels:
                    if isinstance(sample.labels, dict):
                        # Use first label for stratification
                        label = list(sample.labels.values())[0]
                    else:
                        label = sample.labels
                    labels.append(str(label))
                else:
                    labels.append('unknown')
            except Exception:
                labels.append('unknown')
        
        return np.array(labels)
    
    def _get_groups(self) -> Optional[np.ndarray]:
        """Get groups for group-based splitting."""
        if not self.split_config.group_by:
            return None
        
        groups = []
        for i in range(len(self.dataset)):
            try:
                sample = self.dataset[i]
                if hasattr(sample, 'metadata') and sample.metadata:
                    group = sample.metadata.get(self.split_config.group_by, 'unknown')
                    groups.append(str(group))
                else:
                    groups.append('unknown')
            except Exception:
                groups.append('unknown')
        
        return np.array(groups)
    
    def _get_timestamps(self) -> Optional[np.ndarray]:
        """Get timestamps for temporal splitting."""
        if not self.split_config.temporal_split:
            return None
        
        timestamps = []
        for i in range(len(self.dataset)):
            try:
                sample = self.dataset[i]
                if hasattr(sample, 'metadata') and sample.metadata:
                    timestamp = sample.metadata.get('timestamp', 0)
                    timestamps.append(timestamp)
                else:
                    timestamps.append(0)
            except Exception:
                timestamps.append(0)
        
        return np.array(timestamps)
    
    def _get_cache_key(self, split_type: str) -> str:
        """Generate cache key for splits."""
        import hashlib
        config_str = f"{self.split_config}_{self.cv_config}_{split_type}"
        return hashlib.md5(config_str.encode()).hexdigest()
    
    def _load_from_cache(self, split_type: str) -> Optional[Union[DatasetSplits, CrossValidationSplits]]:
        """Load splits from cache."""
        if not self.enable_caching or not self.cache_dir:
            return None
        
        cache_key = self._get_cache_key(split_type)
        cache_path = self.cache_dir / f"splits_{cache_key}.pkl"
        
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
                
                # Verify cache validity
                if cached_data.get('dataset_size') == len(self.dataset):
                    logger.debug(f"Loaded splits from cache: {split_type}")
                    return cached_data['splits']
            except Exception as e:
                logger.warning(f"Failed to load splits from cache: {e}")
        
        return None
    
    def _save_to_cache(self, split_type: str, splits: Union[DatasetSplits, CrossValidationSplits]):
        """Save splits to cache."""
        if not self.enable_caching or not self.cache_dir:
            return
        
        cache_key = self._get_cache_key(split_type)
        cache_path = self.cache_dir / f"splits_{cache_key}.pkl"
        
        try:
            cached_data = {
                'dataset_size': len(self.dataset),
                'splits': splits,
                'timestamp': time.time()
            }
            
            with open(cache_path, 'wb') as f:
                pickle.dump(cached_data, f)
            
            logger.debug(f"Saved splits to cache: {split_type}")
        except Exception as e:
            logger.warning(f"Failed to save splits to cache: {e}")
    
    def create_train_val_test_splits(self) -> DatasetSplits:
        """Create train/validation/test splits."""
        # Try cache first
        cached_splits = self._load_from_cache('train_val_test')
        if cached_splits:
            return cached_splits
        
        logger.info("Creating train/validation/test splits...")
        
        # Get stratification labels and groups
        stratify_labels = self._get_stratification_labels()
        groups = self._get_groups()
        timestamps = self._get_timestamps()
        
        # Create indices
        indices = np.arange(len(self.dataset))
        
        if self.split_config.temporal_split and timestamps is not None:
            # Temporal split - sort by timestamp
            sorted_indices = np.argsort(timestamps)
            indices = indices[sorted_indices]
            
            # Calculate split points
            train_end = int(len(indices) * self.split_config.train_ratio)
            val_end = train_end + int(len(indices) * self.split_config.val_ratio)
            
            train_indices = indices[:train_end].tolist()
            val_indices = indices[train_end:val_end].tolist()
            test_indices = indices[val_end:].tolist()
            
        elif groups is not None:
            # Group-based split
            train_indices, temp_indices = train_test_split(
                indices,
                test_size=1 - self.split_config.train_ratio,
                random_state=self.split_config.random_state,
                stratify=groups if self.split_config.stratify else None
            )
            
            val_indices, test_indices = train_test_split(
                temp_indices,
                test_size=self.split_config.test_ratio / (self.split_config.val_ratio + self.split_config.test_ratio),
                random_state=self.split_config.random_state,
                stratify=groups[temp_indices] if self.split_config.stratify else None
            )
            
            train_indices = train_indices.tolist()
            val_indices = val_indices.tolist()
            test_indices = test_indices.tolist()
            
        else:
            # Standard stratified split
            train_indices, temp_indices = train_test_split(
                indices,
                test_size=1 - self.split_config.train_ratio,
                random_state=self.split_config.random_state,
                stratify=stratify_labels if self.split_config.stratify else None
            )
            
            val_indices, test_indices = train_test_split(
                temp_indices,
                test_size=self.split_config.test_ratio / (self.split_config.val_ratio + self.split_config.test_ratio),
                random_state=self.split_config.random_state,
                stratify=stratify_labels[temp_indices] if self.split_config.stratify else None
            )
            
            train_indices = train_indices.tolist()
            val_indices = val_indices.tolist()
            test_indices = test_indices.tolist()
        
        # Ensure minimum samples per class
        if self.split_config.ensure_balance and stratify_labels is not None:
            train_indices, val_indices, test_indices = self._ensure_class_balance(
                train_indices, val_indices, test_indices, stratify_labels
            )
        
        # Create splits
        splits = DatasetSplits(
            train_indices=train_indices,
            val_indices=val_indices,
            test_indices=test_indices,
            metadata={
                'total_samples': len(self.dataset),
                'train_samples': len(train_indices),
                'val_samples': len(val_indices),
                'test_samples': len(test_indices),
                'stratify_labels': stratify_labels is not None,
                'temporal_split': self.split_config.temporal_split,
                'group_by': self.split_config.group_by
            },
            split_config=self.split_config
        )
        
        # Save to cache
        self._save_to_cache('train_val_test', splits)
        
        logger.info(f"Created splits: Train={len(train_indices)}, Val={len(val_indices)}, Test={len(test_indices)}")
        
        return splits
    
    def create_cross_validation_splits(self) -> CrossValidationSplits:
        """Create cross-validation splits."""
        # Try cache first
        cached_splits = self._load_from_cache('cross_validation')
        if cached_splits:
            return cached_splits
        
        logger.info(f"Creating {self.cv_config.n_splits}-fold cross-validation splits...")
        
        # Get stratification labels and groups
        stratify_labels = self._get_stratification_labels()
        groups = self._get_groups()
        timestamps = self._get_timestamps()
        
        # Create indices
        indices = np.arange(len(self.dataset))
        
        if self.cv_config.temporal_cv and timestamps is not None:
            # Temporal cross-validation
            tscv = TimeSeriesSplit(n_splits=self.cv_config.n_splits)
            fold_splits = []
            
            for train_idx, val_idx in tscv.split(indices):
                fold_splits.append((train_idx.tolist(), val_idx.tolist()))
                
        elif groups is not None:
            # Group-based cross-validation
            gkf = GroupKFold(n_splits=self.cv_config.n_splits)
            fold_splits = []
            
            for train_idx, val_idx in gkf.split(indices, groups=groups):
                fold_splits.append((train_idx.tolist(), val_idx.tolist()))
                
        else:
            # Standard stratified cross-validation
            if self.cv_config.stratify and stratify_labels is not None:
                skf = StratifiedKFold(
                    n_splits=self.cv_config.n_splits,
                    shuffle=self.cv_config.shuffle,
                    random_state=self.cv_config.random_state
                )
                fold_splits = []
                
                for train_idx, val_idx in skf.split(indices, stratify_labels):
                    fold_splits.append((train_idx.tolist(), val_idx.tolist()))
            else:
                kf = KFold(
                    n_splits=self.cv_config.n_splits,
                    shuffle=self.cv_config.shuffle,
                    random_state=self.cv_config.random_state
                )
                fold_splits = []
                
                for train_idx, val_idx in kf.split(indices):
                    fold_splits.append((train_idx.tolist(), val_idx.tolist()))
        
        # Create splits
        splits = CrossValidationSplits(
            fold_splits=fold_splits,
            metadata={
                'total_samples': len(self.dataset),
                'n_folds': self.cv_config.n_splits,
                'stratify_labels': stratify_labels is not None,
                'temporal_cv': self.cv_config.temporal_cv,
                'group_by': self.cv_config.group_by
            },
            cv_config=self.cv_config
        )
        
        # Save to cache
        self._save_to_cache('cross_validation', splits)
        
        logger.info(f"Created {len(fold_splits)} cross-validation folds")
        
        return splits
    
    def _ensure_class_balance(
        self,
        train_indices: List[int],
        val_indices: List[int],
        test_indices: List[int],
        labels: np.ndarray
    ) -> Tuple[List[int], List[int], List[int]]:
        """Ensure minimum samples per class in each split."""
        unique_labels, label_counts = np.unique(labels, return_counts=True)
        min_samples = self.split_config.min_samples_per_class
        
        balanced_train = []
        balanced_val = []
        balanced_test = []
        
        for label in unique_labels:
            label_indices = np.where(labels == label)[0]
            
            # Ensure minimum samples in each split
            train_label = [idx for idx in train_indices if idx in label_indices]
            val_label = [idx for idx in val_indices if idx in label_indices]
            test_label = [idx for idx in test_indices if idx in label_indices]
            
            # Redistribute if needed
            if len(train_label) < min_samples:
                needed = min_samples - len(train_label)
                if len(val_label) > min_samples:
                    transfer = val_label[:needed]
                    val_label = val_label[needed:]
                    train_label.extend(transfer)
                elif len(test_label) > min_samples:
                    transfer = test_label[:needed]
                    test_label = test_label[needed:]
                    train_label.extend(transfer)
            
            if len(val_label) < min_samples:
                needed = min_samples - len(val_label)
                if len(test_label) > min_samples:
                    transfer = test_label[:needed]
                    test_label = test_label[needed:]
                    val_label.extend(transfer)
            
            balanced_train.extend(train_label)
            balanced_val.extend(val_label)
            balanced_test.extend(test_label)
        
        return balanced_train, balanced_val, balanced_test

# =============================================================================
# DATASET WRAPPERS
# =============================================================================

class SplitDataset(Dataset):
    """Dataset wrapper for splits."""
    
    def __init__(self, dataset: Dataset, indices: List[int]):
        self.dataset = dataset
        self.indices = indices
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int):
        return self.dataset[self.indices[idx]]

class CrossValidationDataset(Dataset):
    """Dataset wrapper for cross-validation."""
    
    def __init__(self, dataset: Dataset, train_indices: List[int], val_indices: List[int]):
        self.dataset = dataset
        self.train_indices = train_indices
        self.val_indices = val_indices
    
    def get_train_dataset(self) -> Dataset:
        return SplitDataset(self.dataset, self.train_indices)
    
    def get_val_dataset(self) -> Dataset:
        return SplitDataset(self.dataset, self.val_indices)

# =============================================================================
# DATA LOADER FACTORIES
# =============================================================================

class OptimizedDataLoaderFactory:
    """Factory for creating optimized data loaders with splits."""
    
    def __init__(
        self,
        dataset: Dataset,
        split_config: Optional[SplitConfig] = None,
        cv_config: Optional[CrossValidationConfig] = None,
        dataloader_kwargs: Optional[Dict[str, Any]] = None
    ):
        self.dataset = dataset
        self.split_config = split_config or SplitConfig()
        self.cv_config = cv_config or CrossValidationConfig()
        self.dataloader_kwargs = dataloader_kwargs or {}
        
        # Create splitter
        self.splitter = OptimizedDataSplitter(
            dataset=dataset,
            split_config=self.split_config,
            cv_config=self.cv_config
        )
    
    def create_train_val_test_loaders(
        self,
        batch_size: int = 32,
        **kwargs
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create train/validation/test data loaders."""
        
        # Get splits
        splits = self.splitter.create_train_val_test_splits()
        
        # Create datasets
        train_dataset = SplitDataset(self.dataset, splits.train_indices)
        val_dataset = SplitDataset(self.dataset, splits.val_indices)
        test_dataset = SplitDataset(self.dataset, splits.test_indices)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            **{**self.dataloader_kwargs, **kwargs}
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            **{**self.dataloader_kwargs, **kwargs}
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            **{**self.dataloader_kwargs, **kwargs}
        )
        
        logger.info(f"Created data loaders: Train={len(train_loader)}, Val={len(val_loader)}, Test={len(test_loader)}")
        
        return train_loader, val_loader, test_loader
    
    def create_cross_validation_loaders(
        self,
        batch_size: int = 32,
        **kwargs
    ) -> List[Tuple[DataLoader, DataLoader]]:
        """Create cross-validation data loaders."""
        
        # Get splits
        cv_splits = self.splitter.create_cross_validation_splits()
        
        # Create data loaders for each fold
        fold_loaders = []
        
        for fold_idx, (train_indices, val_indices) in enumerate(cv_splits.fold_splits):
            train_dataset = SplitDataset(self.dataset, train_indices)
            val_dataset = SplitDataset(self.dataset, val_indices)
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                **{**self.dataloader_kwargs, **kwargs}
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                **{**self.dataloader_kwargs, **kwargs}
            )
            
            fold_loaders.append((train_loader, val_loader))
            
            logger.info(f"Created fold {fold_idx + 1}: Train={len(train_loader)}, Val={len(val_loader)}")
        
        return fold_loaders

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_balanced_splits(
    dataset: Dataset,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    stratify: bool = True,
    **kwargs
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create balanced train/validation/test splits."""
    
    split_config = SplitConfig(
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        stratify=stratify,
        ensure_balance=True,
        **kwargs
    )
    
    factory = OptimizedDataLoaderFactory(dataset, split_config=split_config)
    return factory.create_train_val_test_loaders()

def create_temporal_splits(
    dataset: Dataset,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    **kwargs
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create temporal train/validation/test splits."""
    
    split_config = SplitConfig(
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        temporal_split=True,
        stratify=False,  # Temporal splits don't use stratification
        **kwargs
    )
    
    factory = OptimizedDataLoaderFactory(dataset, split_config=split_config)
    return factory.create_train_val_test_loaders()

def create_group_splits(
    dataset: Dataset,
    group_by: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    **kwargs
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create group-based train/validation/test splits."""
    
    split_config = SplitConfig(
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        group_by=group_by,
        stratify=True,
        **kwargs
    )
    
    factory = OptimizedDataLoaderFactory(dataset, split_config=split_config)
    return factory.create_train_val_test_loaders()

def create_cross_validation(
    dataset: Dataset,
    n_splits: int = 5,
    stratify: bool = True,
    temporal_cv: bool = False,
    **kwargs
) -> List[Tuple[DataLoader, DataLoader]]:
    """Create cross-validation data loaders."""
    
    cv_config = CrossValidationConfig(
        n_splits=n_splits,
        stratify=stratify,
        temporal_cv=temporal_cv,
        **kwargs
    )
    
    factory = OptimizedDataLoaderFactory(dataset, cv_config=cv_config)
    return factory.create_cross_validation_loaders()

def analyze_splits(
    dataset: Dataset,
    splits: DatasetSplits,
    labels: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """Analyze the quality of data splits."""
    
    analysis = {
        'split_sizes': {
            'train': len(splits.train_indices),
            'val': len(splits.val_indices),
            'test': len(splits.test_indices)
        },
        'split_ratios': {
            'train': len(splits.train_indices) / len(dataset),
            'val': len(splits.val_indices) / len(dataset),
            'test': len(splits.test_indices) / len(dataset)
        }
    }
    
    if labels is not None:
        # Analyze class distribution
        unique_labels, _ = np.unique(labels, return_counts=True)
        class_distribution = {}
        
        for split_name, indices in [('train', splits.train_indices), 
                                   ('val', splits.val_indices), 
                                   ('test', splits.test_indices)]:
            split_labels = labels[indices]
            unique, counts = np.unique(split_labels, return_counts=True)
            class_distribution[split_name] = dict(zip(unique, counts))
        
        analysis['class_distribution'] = class_distribution
        
        # Calculate balance metrics
        balance_metrics = {}
        for split_name, dist in class_distribution.items():
            counts = list(dist.values())
            balance_metrics[split_name] = {
                'min_count': min(counts),
                'max_count': max(counts),
                'std': np.std(counts),
                'cv': np.std(counts) / np.mean(counts) if np.mean(counts) > 0 else 0
            }
        
        analysis['balance_metrics'] = balance_metrics
    
    return analysis

# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def get_data_splitter_factory(
    dataset: Dataset,
    cache_dir: Optional[str] = None,
    **kwargs
) -> OptimizedDataLoaderFactory:
    """Get data splitter factory with default settings."""
    
    return OptimizedDataLoaderFactory(
        dataset=dataset,
        split_config=SplitConfig(**kwargs),
        cv_config=CrossValidationConfig(**kwargs),
        dataloader_kwargs={
            'num_workers': 4,
            'pin_memory': True,
            'persistent_workers': True
        }
    )

# Global factory instance
data_splitter_factory = None

def get_global_data_splitter_factory(dataset: Dataset, **kwargs):
    """Get global data splitter factory."""
    global data_splitter_factory
    if data_splitter_factory is None:
        data_splitter_factory = get_data_splitter_factory(dataset, **kwargs)
    return data_splitter_factory 