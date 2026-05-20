from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import asyncio
import time
import logging
import json
import pickle
from typing import Dict, Any, Optional, List, Union, Tuple, Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
import warnings
from collections import defaultdict
import torch
import torch.nn as nn
from torch.utils.data import Dataset, Subset, DataLoader, random_split
import numpy as np
import pandas as pd
from sklearn.model_selection import (
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import make_scorer, accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from .production_transformers import DeviceManager
from .efficient_data_loader import (
        from sklearn.model_selection import KFold
        from sklearn.model_selection import StratifiedKFold
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.model_selection import GroupKFold
        from sklearn.model_selection import LeaveOneOut
        from sklearn.model_selection import LeavePOut
        from sklearn.model_selection import RepeatedStratifiedKFold
        from .model_training import ModelTrainer
        from .efficient_data_loader import DataLoaderFactory
    from .efficient_data_loader import DataLoaderManager
from typing import Any, List, Dict, Optional
"""
🚀 Data Splitting & Cross-Validation System - Production Ready
==============================================================

Enterprise-grade data splitting and cross-validation system with
proper train/validation/test splits, stratified sampling, time series splits,
and advanced cross-validation strategies for AI training.
"""


    train_test_split, StratifiedShuffleSplit, StratifiedKFold,
    TimeSeriesSplit, GroupShuffleSplit, GroupKFold,
    LeaveOneOut, LeavePOut, RepeatedStratifiedKFold,
    cross_val_score, cross_validate
)

# Import our production engines
    DataLoaderManager, DataLoaderConfig, DataFormat, CacheStrategy,
    OptimizedTextDataset, CachedDataset
)

logger = logging.getLogger(__name__)

class SplitStrategy(Enum):
    """Available split strategies."""
    RANDOM = "random"
    STRATIFIED = "stratified"
    TIME_SERIES = "time_series"
    GROUP = "group"
    CUSTOM = "custom"

class CrossValidationStrategy(Enum):
    """Available cross-validation strategies."""
    K_FOLD = "k_fold"
    STRATIFIED_K_FOLD = "stratified_k_fold"
    TIME_SERIES_CV = "time_series_cv"
    GROUP_K_FOLD = "group_k_fold"
    LEAVE_ONE_OUT = "leave_one_out"
    LEAVE_P_OUT = "leave_p_out"
    REPEATED_STRATIFIED_K_FOLD = "repeated_stratified_k_fold"

@dataclass
class SplitConfig:
    """Configuration for data splitting."""
    strategy: SplitStrategy = SplitStrategy.STRATIFIED
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    random_state: int = 42
    shuffle: bool = True
    
    # Stratified splitting
    stratify_by: Optional[str] = None  # Column name for stratification
    
    # Time series splitting
    time_column: Optional[str] = None  # Column name for time ordering
    
    # Group splitting
    group_column: Optional[str] = None  # Column name for grouping
    
    # Custom splitting
    custom_split_function: Optional[Callable] = None
    
    def __post_init__(self) -> Any:
        """Validate configuration."""
        total_ratio = self.train_ratio + self.val_ratio + self.test_ratio
        if not np.isclose(total_ratio, 1.0, atol=1e-6):
            raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio}")

@dataclass
class CrossValidationConfig:
    """Configuration for cross-validation."""
    strategy: CrossValidationStrategy = CrossValidationStrategy.STRATIFIED_K_FOLD
    n_splits: int = 5
    n_repeats: int = 3  # For repeated strategies
    test_size: float = 0.2
    random_state: int = 42
    shuffle: bool = True
    
    # Stratified CV
    stratify_by: Optional[str] = None
    
    # Time series CV
    time_column: Optional[str] = None
    
    # Group CV
    group_column: Optional[str] = None
    
    # Leave-P-Out
    p: int = 1  # For LeavePOut
    
    # Custom CV
    custom_cv_function: Optional[Callable] = None

@dataclass
class SplitResult:
    """Result of data splitting."""
    train_indices: List[int]
    val_indices: List[int]
    test_indices: List[int]
    train_dataset: Dataset
    val_dataset: Dataset
    test_dataset: Dataset
    split_info: Dict[str, Any]
    
    def get_split_sizes(self) -> Dict[str, int]:
        """Get sizes of each split."""
        return {
            'train': len(self.train_indices),
            'val': len(self.val_indices),
            'test': len(self.test_indices),
            'total': len(self.train_indices) + len(self.val_indices) + len(self.test_indices)
        }
    
    def get_split_ratios(self) -> Dict[str, float]:
        """Get ratios of each split."""
        sizes = self.get_split_sizes()
        total = sizes['total']
        return {
            'train': sizes['train'] / total,
            'val': sizes['val'] / total,
            'test': sizes['test'] / total
        }

@dataclass
class CrossValidationResult:
    """Result of cross-validation."""
    fold_results: List[Dict[str, Any]]
    mean_scores: Dict[str, float]
    std_scores: Dict[str, float]
    best_fold: int
    worst_fold: int
    cv_info: Dict[str, Any]
    
    def get_summary(self) -> Dict[str, Any]:
        """Get CV summary."""
        return {
            'mean_scores': self.mean_scores,
            'std_scores': self.std_scores,
            'best_fold': self.best_fold,
            'worst_fold': self.worst_fold,
            'n_folds': len(self.fold_results)
        }

class DataSplitter:
    """Production-ready data splitter."""
    
    def __init__(self, device_manager: DeviceManager):
        
    """__init__ function."""
self.device_manager = device_manager
        self.logger = logging.getLogger(f"{__name__}.DataSplitter")
    
    def split_dataset(self, dataset: Dataset, config: SplitConfig) -> SplitResult:
        """Split dataset according to configuration."""
        self.logger.info(f"Splitting dataset with strategy: {config.strategy.value}")
        
        total_size = len(dataset)
        indices = list(range(total_size))
        
        if config.strategy == SplitStrategy.RANDOM:
            return self._random_split(dataset, indices, config)
        elif config.strategy == SplitStrategy.STRATIFIED:
            return self._stratified_split(dataset, indices, config)
        elif config.strategy == SplitStrategy.TIME_SERIES:
            return self._time_series_split(dataset, indices, config)
        elif config.strategy == SplitStrategy.GROUP:
            return self._group_split(dataset, indices, config)
        elif config.strategy == SplitStrategy.CUSTOM:
            return self._custom_split(dataset, indices, config)
        else:
            raise ValueError(f"Unknown split strategy: {config.strategy}")
    
    def _random_split(self, dataset: Dataset, indices: List[int], config: SplitConfig) -> SplitResult:
        """Random split."""
        # Calculate split sizes
        train_size = int(config.train_ratio * len(indices))
        val_size = int(config.val_ratio * len(indices))
        test_size = len(indices) - train_size - val_size
        
        # Split indices
        train_indices, temp_indices = train_test_split(
            indices,
            train_size=train_size,
            random_state=config.random_state,
            shuffle=config.shuffle
        )
        
        val_indices, test_indices = train_test_split(
            temp_indices,
            train_size=val_size,
            random_state=config.random_state,
            shuffle=config.shuffle
        )
        
        # Create datasets
        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)
        test_dataset = Subset(dataset, test_indices)
        
        split_info = {
            'strategy': 'random',
            'train_size': len(train_indices),
            'val_size': len(val_indices),
            'test_size': len(test_indices),
            'train_ratio': config.train_ratio,
            'val_ratio': config.val_ratio,
            'test_ratio': config.test_ratio
        }
        
        return SplitResult(
            train_indices=train_indices,
            val_indices=val_indices,
            test_indices=test_indices,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            split_info=split_info
        )
    
    def _stratified_split(self, dataset: Dataset, indices: List[int], config: SplitConfig) -> SplitResult:
        """Stratified split based on labels."""
        # Extract labels for stratification
        labels = self._extract_labels(dataset, indices)
        
        # Calculate split sizes
        train_size = int(config.train_ratio * len(indices))
        val_size = int(config.val_ratio * len(indices))
        
        # Stratified split
        train_indices, temp_indices = train_test_split(
            indices,
            train_size=train_size,
            stratify=labels,
            random_state=config.random_state,
            shuffle=config.shuffle
        )
        
        # Get labels for validation split
        temp_labels = [labels[indices.index(idx)] for idx in temp_indices]
        
        val_indices, test_indices = train_test_split(
            temp_indices,
            train_size=val_size,
            stratify=temp_labels,
            random_state=config.random_state,
            shuffle=config.shuffle
        )
        
        # Create datasets
        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)
        test_dataset = Subset(dataset, test_indices)
        
        # Calculate class distribution
        train_labels = [labels[indices.index(idx)] for idx in train_indices]
        val_labels = [labels[indices.index(idx)] for idx in val_indices]
        test_labels = [labels[indices.index(idx)] for idx in test_indices]
        
        split_info = {
            'strategy': 'stratified',
            'train_size': len(train_indices),
            'val_size': len(val_indices),
            'test_size': len(test_indices),
            'train_ratio': config.train_ratio,
            'val_ratio': config.val_ratio,
            'test_ratio': config.test_ratio,
            'class_distribution': {
                'train': self._get_class_distribution(train_labels),
                'val': self._get_class_distribution(val_labels),
                'test': self._get_class_distribution(test_labels)
            }
        }
        
        return SplitResult(
            train_indices=train_indices,
            val_indices=val_indices,
            test_indices=test_indices,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            split_info=split_info
        )
    
    def _time_series_split(self, dataset: Dataset, indices: List[int], config: SplitConfig) -> SplitResult:
        """Time series split."""
        if config.time_column is None:
            raise ValueError("time_column must be specified for time series split")
        
        # Sort indices by time
        sorted_indices = self._sort_by_time(dataset, indices, config.time_column)
        
        # Calculate split sizes
        train_size = int(config.train_ratio * len(sorted_indices))
        val_size = int(config.val_ratio * len(sorted_indices))
        
        # Split in order
        train_indices = sorted_indices[:train_size]
        val_indices = sorted_indices[train_size:train_size + val_size]
        test_indices = sorted_indices[train_size + val_size:]
        
        # Create datasets
        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)
        test_dataset = Subset(dataset, test_indices)
        
        split_info = {
            'strategy': 'time_series',
            'train_size': len(train_indices),
            'val_size': len(val_indices),
            'test_size': len(test_indices),
            'time_column': config.time_column
        }
        
        return SplitResult(
            train_indices=train_indices,
            val_indices=val_indices,
            test_indices=test_indices,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            split_info=split_info
        )
    
    def _group_split(self, dataset: Dataset, indices: List[int], config: SplitConfig) -> SplitResult:
        """Group-based split."""
        if config.group_column is None:
            raise ValueError("group_column must be specified for group split")
        
        # Extract groups
        groups = self._extract_groups(dataset, indices, config.group_column)
        
        # Calculate split sizes
        train_size = int(config.train_ratio * len(indices))
        val_size = int(config.val_ratio * len(indices))
        
        # Group split
        train_indices, temp_indices = train_test_split(
            indices,
            train_size=train_size,
            stratify=groups,
            random_state=config.random_state,
            shuffle=config.shuffle
        )
        
        temp_groups = [groups[indices.index(idx)] for idx in temp_indices]
        
        val_indices, test_indices = train_test_split(
            temp_indices,
            train_size=val_size,
            stratify=temp_groups,
            random_state=config.random_state,
            shuffle=config.shuffle
        )
        
        # Create datasets
        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)
        test_dataset = Subset(dataset, test_indices)
        
        split_info = {
            'strategy': 'group',
            'train_size': len(train_indices),
            'val_size': len(val_indices),
            'test_size': len(test_indices),
            'group_column': config.group_column
        }
        
        return SplitResult(
            train_indices=train_indices,
            val_indices=val_indices,
            test_indices=test_indices,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            split_info=split_info
        )
    
    def _custom_split(self, dataset: Dataset, indices: List[int], config: SplitConfig) -> SplitResult:
        """Custom split using provided function."""
        if config.custom_split_function is None:
            raise ValueError("custom_split_function must be provided for custom split")
        
        # Use custom function
        train_indices, val_indices, test_indices = config.custom_split_function(
            dataset, indices, config
        )
        
        # Create datasets
        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)
        test_dataset = Subset(dataset, test_indices)
        
        split_info = {
            'strategy': 'custom',
            'train_size': len(train_indices),
            'val_size': len(val_indices),
            'test_size': len(test_indices)
        }
        
        return SplitResult(
            train_indices=train_indices,
            val_indices=val_indices,
            test_indices=test_indices,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            split_info=split_info
        )
    
    def _extract_labels(self, dataset: Dataset, indices: List[int]) -> List[int]:
        """Extract labels from dataset."""
        labels = []
        for idx in indices:
            item = dataset[idx]
            if isinstance(item, dict) and 'labels' in item:
                label = item['labels']
                if isinstance(label, torch.Tensor):
                    label = label.item()
                labels.append(label)
            else:
                # Default label if not found
                labels.append(0)
        return labels
    
    def _extract_groups(self, dataset: Dataset, indices: List[int], group_column: str) -> List[str]:
        """Extract groups from dataset."""
        groups = []
        for idx in indices:
            item = dataset[idx]
            if isinstance(item, dict) and group_column in item:
                groups.append(str(item[group_column]))
            else:
                groups.append('default')
        return groups
    
    def _sort_by_time(self, dataset: Dataset, indices: List[int], time_column: str) -> List[int]:
        """Sort indices by time column."""
        time_values = []
        for idx in indices:
            item = dataset[idx]
            if isinstance(item, dict) and time_column in item:
                time_values.append(item[time_column])
            else:
                time_values.append(0)
        
        # Sort by time values
        sorted_pairs = sorted(zip(indices, time_values), key=lambda x: x[1])
        return [idx for idx, _ in sorted_pairs]
    
    def _get_class_distribution(self, labels: List[int]) -> Dict[int, int]:
        """Get class distribution."""
        distribution = defaultdict(int)
        for label in labels:
            distribution[label] += 1
        return dict(distribution)

class CrossValidator:
    """Production-ready cross-validator."""
    
    def __init__(self, device_manager: DeviceManager):
        
    """__init__ function."""
self.device_manager = device_manager
        self.logger = logging.getLogger(f"{__name__}.CrossValidator")
    
    async def cross_validate(self, dataset: Dataset, config: CrossValidationConfig,
                           model_class, training_config, 
                           data_loader_manager: DataLoaderManager) -> CrossValidationResult:
        """Perform cross-validation."""
        self.logger.info(f"Starting cross-validation with strategy: {config.strategy.value}")
        
        # Create CV splits
        cv_splits = self._create_cv_splits(dataset, config)
        
        # Perform cross-validation
        fold_results = []
        best_score = float('-inf')
        worst_score = float('inf')
        best_fold = 0
        worst_fold = 0
        
        for fold_idx, (train_indices, val_indices) in enumerate(cv_splits):
            self.logger.info(f"Training fold {fold_idx + 1}/{len(cv_splits)}")
            
            # Create fold datasets
            train_dataset = Subset(dataset, train_indices)
            val_dataset = Subset(dataset, val_indices)
            
            # Train and evaluate model
            fold_result = await self._train_and_evaluate_fold(
                train_dataset, val_dataset, model_class, training_config,
                data_loader_manager, fold_idx
            )
            
            fold_results.append(fold_result)
            
            # Track best/worst folds
            score = fold_result['val_f1_score']
            if score > best_score:
                best_score = score
                best_fold = fold_idx
            if score < worst_score:
                worst_score = score
                worst_fold = fold_idx
        
        # Calculate summary statistics
        mean_scores, std_scores = self._calculate_cv_statistics(fold_results)
        
        cv_info = {
            'strategy': config.strategy.value,
            'n_splits': len(cv_splits),
            'n_repeats': config.n_repeats if hasattr(config, 'n_repeats') else 1,
            'best_score': best_score,
            'worst_score': worst_score
        }
        
        return CrossValidationResult(
            fold_results=fold_results,
            mean_scores=mean_scores,
            std_scores=std_scores,
            best_fold=best_fold,
            worst_fold=worst_fold,
            cv_info=cv_info
        )
    
    def _create_cv_splits(self, dataset: Dataset, config: CrossValidationConfig) -> List[Tuple[List[int], List[int]]]:
        """Create cross-validation splits."""
        indices = list(range(len(dataset)))
        
        if config.strategy == CrossValidationStrategy.K_FOLD:
            return self._create_k_fold_splits(indices, config)
        elif config.strategy == CrossValidationStrategy.STRATIFIED_K_FOLD:
            return self._create_stratified_k_fold_splits(dataset, indices, config)
        elif config.strategy == CrossValidationStrategy.TIME_SERIES_CV:
            return self._create_time_series_cv_splits(dataset, indices, config)
        elif config.strategy == CrossValidationStrategy.GROUP_K_FOLD:
            return self._create_group_k_fold_splits(dataset, indices, config)
        elif config.strategy == CrossValidationStrategy.LEAVE_ONE_OUT:
            return self._create_leave_one_out_splits(indices, config)
        elif config.strategy == CrossValidationStrategy.LEAVE_P_OUT:
            return self._create_leave_p_out_splits(indices, config)
        elif config.strategy == CrossValidationStrategy.REPEATED_STRATIFIED_K_FOLD:
            return self._create_repeated_stratified_k_fold_splits(dataset, indices, config)
        else:
            raise ValueError(f"Unknown CV strategy: {config.strategy}")
    
    def _create_k_fold_splits(self, indices: List[int], config: CrossValidationConfig) -> List[Tuple[List[int], List[int]]]:
        """Create K-fold splits."""
        
        kf = KFold(n_splits=config.n_splits, shuffle=config.shuffle, random_state=config.random_state)
        splits = []
        
        for train_idx, val_idx in kf.split(indices):
            train_indices = [indices[i] for i in train_idx]
            val_indices = [indices[i] for i in val_idx]
            splits.append((train_indices, val_indices))
        
        return splits
    
    def _create_stratified_k_fold_splits(self, dataset: Dataset, indices: List[int], config: CrossValidationConfig) -> List[Tuple[List[int], List[int]]]:
        """Create stratified K-fold splits."""
        
        # Extract labels for stratification
        labels = self._extract_labels(dataset, indices)
        
        skf = StratifiedKFold(n_splits=config.n_splits, shuffle=config.shuffle, random_state=config.random_state)
        splits = []
        
        for train_idx, val_idx in skf.split(indices, labels):
            train_indices = [indices[i] for i in train_idx]
            val_indices = [indices[i] for i in val_idx]
            splits.append((train_indices, val_indices))
        
        return splits
    
    def _create_time_series_cv_splits(self, dataset: Dataset, indices: List[int], config: CrossValidationConfig) -> List[Tuple[List[int], List[int]]]:
        """Create time series CV splits."""
        
        if config.time_column is None:
            raise ValueError("time_column must be specified for time series CV")
        
        # Sort indices by time
        sorted_indices = self._sort_by_time(dataset, indices, config.time_column)
        
        tscv = TimeSeriesSplit(n_splits=config.n_splits)
        splits = []
        
        for train_idx, val_idx in tscv.split(sorted_indices):
            train_indices = [sorted_indices[i] for i in train_idx]
            val_indices = [sorted_indices[i] for i in val_idx]
            splits.append((train_indices, val_indices))
        
        return splits
    
    def _create_group_k_fold_splits(self, dataset: Dataset, indices: List[int], config: CrossValidationConfig) -> List[Tuple[List[int], List[int]]]:
        """Create group K-fold splits."""
        
        if config.group_column is None:
            raise ValueError("group_column must be specified for group K-fold CV")
        
        # Extract groups
        groups = self._extract_groups(dataset, indices, config.group_column)
        
        gkf = GroupKFold(n_splits=config.n_splits)
        splits = []
        
        for train_idx, val_idx in gkf.split(indices, groups=groups):
            train_indices = [indices[i] for i in train_idx]
            val_indices = [indices[i] for i in val_idx]
            splits.append((train_indices, val_indices))
        
        return splits
    
    def _create_leave_one_out_splits(self, indices: List[int], config: CrossValidationConfig) -> List[Tuple[List[int], List[int]]]:
        """Create leave-one-out splits."""
        
        loo = LeaveOneOut()
        splits = []
        
        for train_idx, val_idx in loo.split(indices):
            train_indices = [indices[i] for i in train_idx]
            val_indices = [indices[i] for i in val_idx]
            splits.append((train_indices, val_indices))
        
        return splits
    
    def _create_leave_p_out_splits(self, indices: List[int], config: CrossValidationConfig) -> List[Tuple[List[int], List[int]]]:
        """Create leave-P-out splits."""
        
        lpo = LeavePOut(p=config.p)
        splits = []
        
        for train_idx, val_idx in lpo.split(indices):
            train_indices = [indices[i] for i in train_idx]
            val_indices = [indices[i] for i in val_idx]
            splits.append((train_indices, val_indices))
        
        return splits
    
    def _create_repeated_stratified_k_fold_splits(self, dataset: Dataset, indices: List[int], config: CrossValidationConfig) -> List[Tuple[List[int], List[int]]]:
        """Create repeated stratified K-fold splits."""
        
        # Extract labels for stratification
        labels = self._extract_labels(dataset, indices)
        
        rskf = RepeatedStratifiedKFold(
            n_splits=config.n_splits,
            n_repeats=config.n_repeats,
            random_state=config.random_state
        )
        splits = []
        
        for train_idx, val_idx in rskf.split(indices, labels):
            train_indices = [indices[i] for i in train_idx]
            val_indices = [indices[i] for i in val_idx]
            splits.append((train_indices, val_indices))
        
        return splits
    
    async def _train_and_evaluate_fold(self, train_dataset: Dataset, val_dataset: Dataset,
                                     model_class, training_config, 
                                     data_loader_manager: DataLoaderManager,
                                     fold_idx: int) -> Dict[str, Any]:
        """Train and evaluate a single fold."""
        # Create model
        num_classes = self._get_num_classes(train_dataset)
        model, _ = self._create_model(model_class, training_config, num_classes)
        
        # Create DataLoaders
        train_loader, val_loader = data_loader_manager.create_dataloaders(
            train_dataset, val_dataset, val_dataset,  # Use val_dataset as test for CV
            DataLoaderConfig(
                batch_size=training_config.batch_size,
                num_workers=4,
                pin_memory=True,
                persistent_workers=True
            )
        )
        
        # Train model (simplified for CV)
        best_score = 0.0
        for epoch in range(min(3, training_config.num_epochs)):  # Shorter training for CV
            # Train epoch
            train_metrics = await self._train_epoch(model, train_loader)
            
            # Validate
            val_metrics = await self._validate_epoch(model, val_loader)
            
            if val_metrics['f1_score'] > best_score:
                best_score = val_metrics['f1_score']
        
        return {
            'fold': fold_idx + 1,
            'train_size': len(train_dataset),
            'val_size': len(val_dataset),
            'train_accuracy': train_metrics['accuracy'],
            'val_accuracy': val_metrics['accuracy'],
            'train_f1_score': train_metrics['f1_score'],
            'val_f1_score': val_metrics['f1_score'],
            'best_score': best_score
        }
    
    def _calculate_cv_statistics(self, fold_results: List[Dict[str, Any]]) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Calculate CV statistics."""
        metrics = ['val_accuracy', 'val_f1_score', 'train_accuracy', 'train_f1_score']
        
        mean_scores = {}
        std_scores = {}
        
        for metric in metrics:
            values = [result[metric] for result in fold_results]
            mean_scores[metric] = np.mean(values)
            std_scores[metric] = np.std(values)
        
        return mean_scores, std_scores
    
    def _extract_labels(self, dataset: Dataset, indices: List[int]) -> List[int]:
        """Extract labels from dataset."""
        labels = []
        for idx in indices:
            item = dataset[idx]
            if isinstance(item, dict) and 'labels' in item:
                label = item['labels']
                if isinstance(label, torch.Tensor):
                    label = label.item()
                labels.append(label)
            else:
                labels.append(0)
        return labels
    
    def _extract_groups(self, dataset: Dataset, indices: List[int], group_column: str) -> List[str]:
        """Extract groups from dataset."""
        groups = []
        for idx in indices:
            item = dataset[idx]
            if isinstance(item, dict) and group_column in item:
                groups.append(str(item[group_column]))
            else:
                groups.append('default')
        return groups
    
    def _sort_by_time(self, dataset: Dataset, indices: List[int], time_column: str) -> List[int]:
        """Sort indices by time column."""
        time_values = []
        for idx in indices:
            item = dataset[idx]
            if isinstance(item, dict) and time_column in item:
                time_values.append(item[time_column])
            else:
                time_values.append(0)
        
        sorted_pairs = sorted(zip(indices, time_values), key=lambda x: x[1])
        return [idx for idx, _ in sorted_pairs]
    
    def _get_num_classes(self, dataset: Dataset) -> int:
        """Get number of classes from dataset."""
        labels = set()
        for i in range(min(100, len(dataset))):  # Sample first 100 items
            item = dataset[i]
            if isinstance(item, dict) and 'labels' in item:
                label = item['labels']
                if isinstance(label, torch.Tensor):
                    label = label.item()
                labels.add(label)
        return max(len(labels), 2)  # At least 2 classes
    
    def _create_model(self, model_class, training_config, num_classes: int):
        """Create model instance."""
        # This would integrate with your existing model creation logic
        trainer = ModelTrainer(self.device_manager)
        return trainer.create_model(training_config, num_classes)
    
    async def _train_epoch(self, model: nn.Module, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        model.train()
        total_loss = 0
        all_predictions = []
        all_targets = []
        
        for batch in train_loader:
            # Move batch to device
            batch = {k: v.to(self.device_manager.get_best_device()) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            outputs = model(**batch)
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
            
            # Collect metrics
            total_loss += loss.item()
            predictions = torch.argmax(outputs.logits, dim=-1)
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(batch['labels'].cpu().numpy())
        
        # Calculate metrics
        accuracy = np.mean(np.array(all_predictions) == np.array(all_targets))
        f1 = f1_score(all_targets, all_predictions, average='weighted')
        
        return {
            'loss': total_loss / len(train_loader),
            'accuracy': accuracy,
            'f1_score': f1
        }
    
    async def _validate_epoch(self, model: nn.Module, val_loader: DataLoader) -> Dict[str, float]:
        """Validate for one epoch."""
        model.eval()
        total_loss = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(self.device_manager.get_best_device()) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                outputs = model(**batch)
                loss = outputs.loss
                
                total_loss += loss.item()
                predictions = torch.argmax(outputs.logits, dim=-1)
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(batch['labels'].cpu().numpy())
        
        # Calculate metrics
        accuracy = np.mean(np.array(all_predictions) == np.array(all_targets))
        f1 = f1_score(all_targets, all_predictions, average='weighted')
        
        return {
            'loss': total_loss / len(val_loader),
            'accuracy': accuracy,
            'f1_score': f1
        }

class DataSplittingManager:
    """Manager for data splitting and cross-validation operations."""
    
    def __init__(self, device_manager: DeviceManager):
        
    """__init__ function."""
self.device_manager = device_manager
        self.splitter = DataSplitter(device_manager)
        self.cross_validator = CrossValidator(device_manager)
        self.logger = logging.getLogger(f"{__name__}.DataSplittingManager")
    
    async def split_and_validate(self, dataset: Dataset, split_config: SplitConfig,
                               cv_config: Optional[CrossValidationConfig] = None,
                               data_loader_manager: Optional[DataLoaderManager] = None) -> Dict[str, Any]:
        """Split dataset and optionally perform cross-validation."""
        # Split dataset
        split_result = self.splitter.split_dataset(dataset, split_config)
        
        result = {
            'split_result': split_result,
            'split_info': split_result.split_info,
            'split_sizes': split_result.get_split_sizes(),
            'split_ratios': split_result.get_split_ratios()
        }
        
        # Perform cross-validation if requested
        if cv_config is not None and data_loader_manager is not None:
            cv_result = await self.cross_validator.cross_validate(
                split_result.train_dataset, cv_config, None, None, data_loader_manager
            )
            result['cv_result'] = cv_result
            result['cv_summary'] = cv_result.get_summary()
        
        return result
    
    def create_dataloaders_from_split(self, split_result: SplitResult,
                                    data_loader_config: DataLoaderConfig) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create DataLoaders from split result."""
        
        factory = DataLoaderFactory(self.device_manager)
        
        # Create DataLoaders
        train_loader = factory.create_dataloader(split_result.train_dataset, data_loader_config)
        
        # Validation and test should not shuffle
        val_config = DataLoaderConfig(**vars(data_loader_config))
        val_config.shuffle = False
        val_loader = factory.create_dataloader(split_result.val_dataset, val_config)
        
        test_config = DataLoaderConfig(**vars(data_loader_config))
        test_config.shuffle = False
        test_loader = factory.create_dataloader(split_result.test_dataset, test_config)
        
        return train_loader, val_loader, test_loader
    
    def analyze_split_quality(self, split_result: SplitResult) -> Dict[str, Any]:
        """Analyze the quality of the split."""
        # Extract labels for analysis
        train_labels = self._extract_labels_from_subset(split_result.train_dataset)
        val_labels = self._extract_labels_from_subset(split_result.val_dataset)
        test_labels = self._extract_labels_from_subset(split_result.test_dataset)
        
        # Calculate class distributions
        train_dist = self._get_class_distribution(train_labels)
        val_dist = self._get_class_distribution(val_labels)
        test_dist = self._get_class_distribution(test_labels)
        
        # Calculate distribution similarity
        all_classes = set(train_dist.keys()) | set(val_dist.keys()) | set(test_dist.keys())
        
        analysis = {
            'class_distributions': {
                'train': train_dist,
                'val': val_dist,
                'test': test_dist
            },
            'class_coverage': {
                'train': len(train_dist),
                'val': len(val_dist),
                'test': len(test_dist),
                'total_unique': len(all_classes)
            },
            'distribution_similarity': self._calculate_distribution_similarity(
                train_dist, val_dist, test_dist
            )
        }
        
        return analysis
    
    def _extract_labels_from_subset(self, subset: Subset) -> List[int]:
        """Extract labels from a subset."""
        labels = []
        for i in range(len(subset)):
            item = subset[i]
            if isinstance(item, dict) and 'labels' in item:
                label = item['labels']
                if isinstance(label, torch.Tensor):
                    label = label.item()
                labels.append(label)
            else:
                labels.append(0)
        return labels
    
    def _get_class_distribution(self, labels: List[int]) -> Dict[int, int]:
        """Get class distribution."""
        distribution = defaultdict(int)
        for label in labels:
            distribution[label] += 1
        return dict(distribution)
    
    def _calculate_distribution_similarity(self, train_dist: Dict[int, int], 
                                        val_dist: Dict[int, int], 
                                        test_dist: Dict[int, int]) -> float:
        """Calculate similarity between distributions."""
        all_classes = set(train_dist.keys()) | set(val_dist.keys()) | set(test_dist.keys())
        
        similarities = []
        for cls in all_classes:
            train_ratio = train_dist.get(cls, 0) / sum(train_dist.values())
            val_ratio = val_dist.get(cls, 0) / sum(val_dist.values())
            test_ratio = test_dist.get(cls, 0) / sum(test_dist.values())
            
            # Calculate similarity as 1 - standard deviation
            ratios = [train_ratio, val_ratio, test_ratio]
            std = np.std(ratios)
            similarity = 1 - std
            similarities.append(similarity)
        
        return np.mean(similarities) if similarities else 0.0

# Factory functions
async def create_data_splitting_manager(device_manager: DeviceManager) -> DataSplittingManager:
    """Create a data splitting manager instance."""
    return DataSplittingManager(device_manager)

# Quick usage functions
async def quick_split_dataset(
    dataset: Dataset,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    strategy: SplitStrategy = SplitStrategy.STRATIFIED
) -> SplitResult:
    """Quick dataset splitting."""
    device_manager = DeviceManager()
    manager = await create_data_splitting_manager(device_manager)
    
    config = SplitConfig(
        strategy=strategy,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_state=42
    )
    
    return manager.splitter.split_dataset(dataset, config)

async def quick_cross_validate(
    dataset: Dataset,
    n_splits: int = 5,
    strategy: CrossValidationStrategy = CrossValidationStrategy.STRATIFIED_K_FOLD
) -> CrossValidationResult:
    """Quick cross-validation."""
    device_manager = DeviceManager()
    manager = await create_data_splitting_manager(device_manager)
    
    config = CrossValidationConfig(
        strategy=strategy,
        n_splits=n_splits,
        random_state=42
    )
    
    # Create mock data loader manager for CV
    data_loader_manager = await DataLoaderManager(device_manager).__aenter__()
    
    return await manager.cross_validator.cross_validate(
        dataset, config, None, None, data_loader_manager
    )

# Example usage
if __name__ == "__main__":
    async def demo():
        
    """demo function."""
# Create sample dataset
        texts = [f"Sample text {i}" for i in range(1000)]
        labels = [i % 3 for i in range(1000)]  # 3 classes
        
        dataset = OptimizedTextDataset(texts, labels)
        
        # Quick split
        split_result = await quick_split_dataset(
            dataset,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            strategy=SplitStrategy.STRATIFIED
        )
        
        print(f"Split sizes: {split_result.get_split_sizes()}")
        print(f"Split ratios: {split_result.get_split_ratios()}")
        
        # Quick cross-validation
        cv_result = await quick_cross_validate(
            split_result.train_dataset,
            n_splits=5,
            strategy=CrossValidationStrategy.STRATIFIED_K_FOLD
        )
        
        print(f"CV summary: {cv_result.get_summary()}")
    
    asyncio.run(demo()) 