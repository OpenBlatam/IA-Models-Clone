from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional, Union, Callable, Iterator
from dataclasses import dataclass, field
import logging
from pathlib import Path
import time
import json
import pickle
from enum import Enum
import os
import random
from sklearn.model_selection import (
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
from typing import Any, List, Dict, Optional
import asyncio
#!/usr/bin/env python3
"""
Data Splitting and Validation System
Comprehensive train/validation/test splits and cross-validation implementation.
"""

    train_test_split, KFold, StratifiedKFold, ShuffleSplit, 
    StratifiedShuffleSplit, LeaveOneOut, LeavePOut, GroupKFold,
    StratifiedGroupKFold, TimeSeriesSplit, RepeatedKFold,
    RepeatedStratifiedKFold, cross_val_score, cross_validate
)
warnings.filterwarnings('ignore')


class SplitType(Enum):
    """Types of data splits."""
    TRAIN_VAL_TEST = "train_val_test"
    TRAIN_TEST = "train_test"
    CROSS_VALIDATION = "cross_validation"
    TIME_SERIES = "time_series"
    GROUP_SPLIT = "group_split"


class CrossValidationType(Enum):
    """Types of cross-validation."""
    K_FOLD = "k_fold"
    STRATIFIED_K_FOLD = "stratified_k_fold"
    SHUFFLE_SPLIT = "shuffle_split"
    STRATIFIED_SHUFFLE_SPLIT = "stratified_shuffle_split"
    LEAVE_ONE_OUT = "leave_one_out"
    LEAVE_P_OUT = "leave_p_out"
    GROUP_K_FOLD = "group_k_fold"
    STRATIFIED_GROUP_K_FOLD = "stratified_group_k_fold"
    TIME_SERIES_SPLIT = "time_series_split"
    REPEATED_K_FOLD = "repeated_k_fold"
    REPEATED_STRATIFIED_K_FOLD = "repeated_stratified_k_fold"


@dataclass
class SplitConfig:
    """Configuration for data splitting."""
    # Split ratios
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    # Cross-validation settings
    cv_folds: int = 5
    cv_repeats: int = 3
    cv_strategy: CrossValidationType = CrossValidationType.STRATIFIED_K_FOLD
    
    # Stratification
    stratify: bool = True
    stratify_column: Optional[str] = None
    
    # Group splitting
    group_column: Optional[str] = None
    
    # Time series
    time_column: Optional[str] = None
    
    # Random state
    random_state: int = 42
    
    # Validation
    validate_splits: bool = True
    min_samples_per_split: int = 10
    
    # Output
    save_splits: bool = True
    splits_file: str = "data_splits.json"


class DataSplitter:
    """Comprehensive data splitting and validation system."""
    
    def __init__(self, config: SplitConfig):
        
    """__init__ function."""
self.config = config
        self.logger = self._setup_logging()
        
        # Set random seeds
        random.seed(config.random_state)
        np.random.seed(config.random_state)
        torch.manual_seed(config.random_state)
        
        # Store splits
        self.splits = {}
        self.cv_splits = {}
        self.split_metadata = {}
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def split_data(self, data: Union[Dataset, np.ndarray, pd.DataFrame], 
                   targets: Optional[Union[np.ndarray, pd.Series]] = None,
                   split_type: SplitType = SplitType.TRAIN_VAL_TEST) -> Dict[str, Any]:
        """Split data into train/validation/test sets."""
        self.logger.info(f"Splitting data using {split_type.value}")
        
        if split_type == SplitType.TRAIN_VAL_TEST:
            return self._train_val_test_split(data, targets)
        elif split_type == SplitType.TRAIN_TEST:
            return self._train_test_split(data, targets)
        elif split_type == SplitType.CROSS_VALIDATION:
            return self._cross_validation_split(data, targets)
        elif split_type == SplitType.TIME_SERIES:
            return self._time_series_split(data, targets)
        elif split_type == SplitType.GROUP_SPLIT:
            return self._group_split(data, targets)
        else:
            raise ValueError(f"Unknown split type: {split_type}")
    
    def _train_val_test_split(self, data: Union[Dataset, np.ndarray, pd.DataFrame],
                              targets: Optional[Union[np.ndarray, pd.Series]] = None) -> Dict[str, Any]:
        """Perform train/validation/test split."""
        # Convert data to indices if it's a Dataset
        if isinstance(data, Dataset):
            indices = list(range(len(data)))
            data_array = indices
        else:
            data_array = data
        
        # Calculate split sizes
        total_size = len(data_array)
        train_size = int(total_size * self.config.train_ratio)
        val_size = int(total_size * self.config.val_ratio)
        test_size = total_size - train_size - val_size
        
        self.logger.info(f"Total samples: {total_size}")
        self.logger.info(f"Train samples: {train_size} ({train_size/total_size:.1%})")
        self.logger.info(f"Validation samples: {val_size} ({val_size/total_size:.1%})")
        self.logger.info(f"Test samples: {test_size} ({test_size/total_size:.1%})")
        
        # Prepare stratification target
        stratify_target = None
        if self.config.stratify and targets is not None:
            stratify_target = targets
        elif self.config.stratify and self.config.stratify_column and isinstance(data, pd.DataFrame):
            stratify_target = data[self.config.stratify_column]
        
        # Perform splits
        if stratify_target is not None:
            # Stratified split
            train_indices, temp_indices = train_test_split(
                range(total_size),
                train_size=train_size,
                stratify=stratify_target,
                random_state=self.config.random_state
            )
            
            # Split remaining data into validation and test
            remaining_targets = [stratify_target[i] for i in temp_indices]
            val_indices, test_indices = train_test_split(
                temp_indices,
                train_size=val_size,
                stratify=remaining_targets,
                random_state=self.config.random_state
            )
        else:
            # Random split
            train_indices, temp_indices = train_test_split(
                range(total_size),
                train_size=train_size,
                random_state=self.config.random_state
            )
            
            val_indices, test_indices = train_test_split(
                temp_indices,
                train_size=val_size,
                random_state=self.config.random_state
            )
        
        # Create splits
        splits = {
            'train': train_indices,
            'val': val_indices,
            'test': test_indices
        }
        
        # Validate splits
        if self.config.validate_splits:
            self._validate_splits(splits, data_array, targets)
        
        # Store splits
        self.splits = splits
        self.split_metadata = {
            'split_type': SplitType.TRAIN_VAL_TEST.value,
            'total_samples': total_size,
            'train_samples': len(train_indices),
            'val_samples': len(val_indices),
            'test_samples': len(test_indices),
            'stratified': stratify_target is not None
        }
        
        # Save splits if requested
        if self.config.save_splits:
            self._save_splits()
        
        return splits
    
    def _train_test_split(self, data: Union[Dataset, np.ndarray, pd.DataFrame],
                          targets: Optional[Union[np.ndarray, pd.Series]] = None) -> Dict[str, Any]:
        """Perform train/test split."""
        # Convert data to indices if it's a Dataset
        if isinstance(data, Dataset):
            indices = list(range(len(data)))
            data_array = indices
        else:
            data_array = data
        
        # Calculate split sizes
        total_size = len(data_array)
        train_size = int(total_size * self.config.train_ratio)
        test_size = total_size - train_size
        
        self.logger.info(f"Total samples: {total_size}")
        self.logger.info(f"Train samples: {train_size} ({train_size/total_size:.1%})")
        self.logger.info(f"Test samples: {test_size} ({test_size/total_size:.1%})")
        
        # Prepare stratification target
        stratify_target = None
        if self.config.stratify and targets is not None:
            stratify_target = targets
        elif self.config.stratify and self.config.stratify_column and isinstance(data, pd.DataFrame):
            stratify_target = data[self.config.stratify_column]
        
        # Perform split
        if stratify_target is not None:
            train_indices, test_indices = train_test_split(
                range(total_size),
                train_size=train_size,
                stratify=stratify_target,
                random_state=self.config.random_state
            )
        else:
            train_indices, test_indices = train_test_split(
                range(total_size),
                train_size=train_size,
                random_state=self.config.random_state
            )
        
        # Create splits
        splits = {
            'train': train_indices,
            'test': test_indices
        }
        
        # Validate splits
        if self.config.validate_splits:
            self._validate_splits(splits, data_array, targets)
        
        # Store splits
        self.splits = splits
        self.split_metadata = {
            'split_type': SplitType.TRAIN_TEST.value,
            'total_samples': total_size,
            'train_samples': len(train_indices),
            'test_samples': len(test_indices),
            'stratified': stratify_target is not None
        }
        
        # Save splits if requested
        if self.config.save_splits:
            self._save_splits()
        
        return splits
    
    def _cross_validation_split(self, data: Union[Dataset, np.ndarray, pd.DataFrame],
                               targets: Optional[Union[np.ndarray, pd.Series]] = None) -> Dict[str, Any]:
        """Perform cross-validation split."""
        # Convert data to indices if it's a Dataset
        if isinstance(data, Dataset):
            indices = list(range(len(data)))
            data_array = indices
        else:
            data_array = data
        
        total_size = len(data_array)
        self.logger.info(f"Performing {self.config.cv_folds}-fold cross-validation on {total_size} samples")
        
        # Prepare stratification target
        stratify_target = None
        if self.config.stratify and targets is not None:
            stratify_target = targets
        elif self.config.stratify and self.config.stratify_column and isinstance(data, pd.DataFrame):
            stratify_target = data[self.config.stratify_column]
        
        # Create cross-validation splits
        cv_splits = []
        
        if self.config.cv_strategy == CrossValidationType.K_FOLD:
            cv = KFold(n_splits=self.config.cv_folds, shuffle=True, random_state=self.config.random_state)
        elif self.config.cv_strategy == CrossValidationType.STRATIFIED_K_FOLD:
            cv = StratifiedKFold(n_splits=self.config.cv_folds, shuffle=True, random_state=self.config.random_state)
        elif self.config.cv_strategy == CrossValidationType.SHUFFLE_SPLIT:
            cv = ShuffleSplit(n_splits=self.config.cv_folds, test_size=1/self.config.cv_folds, random_state=self.config.random_state)
        elif self.config.cv_strategy == CrossValidationType.STRATIFIED_SHUFFLE_SPLIT:
            cv = StratifiedShuffleSplit(n_splits=self.config.cv_folds, test_size=1/self.config.cv_folds, random_state=self.config.random_state)
        elif self.config.cv_strategy == CrossValidationType.LEAVE_ONE_OUT:
            cv = LeaveOneOut()
        elif self.config.cv_strategy == CrossValidationType.LEAVE_P_OUT:
            cv = LeavePOut(p=2)
        elif self.config.cv_strategy == CrossValidationType.GROUP_K_FOLD:
            if self.config.group_column and isinstance(data, pd.DataFrame):
                groups = data[self.config.group_column]
                cv = GroupKFold(n_splits=self.config.cv_folds)
            else:
                raise ValueError("Group column required for GroupKFold")
        elif self.config.cv_strategy == CrossValidationType.STRATIFIED_GROUP_K_FOLD:
            if self.config.group_column and isinstance(data, pd.DataFrame):
                groups = data[self.config.group_column]
                cv = StratifiedGroupKFold(n_splits=self.config.cv_folds, shuffle=True, random_state=self.config.random_state)
            else:
                raise ValueError("Group column required for StratifiedGroupKFold")
        elif self.config.cv_strategy == CrossValidationType.TIME_SERIES_SPLIT:
            cv = TimeSeriesSplit(n_splits=self.config.cv_folds)
        elif self.config.cv_strategy == CrossValidationType.REPEATED_K_FOLD:
            cv = RepeatedKFold(n_splits=self.config.cv_folds, n_repeats=self.config.cv_repeats, random_state=self.config.random_state)
        elif self.config.cv_strategy == CrossValidationType.REPEATED_STRATIFIED_K_FOLD:
            cv = RepeatedStratifiedKFold(n_splits=self.config.cv_folds, n_repeats=self.config.cv_repeats, random_state=self.config.random_state)
        else:
            raise ValueError(f"Unknown cross-validation strategy: {self.config.cv_strategy}")
        
        # Generate splits
        if self.config.cv_strategy in [CrossValidationType.GROUP_K_FOLD, CrossValidationType.STRATIFIED_GROUP_K_FOLD]:
            split_generator = cv.split(data_array, stratify_target, groups)
        else:
            split_generator = cv.split(data_array, stratify_target) if stratify_target is not None else cv.split(data_array)
        
        for fold, (train_indices, val_indices) in enumerate(split_generator):
            cv_splits.append({
                'fold': fold,
                'train': train_indices,
                'val': val_indices
            })
            
            self.logger.info(f"Fold {fold + 1}: Train={len(train_indices)}, Val={len(val_indices)}")
        
        # Store CV splits
        self.cv_splits = cv_splits
        self.split_metadata = {
            'split_type': SplitType.CROSS_VALIDATION.value,
            'cv_strategy': self.config.cv_strategy.value,
            'cv_folds': self.config.cv_folds,
            'cv_repeats': self.config.cv_repeats,
            'total_samples': total_size,
            'stratified': stratify_target is not None
        }
        
        # Save splits if requested
        if self.config.save_splits:
            self._save_splits()
        
        return {'cv_splits': cv_splits}
    
    def _time_series_split(self, data: Union[Dataset, np.ndarray, pd.DataFrame],
                           targets: Optional[Union[np.ndarray, pd.Series]] = None) -> Dict[str, Any]:
        """Perform time series split."""
        if self.config.time_column is None:
            raise ValueError("Time column required for time series split")
        
        # Convert data to DataFrame if needed
        if isinstance(data, Dataset):
            raise ValueError("Time series split requires DataFrame or array with time column")
        
        if isinstance(data, np.ndarray):
            # Assume first column is time
            df = pd.DataFrame(data)
            time_col = 0
        else:
            df = data
            time_col = self.config.time_column
        
        # Sort by time
        df_sorted = df.sort_values(time_col)
        total_size = len(df_sorted)
        
        self.logger.info(f"Performing time series split on {total_size} samples")
        
        # Calculate split sizes
        train_size = int(total_size * self.config.train_ratio)
        val_size = int(total_size * self.config.val_ratio)
        test_size = total_size - train_size - val_size
        
        # Create splits
        train_indices = list(range(train_size))
        val_indices = list(range(train_size, train_size + val_size))
        test_indices = list(range(train_size + val_size, total_size))
        
        splits = {
            'train': train_indices,
            'val': val_indices,
            'test': test_indices
        }
        
        # Validate splits
        if self.config.validate_splits:
            self._validate_splits(splits, list(range(total_size)), targets)
        
        # Store splits
        self.splits = splits
        self.split_metadata = {
            'split_type': SplitType.TIME_SERIES.value,
            'time_column': self.config.time_column,
            'total_samples': total_size,
            'train_samples': len(train_indices),
            'val_samples': len(val_indices),
            'test_samples': len(test_indices)
        }
        
        # Save splits if requested
        if self.config.save_splits:
            self._save_splits()
        
        return splits
    
    def _group_split(self, data: Union[Dataset, np.ndarray, pd.DataFrame],
                     targets: Optional[Union[np.ndarray, pd.Series]] = None) -> Dict[str, Any]:
        """Perform group-based split."""
        if self.config.group_column is None:
            raise ValueError("Group column required for group split")
        
        # Convert data to DataFrame if needed
        if isinstance(data, Dataset):
            raise ValueError("Group split requires DataFrame or array with group column")
        
        if isinstance(data, np.ndarray):
            # Assume first column is group
            df = pd.DataFrame(data)
            group_col = 0
        else:
            df = data
            group_col = self.config.group_column
        
        # Get unique groups
        groups = df[group_col].unique()
        total_groups = len(groups)
        
        self.logger.info(f"Performing group split on {total_groups} groups")
        
        # Calculate split sizes
        train_groups = int(total_groups * self.config.train_ratio)
        val_groups = int(total_groups * self.config.val_ratio)
        test_groups = total_groups - train_groups - val_groups
        
        # Shuffle groups
        np.random.shuffle(groups)
        
        # Split groups
        train_group_ids = groups[:train_groups]
        val_group_ids = groups[train_groups:train_groups + val_groups]
        test_group_ids = groups[train_groups + val_groups:]
        
        # Get indices for each split
        train_indices = df[df[group_col].isin(train_group_ids)].index.tolist()
        val_indices = df[df[group_col].isin(val_group_ids)].index.tolist()
        test_indices = df[df[group_col].isin(test_group_ids)].index.tolist()
        
        splits = {
            'train': train_indices,
            'val': val_indices,
            'test': test_indices
        }
        
        # Validate splits
        if self.config.validate_splits:
            self._validate_splits(splits, list(range(len(df))), targets)
        
        # Store splits
        self.splits = splits
        self.split_metadata = {
            'split_type': SplitType.GROUP_SPLIT.value,
            'group_column': self.config.group_column,
            'total_groups': total_groups,
            'train_groups': len(train_group_ids),
            'val_groups': len(val_group_ids),
            'test_groups': len(test_group_ids),
            'total_samples': len(df),
            'train_samples': len(train_indices),
            'val_samples': len(val_indices),
            'test_samples': len(test_indices)
        }
        
        # Save splits if requested
        if self.config.save_splits:
            self._save_splits()
        
        return splits
    
    def _validate_splits(self, splits: Dict[str, List[int]], data: Any, targets: Optional[Any] = None):
        """Validate data splits."""
        self.logger.info("Validating data splits...")
        
        # Check minimum samples per split
        for split_name, indices in splits.items():
            if len(indices) < self.config.min_samples_per_split:
                self.logger.warning(f"{split_name} split has only {len(indices)} samples (minimum: {self.config.min_samples_per_split})")
        
        # Check for overlap
        all_indices = []
        for indices in splits.values():
            all_indices.extend(indices)
        
        if len(all_indices) != len(set(all_indices)):
            self.logger.warning("Overlapping indices detected in splits")
        
        # Check stratification if targets provided
        if targets is not None:
            for split_name, indices in splits.items():
                split_targets = [targets[i] for i in indices]
                unique_targets, counts = np.unique(split_targets, return_counts=True)
                self.logger.info(f"{split_name} split target distribution: {dict(zip(unique_targets, counts))}")
        
        self.logger.info("Split validation completed")
    
    def _save_splits(self) -> Any:
        """Save splits to file."""
        save_data = {
            'splits': self.splits,
            'cv_splits': self.cv_splits,
            'metadata': self.split_metadata,
            'config': {
                'train_ratio': self.config.train_ratio,
                'val_ratio': self.config.val_ratio,
                'test_ratio': self.config.test_ratio,
                'cv_folds': self.config.cv_folds,
                'cv_strategy': self.config.cv_strategy.value,
                'random_state': self.config.random_state
            }
        }
        
        with open(self.config.splits_file, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            json.dump(save_data, f, indent=2)
        
        self.logger.info(f"Splits saved to {self.config.splits_file}")
    
    def load_splits(self, splits_file: str) -> Dict[str, Any]:
        """Load splits from file."""
        with open(splits_file, 'r') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            save_data = json.load(f)
        
        self.splits = save_data.get('splits', {})
        self.cv_splits = save_data.get('cv_splits', {})
        self.split_metadata = save_data.get('metadata', {})
        
        self.logger.info(f"Splits loaded from {splits_file}")
        return save_data
    
    def create_dataloaders(self, dataset: Dataset, batch_size: int = 32,
                          num_workers: int = 4, pin_memory: bool = True) -> Dict[str, DataLoader]:
        """Create DataLoaders from splits."""
        dataloaders = {}
        
        if self.splits:
            for split_name, indices in self.splits.items():
                subset = Subset(dataset, indices)
                dataloaders[split_name] = DataLoader(
                    subset,
                    batch_size=batch_size,
                    shuffle=(split_name == 'train'),
                    num_workers=num_workers,
                    pin_memory=pin_memory
                )
        
        return dataloaders
    
    def create_cv_dataloaders(self, dataset: Dataset, fold: int = 0,
                             batch_size: int = 32, num_workers: int = 4,
                             pin_memory: bool = True) -> Dict[str, DataLoader]:
        """Create DataLoaders for a specific cross-validation fold."""
        if not self.cv_splits or fold >= len(self.cv_splits):
            raise ValueError(f"Invalid fold {fold}")
        
        fold_data = self.cv_splits[fold]
        
        train_subset = Subset(dataset, fold_data['train'])
        val_subset = Subset(dataset, fold_data['val'])
        
        dataloaders = {
            'train': DataLoader(
                train_subset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=pin_memory
            ),
            'val': DataLoader(
                val_subset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory
            )
        }
        
        return dataloaders


class CrossValidationEvaluator:
    """Cross-validation evaluation system."""
    
    def __init__(self, model_class: type, model_params: Dict[str, Any],
                 splitter: DataSplitter):
        
    """__init__ function."""
self.model_class = model_class
        self.model_params = model_params
        self.splitter = splitter
        self.logger = self._setup_logging()
        self.results = {}
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def evaluate_cv(self, dataset: Dataset, targets: np.ndarray,
                   batch_size: int = 32, num_epochs: int = 10,
                   device: str = 'cpu') -> Dict[str, Any]:
        """Evaluate model using cross-validation."""
        self.logger.info("Starting cross-validation evaluation")
        
        cv_results = []
        
        for fold in range(len(self.splitter.cv_splits)):
            self.logger.info(f"Evaluating fold {fold + 1}/{len(self.splitter.cv_splits)}")
            
            # Create dataloaders for this fold
            dataloaders = self.splitter.create_cv_dataloaders(
                dataset, fold, batch_size=batch_size
            )
            
            # Create and train model
            model = self.model_class(**self.model_params).to(device)
            fold_result = self._train_and_evaluate_fold(
                model, dataloaders, targets, num_epochs, device
            )
            
            cv_results.append(fold_result)
        
        # Aggregate results
        aggregated_results = self._aggregate_cv_results(cv_results)
        
        self.results = {
            'cv_results': cv_results,
            'aggregated_results': aggregated_results
        }
        
        return self.results
    
    def _train_and_evaluate_fold(self, model: nn.Module, dataloaders: Dict[str, DataLoader],
                                 targets: np.ndarray, num_epochs: int, device: str) -> Dict[str, Any]:
        """Train and evaluate model for a single fold."""
        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters())
        
        train_losses = []
        val_losses = []
        val_accuracies = []
        
        # Training loop
        for epoch in range(num_epochs):
            # Training
            model.train()
            train_loss = 0.0
            for batch_idx, (data, target) in enumerate(dataloaders['train']):
                data, target = data.to(device), target.to(device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(dataloaders['train'])
            train_losses.append(train_loss)
            
            # Validation
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for data, target in dataloaders['val']:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    loss = criterion(output, target)
                    val_loss += loss.item()
                    
                    _, predicted = torch.max(output.data, 1)
                    total += target.size(0)
                    correct += (predicted == target).sum().item()
            
            val_loss /= len(dataloaders['val'])
            val_accuracy = correct / total
            
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)
            
            if (epoch + 1) % 5 == 0:
                self.logger.info(f"Epoch {epoch + 1}/{num_epochs}: "
                               f"Train Loss: {train_loss:.4f}, "
                               f"Val Loss: {val_loss:.4f}, "
                               f"Val Acc: {val_accuracy:.4f}")
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_accuracies': val_accuracies,
            'final_val_accuracy': val_accuracies[-1],
            'final_val_loss': val_losses[-1]
        }
    
    def _aggregate_cv_results(self, cv_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate cross-validation results."""
        final_accuracies = [result['final_val_accuracy'] for result in cv_results]
        final_losses = [result['final_val_loss'] for result in cv_results]
        
        aggregated = {
            'mean_accuracy': np.mean(final_accuracies),
            'std_accuracy': np.std(final_accuracies),
            'mean_loss': np.mean(final_losses),
            'std_loss': np.std(final_losses),
            'min_accuracy': np.min(final_accuracies),
            'max_accuracy': np.max(final_accuracies),
            'min_loss': np.min(final_losses),
            'max_loss': np.max(final_losses)
        }
        
        self.logger.info(f"Cross-validation results:")
        self.logger.info(f"  Mean Accuracy: {aggregated['mean_accuracy']:.4f} ± {aggregated['std_accuracy']:.4f}")
        self.logger.info(f"  Mean Loss: {aggregated['mean_loss']:.4f} ± {aggregated['std_loss']:.4f}")
        self.logger.info(f"  Accuracy Range: {aggregated['min_accuracy']:.4f} - {aggregated['max_accuracy']:.4f}")
        
        return aggregated
    
    def plot_cv_results(self, save_path: Optional[str] = None):
        """Plot cross-validation results."""
        if not self.results:
            self.logger.warning("No results to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot accuracy per fold
        accuracies = [result['final_val_accuracy'] for result in self.results['cv_results']]
        axes[0, 0].bar(range(1, len(accuracies) + 1), accuracies)
        axes[0, 0].set_title('Validation Accuracy per Fold')
        axes[0, 0].set_xlabel('Fold')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].axhline(y=self.results['aggregated_results']['mean_accuracy'], 
                           color='r', linestyle='--', label='Mean')
        axes[0, 0].legend()
        
        # Plot loss per fold
        losses = [result['final_val_loss'] for result in self.results['cv_results']]
        axes[0, 1].bar(range(1, len(losses) + 1), losses)
        axes[0, 1].set_title('Validation Loss per Fold')
        axes[0, 1].set_xlabel('Fold')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].axhline(y=self.results['aggregated_results']['mean_loss'], 
                           color='r', linestyle='--', label='Mean')
        axes[0, 1].legend()
        
        # Plot training curves for first fold
        if self.results['cv_results']:
            first_fold = self.results['cv_results'][0]
            axes[1, 0].plot(first_fold['train_losses'], label='Train Loss')
            axes[1, 0].plot(first_fold['val_losses'], label='Val Loss')
            axes[1, 0].set_title('Training Curves (Fold 1)')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].legend()
            
            axes[1, 1].plot(first_fold['val_accuracies'], label='Val Accuracy')
            axes[1, 1].set_title('Validation Accuracy (Fold 1)')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Accuracy')
            axes[1, 1].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Plot saved to {save_path}")
        
        plt.show()


def demonstrate_data_splitting():
    """Demonstrate data splitting and validation capabilities."""
    print("Data Splitting and Validation Demonstration")
    print("=" * 50)
    
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    n_classes = 3
    
    # Generate synthetic data
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, n_classes, n_samples)
    
    # Create DataFrame with additional columns
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])
    df['target'] = y
    df['group'] = np.random.randint(0, 10, n_samples)  # 10 groups
    df['time'] = np.arange(n_samples)  # Time series
    
    # Create simple dataset
    class SimpleDataset(Dataset):
        def __init__(self, data, targets) -> Any:
            self.data = torch.FloatTensor(data)
            self.targets = torch.LongTensor(targets)
        
        def __len__(self) -> Any:
            return len(self.data)
        
        def __getitem__(self, idx) -> Optional[Dict[str, Any]]:
            return self.data[idx], self.targets[idx]
    
    dataset = SimpleDataset(X, y)
    
    # Test different split configurations
    configs = [
        SplitConfig(
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            stratify=True,
            random_state=42
        ),
        SplitConfig(
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
            stratify=True,
            random_state=42
        ),
        SplitConfig(
            cv_folds=5,
            cv_strategy=CrossValidationType.STRATIFIED_K_FOLD,
            stratify=True,
            random_state=42
        )
    ]
    
    results = {}
    
    for i, config in enumerate(configs):
        print(f"\nTesting configuration {i+1}:")
        print(f"  Train ratio: {config.train_ratio}")
        print(f"  Val ratio: {config.val_ratio}")
        print(f"  Test ratio: {config.test_ratio}")
        print(f"  CV folds: {config.cv_folds}")
        print(f"  CV strategy: {config.cv_strategy.value}")
        
        try:
            # Create splitter
            splitter = DataSplitter(config)
            
            if config.cv_folds > 0:
                # Cross-validation split
                splits = splitter.split_data(
                    dataset, targets=y,
                    split_type=SplitType.CROSS_VALIDATION
                )
                
                print(f"  Created {len(splits['cv_splits'])} CV folds")
                
                # Test dataloader creation
                dataloaders = splitter.create_cv_dataloaders(dataset, fold=0, batch_size=32)
                print(f"  Created dataloaders for fold 0: {list(dataloaders.keys())}")
                
            else:
                # Train/val/test split
                splits = splitter.split_data(
                    dataset, targets=y,
                    split_type=SplitType.TRAIN_VAL_TEST
                )
                
                print(f"  Created splits: {list(splits.keys())}")
                
                # Test dataloader creation
                dataloaders = splitter.create_dataloaders(dataset, batch_size=32)
                print(f"  Created dataloaders: {list(dataloaders.keys())}")
            
            results[f"config_{i}"] = {
                'config': config,
                'splits': splits,
                'success': True
            }
            
        except Exception as e:
            print(f"  Error: {e}")
            results[f"config_{i}"] = {
                'config': config,
                'error': str(e),
                'success': False
            }
    
    # Test group and time series splits
    print(f"\nTesting group split:")
    try:
        group_config = SplitConfig(
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            group_column='group',
            random_state=42
        )
        
        group_splitter = DataSplitter(group_config)
        group_splits = group_splitter.split_data(
            df, targets=y,
            split_type=SplitType.GROUP_SPLIT
        )
        
        print(f"  Group split created successfully")
        print(f"  Metadata: {group_splitter.split_metadata}")
        
    except Exception as e:
        print(f"  Group split error: {e}")
    
    print(f"\nTesting time series split:")
    try:
        time_config = SplitConfig(
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            time_column='time',
            random_state=42
        )
        
        time_splitter = DataSplitter(time_config)
        time_splits = time_splitter.split_data(
            df, targets=y,
            split_type=SplitType.TIME_SERIES
        )
        
        print(f"  Time series split created successfully")
        print(f"  Metadata: {time_splitter.split_metadata}")
        
    except Exception as e:
        print(f"  Time series split error: {e}")
    
    return results


if __name__ == "__main__":
    # Demonstrate data splitting and validation
    results = demonstrate_data_splitting()
    print("\nData splitting and validation demonstration completed!") 