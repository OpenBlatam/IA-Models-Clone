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
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
import logging
from sklearn.model_selection import (
from sklearn.preprocessing import LabelEncoder
import warnings
        from collections import Counter
    import pandas as pd
from typing import Any, List, Dict, Optional
import asyncio
#!/usr/bin/env python3
"""
Data Splitting and Cross-Validation Framework for SEO Deep Learning System
- Proper train/validation/test splits
- Stratified sampling for imbalanced datasets
- Multiple cross-validation strategies
- Time series aware splitting
- Custom splitting strategies for SEO data
"""

    train_test_split, StratifiedKFold, KFold, TimeSeriesSplit,
    GroupKFold, StratifiedGroupKFold, LeaveOneOut, LeavePOut,
    RepeatedStratifiedKFold, RepeatedKFold
)
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class DataSplitConfig:
    """Configuration for data splitting"""
    # Split ratios
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    # Cross-validation configuration
    use_cross_validation: bool = False
    cv_folds: int = 5
    cv_strategy: str = "stratified"  # "stratified", "kfold", "timeseries", "group", "stratified_group"
    cv_repeats: int = 1
    
    # Stratification configuration
    stratify_by: Optional[str] = None  # Column name for stratification
    group_by: Optional[str] = None  # Column name for group-based splitting
    
    # Time series configuration
    time_column: Optional[str] = None  # Column name for time ordering
    time_series_split: bool = False
    
    # SEO-specific configuration
    seo_domain_split: bool = False  # Split by domain to avoid data leakage
    seo_keyword_split: bool = False  # Split by keyword groups
    seo_content_type_split: bool = False  # Split by content type
    
    # Random state
    random_state: int = 42
    
    # Validation
    shuffle: bool = True
    preserve_order: bool = False  # For time series data
    
    def __post_init__(self) -> Any:
        """Validate configuration"""
        total_ratio = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total_ratio - 1.0) > 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio}")
        
        if self.cv_folds < 2:
            raise ValueError(f"CV folds must be >= 2, got {self.cv_folds}")
        
        if self.cv_repeats < 1:
            raise ValueError(f"CV repeats must be >= 1, got {self.cv_repeats}")

@dataclass
class DataSplit:
    """Container for data splits"""
    train_indices: List[int]
    val_indices: List[int]
    test_indices: List[int]
    split_info: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CrossValidationSplit:
    """Container for cross-validation splits"""
    fold_splits: List[DataSplit]
    cv_info: Dict[str, Any] = field(default_factory=dict)

class DataSplitter:
    """Advanced data splitting with multiple strategies"""
    
    def __init__(self, config: DataSplitConfig):
        
    """__init__ function."""
self.config = config
        self.label_encoder = None
        self.group_encoder = None
        
        logger.info(f"DataSplitter initialized with config: {config}")
    
    def split_data(self, 
                  data: Union[List, np.ndarray, pd.DataFrame, Dataset],
                  labels: Optional[Union[List, np.ndarray]] = None,
                  groups: Optional[Union[List, np.ndarray]] = None) -> DataSplit:
        """Split data into train/validation/test sets"""
        
        if isinstance(data, Dataset):
            return self._split_dataset(data, labels, groups)
        elif isinstance(data, pd.DataFrame):
            return self._split_dataframe(data, labels, groups)
        else:
            return self._split_array(data, labels, groups)
    
    def _split_dataset(self, 
                      dataset: Dataset, 
                      labels: Optional[List] = None,
                      groups: Optional[List] = None) -> DataSplit:
        """Split PyTorch dataset"""
        n_samples = len(dataset)
        indices = list(range(n_samples))
        
        # Get labels from dataset if not provided
        if labels is None:
            labels = self._extract_labels_from_dataset(dataset)
        
        # Get groups from dataset if not provided
        if groups is None and self.config.group_by:
            groups = self._extract_groups_from_dataset(dataset)
        
        return self._perform_split(indices, labels, groups)
    
    def _split_dataframe(self, 
                        df: pd.DataFrame, 
                        labels: Optional[List] = None,
                        groups: Optional[List] = None) -> DataSplit:
        """Split pandas DataFrame"""
        n_samples = len(df)
        indices = list(range(n_samples))
        
        # Extract labels from DataFrame
        if labels is None and self.config.stratify_by:
            labels = df[self.config.stratify_by].values
        
        # Extract groups from DataFrame
        if groups is None and self.config.group_by:
            groups = df[self.config.group_by].values
        
        return self._perform_split(indices, labels, groups)
    
    def _split_array(self, 
                    data: Union[List, np.ndarray], 
                    labels: Optional[List] = None,
                    groups: Optional[List] = None) -> DataSplit:
        """Split array-like data"""
        n_samples = len(data)
        indices = list(range(n_samples))
        
        return self._perform_split(indices, labels, groups)
    
    def _extract_labels_from_dataset(self, dataset: Dataset) -> List:
        """Extract labels from dataset"""
        labels = []
        for i in range(min(1000, len(dataset))):  # Sample first 1000 items
            try:
                item = dataset[i]
                if isinstance(item, dict) and 'labels' in item:
                    labels.append(item['labels'])
                elif isinstance(item, (list, tuple)) and len(item) > 1:
                    labels.append(item[1])
                else:
                    labels.append(0)  # Default label
            except:
                labels.append(0)
        
        # Extend with default labels if needed
        while len(labels) < len(dataset):
            labels.append(0)
        
        return labels[:len(dataset)]
    
    def _extract_groups_from_dataset(self, dataset: Dataset) -> List:
        """Extract groups from dataset"""
        groups = []
        for i in range(min(1000, len(dataset))):  # Sample first 1000 items
            try:
                item = dataset[i]
                if isinstance(item, dict) and 'group' in item:
                    groups.append(item['group'])
                elif isinstance(item, dict) and 'domain' in item:
                    groups.append(item['domain'])
                else:
                    groups.append(f"group_{i}")
            except:
                groups.append(f"group_{i}")
        
        # Extend with default groups if needed
        while len(groups) < len(dataset):
            groups.append(f"group_{len(groups)}")
        
        return groups[:len(dataset)]
    
    def _perform_split(self, 
                      indices: List[int], 
                      labels: Optional[List] = None,
                      groups: Optional[List] = None) -> DataSplit:
        """Perform the actual data splitting"""
        
        if self.config.time_series_split:
            return self._time_series_split(indices)
        
        # First split: train vs (val + test)
        train_size = int(self.config.train_ratio * len(indices))
        remaining_size = len(indices) - train_size
        
        if labels is not None and self.config.stratify_by:
            # Stratified split
            train_indices, remaining_indices = train_test_split(
                indices,
                train_size=train_size,
                stratify=labels,
                random_state=self.config.random_state,
                shuffle=self.config.shuffle
            )
        elif groups is not None and self.config.group_by:
            # Group-based split
            train_indices, remaining_indices = self._group_based_split(
                indices, groups, train_size
            )
        else:
            # Random split
            if self.config.shuffle:
                np.random.seed(self.config.random_state)
                np.random.shuffle(indices)
            
            train_indices = indices[:train_size]
            remaining_indices = indices[train_size:]
        
        # Second split: validation vs test
        val_size = int(self.config.val_ratio * len(indices))
        test_size = remaining_size - val_size
        
        if labels is not None and self.config.stratify_by:
            # Get labels for remaining data
            remaining_labels = [labels[i] for i in remaining_indices]
            
            val_indices, test_indices = train_test_split(
                remaining_indices,
                train_size=val_size,
                stratify=remaining_labels,
                random_state=self.config.random_state,
                shuffle=self.config.shuffle
            )
        else:
            # Random split
            if self.config.shuffle:
                np.random.shuffle(remaining_indices)
            
            val_indices = remaining_indices[:val_size]
            test_indices = remaining_indices[val_size:]
        
        # Create split info
        split_info = {
            'total_samples': len(indices),
            'train_samples': len(train_indices),
            'val_samples': len(val_indices),
            'test_samples': len(test_indices),
            'train_ratio': len(train_indices) / len(indices),
            'val_ratio': len(val_indices) / len(indices),
            'test_ratio': len(test_indices) / len(indices),
            'stratified': labels is not None and self.config.stratify_by,
            'grouped': groups is not None and self.config.group_by,
            'time_series': self.config.time_series_split
        }
        
        return DataSplit(
            train_indices=train_indices,
            val_indices=val_indices,
            test_indices=test_indices,
            split_info=split_info
        )
    
    def _time_series_split(self, indices: List[int]) -> DataSplit:
        """Time series aware splitting"""
        n_samples = len(indices)
        
        # Calculate split points
        train_end = int(self.config.train_ratio * n_samples)
        val_end = train_end + int(self.config.val_ratio * n_samples)
        
        train_indices = indices[:train_end]
        val_indices = indices[train_end:val_end]
        test_indices = indices[val_end:]
        
        split_info = {
            'total_samples': n_samples,
            'train_samples': len(train_indices),
            'val_samples': len(val_indices),
            'test_samples': len(test_indices),
            'time_series': True,
            'preserve_order': True
        }
        
        return DataSplit(
            train_indices=train_indices,
            val_indices=val_indices,
            test_indices=test_indices,
            split_info=split_info
        )
    
    def _group_based_split(self, 
                          indices: List[int], 
                          groups: List, 
                          train_size: int) -> Tuple[List[int], List[int]]:
        """Group-based splitting to avoid data leakage"""
        # Encode groups
        if self.group_encoder is None:
            self.group_encoder = LabelEncoder()
            self.group_encoder.fit(groups)
        
        group_encoded = self.group_encoder.transform(groups)
        
        # Use GroupKFold for splitting
        group_kfold = GroupKFold(n_splits=2)
        splits = list(group_kfold.split(indices, groups=group_encoded))
        
        train_indices = [indices[i] for i in splits[0][0]]
        remaining_indices = [indices[i] for i in splits[0][1]]
        
        return train_indices, remaining_indices

class CrossValidator:
    """Cross-validation with multiple strategies"""
    
    def __init__(self, config: DataSplitConfig):
        
    """__init__ function."""
self.config = config
        self.splits = []
        
        logger.info(f"CrossValidator initialized with {config.cv_folds} folds")
    
    def cross_validate(self, 
                      data: Union[List, np.ndarray, pd.DataFrame, Dataset],
                      labels: Optional[Union[List, np.ndarray]] = None,
                      groups: Optional[Union[List, np.ndarray]] = None) -> CrossValidationSplit:
        """Perform cross-validation"""
        
        if isinstance(data, Dataset):
            return self._cross_validate_dataset(data, labels, groups)
        elif isinstance(data, pd.DataFrame):
            return self._cross_validate_dataframe(data, labels, groups)
        else:
            return self._cross_validate_array(data, labels, groups)
    
    def _cross_validate_dataset(self, 
                               dataset: Dataset, 
                               labels: Optional[List] = None,
                               groups: Optional[List] = None) -> CrossValidationSplit:
        """Cross-validate PyTorch dataset"""
        n_samples = len(dataset)
        indices = list(range(n_samples))
        
        # Get labels from dataset if not provided
        if labels is None:
            labels = self._extract_labels_from_dataset(dataset)
        
        # Get groups from dataset if not provided
        if groups is None and self.config.group_by:
            groups = self._extract_groups_from_dataset(dataset)
        
        return self._perform_cross_validation(indices, labels, groups)
    
    def _cross_validate_dataframe(self, 
                                 df: pd.DataFrame, 
                                 labels: Optional[List] = None,
                                 groups: Optional[List] = None) -> CrossValidationSplit:
        """Cross-validate pandas DataFrame"""
        n_samples = len(df)
        indices = list(range(n_samples))
        
        # Extract labels from DataFrame
        if labels is None and self.config.stratify_by:
            labels = df[self.config.stratify_by].values
        
        # Extract groups from DataFrame
        if groups is None and self.config.group_by:
            groups = df[self.config.group_by].values
        
        return self._perform_cross_validation(indices, labels, groups)
    
    def _cross_validate_array(self, 
                             data: Union[List, np.ndarray], 
                             labels: Optional[List] = None,
                             groups: Optional[List] = None) -> CrossValidationSplit:
        """Cross-validate array-like data"""
        n_samples = len(data)
        indices = list(range(n_samples))
        
        return self._perform_cross_validation(indices, labels, groups)
    
    def _extract_labels_from_dataset(self, dataset: Dataset) -> List:
        """Extract labels from dataset"""
        labels = []
        for i in range(min(1000, len(dataset))):  # Sample first 1000 items
            try:
                item = dataset[i]
                if isinstance(item, dict) and 'labels' in item:
                    labels.append(item['labels'])
                elif isinstance(item, (list, tuple)) and len(item) > 1:
                    labels.append(item[1])
                else:
                    labels.append(0)  # Default label
            except:
                labels.append(0)
        
        # Extend with default labels if needed
        while len(labels) < len(dataset):
            labels.append(0)
        
        return labels[:len(dataset)]
    
    def _extract_groups_from_dataset(self, dataset: Dataset) -> List:
        """Extract groups from dataset"""
        groups = []
        for i in range(min(1000, len(dataset))):  # Sample first 1000 items
            try:
                item = dataset[i]
                if isinstance(item, dict) and 'group' in item:
                    groups.append(item['group'])
                elif isinstance(item, dict) and 'domain' in item:
                    groups.append(item['domain'])
                else:
                    groups.append(f"group_{i}")
            except:
                groups.append(f"group_{i}")
        
        # Extend with default groups if needed
        while len(groups) < len(dataset):
            groups.append(f"group_{len(groups)}")
        
        return groups[:len(dataset)]
    
    def _perform_cross_validation(self, 
                                 indices: List[int], 
                                 labels: Optional[List] = None,
                                 groups: Optional[List] = None) -> CrossValidationSplit:
        """Perform cross-validation with specified strategy"""
        
        # Create cross-validation splitter
        if self.config.cv_strategy == "stratified":
            if self.config.cv_repeats > 1:
                cv = RepeatedStratifiedKFold(
                    n_splits=self.config.cv_folds,
                    n_repeats=self.config.cv_repeats,
                    random_state=self.config.random_state
                )
            else:
                cv = StratifiedKFold(
                    n_splits=self.config.cv_folds,
                    shuffle=self.config.shuffle,
                    random_state=self.config.random_state
                )
        elif self.config.cv_strategy == "kfold":
            if self.config.cv_repeats > 1:
                cv = RepeatedKFold(
                    n_splits=self.config.cv_folds,
                    n_repeats=self.config.cv_repeats,
                    random_state=self.config.random_state
                )
            else:
                cv = KFold(
                    n_splits=self.config.cv_folds,
                    shuffle=self.config.shuffle,
                    random_state=self.config.random_state
                )
        elif self.config.cv_strategy == "timeseries":
            cv = TimeSeriesSplit(n_splits=self.config.cv_folds)
        elif self.config.cv_strategy == "group":
            cv = GroupKFold(n_splits=self.config.cv_folds)
        elif self.config.cv_strategy == "stratified_group":
            cv = StratifiedGroupKFold(n_splits=self.config.cv_folds, shuffle=self.config.shuffle, random_state=self.config.random_state)
        else:
            raise ValueError(f"Unknown CV strategy: {self.config.cv_strategy}")
        
        # Generate splits
        fold_splits = []
        
        if self.config.cv_strategy in ["group", "stratified_group"] and groups is not None:
            splits = cv.split(indices, labels, groups=groups)
        elif self.config.cv_strategy in ["stratified", "stratified_group"] and labels is not None:
            splits = cv.split(indices, labels)
        else:
            splits = cv.split(indices)
        
        for fold_idx, (train_val_indices, test_indices) in enumerate(splits):
            # Split train_val into train and validation
            train_val_size = len(train_val_indices)
            val_size = int(train_val_size * self.config.val_ratio / (self.config.train_ratio + self.config.val_ratio))
            
            if labels is not None and self.config.stratify_by:
                # Stratified split for train/val
                train_val_labels = [labels[i] for i in train_val_indices]
                train_indices, val_indices = train_test_split(
                    train_val_indices,
                    test_size=val_size,
                    stratify=train_val_labels,
                    random_state=self.config.random_state,
                    shuffle=self.config.shuffle
                )
            else:
                # Random split for train/val
                if self.config.shuffle:
                    np.random.seed(self.config.random_state + fold_idx)
                    np.random.shuffle(train_val_indices)
                
                train_indices = train_val_indices[:-val_size]
                val_indices = train_val_indices[-val_size:]
            
            # Create split info
            split_info = {
                'fold': fold_idx,
                'total_samples': len(indices),
                'train_samples': len(train_indices),
                'val_samples': len(val_indices),
                'test_samples': len(test_indices),
                'cv_strategy': self.config.cv_strategy
            }
            
            fold_splits.append(DataSplit(
                train_indices=[indices[i] for i in train_indices],
                val_indices=[indices[i] for i in val_indices],
                test_indices=[indices[i] for i in test_indices],
                split_info=split_info
            ))
        
        # Create CV info
        cv_info = {
            'cv_strategy': self.config.cv_strategy,
            'cv_folds': self.config.cv_folds,
            'cv_repeats': self.config.cv_repeats,
            'total_samples': len(indices),
            'stratified': labels is not None and self.config.stratify_by,
            'grouped': groups is not None and self.config.group_by
        }
        
        return CrossValidationSplit(
            fold_splits=fold_splits,
            cv_info=cv_info
        )

class SEOSpecificSplitter:
    """SEO-specific data splitting strategies"""
    
    def __init__(self, config: DataSplitConfig):
        
    """__init__ function."""
self.config = config
        self.data_splitter = DataSplitter(config)
        self.cross_validator = CrossValidator(config)
        
        logger.info("SEOSpecificSplitter initialized")
    
    def split_by_domain(self, data: Union[List, pd.DataFrame, Dataset]) -> DataSplit:
        """Split data by domain to avoid cross-domain leakage"""
        if isinstance(data, pd.DataFrame):
            domains = data['domain'].values if 'domain' in data.columns else None
        elif isinstance(data, Dataset):
            domains = self._extract_domains_from_dataset(data)
        else:
            domains = None
        
        return self.data_splitter.split_data(data, groups=domains)
    
    def split_by_keyword_groups(self, data: Union[List, pd.DataFrame, Dataset]) -> DataSplit:
        """Split data by keyword groups"""
        if isinstance(data, pd.DataFrame):
            keywords = data['keyword'].values if 'keyword' in data.columns else None
        elif isinstance(data, Dataset):
            keywords = self._extract_keywords_from_dataset(data)
        else:
            keywords = None
        
        # Group keywords by similarity or category
        keyword_groups = self._group_keywords(keywords) if keywords else None
        
        return self.data_splitter.split_data(data, groups=keyword_groups)
    
    def split_by_content_type(self, data: Union[List, pd.DataFrame, Dataset]) -> DataSplit:
        """Split data by content type (blog, product, landing page, etc.)"""
        if isinstance(data, pd.DataFrame):
            content_types = data['content_type'].values if 'content_type' in data.columns else None
        elif isinstance(data, Dataset):
            content_types = self._extract_content_types_from_dataset(data)
        else:
            content_types = None
        
        return self.data_splitter.split_data(data, groups=content_types)
    
    def _extract_domains_from_dataset(self, dataset: Dataset) -> List:
        """Extract domains from dataset"""
        domains = []
        for i in range(min(1000, len(dataset))):
            try:
                item = dataset[i]
                if isinstance(item, dict) and 'domain' in item:
                    domains.append(item['domain'])
                else:
                    domains.append(f"domain_{i}")
            except:
                domains.append(f"domain_{i}")
        
        while len(domains) < len(dataset):
            domains.append(f"domain_{len(domains)}")
        
        return domains[:len(dataset)]
    
    def _extract_keywords_from_dataset(self, dataset: Dataset) -> List:
        """Extract keywords from dataset"""
        keywords = []
        for i in range(min(1000, len(dataset))):
            try:
                item = dataset[i]
                if isinstance(item, dict) and 'keyword' in item:
                    keywords.append(item['keyword'])
                else:
                    keywords.append(f"keyword_{i}")
            except:
                keywords.append(f"keyword_{i}")
        
        while len(keywords) < len(dataset):
            keywords.append(f"keyword_{len(keywords)}")
        
        return keywords[:len(dataset)]
    
    def _extract_content_types_from_dataset(self, dataset: Dataset) -> List:
        """Extract content types from dataset"""
        content_types = []
        for i in range(min(1000, len(dataset))):
            try:
                item = dataset[i]
                if isinstance(item, dict) and 'content_type' in item:
                    content_types.append(item['content_type'])
                else:
                    content_types.append("unknown")
            except:
                content_types.append("unknown")
        
        while len(content_types) < len(dataset):
            content_types.append("unknown")
        
        return content_types[:len(dataset)]
    
    def _group_keywords(self, keywords: List) -> List:
        """Group keywords by similarity or category"""
        # Simple grouping by first word (can be enhanced with semantic similarity)
        keyword_groups = []
        for keyword in keywords:
            if isinstance(keyword, str) and keyword:
                first_word = keyword.split()[0].lower()
                keyword_groups.append(first_word)
            else:
                keyword_groups.append("unknown")
        
        return keyword_groups

class DataSplitManager:
    """High-level manager for data splitting and cross-validation"""
    
    def __init__(self, config: DataSplitConfig):
        
    """__init__ function."""
self.config = config
        self.data_splitter = DataSplitter(config)
        self.cross_validator = CrossValidator(config)
        self.seo_splitter = SEOSpecificSplitter(config)
        
        logger.info("DataSplitManager initialized")
    
    def create_splits(self, 
                     data: Union[List, np.ndarray, pd.DataFrame, Dataset],
                     labels: Optional[Union[List, np.ndarray]] = None,
                     groups: Optional[Union[List, np.ndarray]] = None) -> Union[DataSplit, CrossValidationSplit]:
        """Create data splits based on configuration"""
        
        if self.config.use_cross_validation:
            return self.cross_validator.cross_validate(data, labels, groups)
        else:
            return self.data_splitter.split_data(data, labels, groups)
    
    def create_seo_splits(self, 
                         data: Union[List, pd.DataFrame, Dataset],
                         split_strategy: str = "domain") -> DataSplit:
        """Create SEO-specific splits"""
        
        if split_strategy == "domain":
            return self.seo_splitter.split_by_domain(data)
        elif split_strategy == "keyword":
            return self.seo_splitter.split_by_keyword_groups(data)
        elif split_strategy == "content_type":
            return self.seo_splitter.split_by_content_type(data)
        else:
            raise ValueError(f"Unknown SEO split strategy: {split_strategy}")
    
    def create_datasets(self, 
                       dataset: Dataset,
                       split: Union[DataSplit, CrossValidationSplit]) -> Union[Tuple[Dataset, Dataset, Dataset], List[Tuple[Dataset, Dataset, Dataset]]]:
        """Create PyTorch datasets from splits"""
        
        if isinstance(split, DataSplit):
            train_dataset = Subset(dataset, split.train_indices)
            val_dataset = Subset(dataset, split.val_indices)
            test_dataset = Subset(dataset, split.test_indices)
            
            return train_dataset, val_dataset, test_dataset
        
        elif isinstance(split, CrossValidationSplit):
            fold_datasets = []
            for fold_split in split.fold_splits:
                train_dataset = Subset(dataset, fold_split.train_indices)
                val_dataset = Subset(dataset, fold_split.val_indices)
                test_dataset = Subset(dataset, fold_split.test_indices)
                
                fold_datasets.append((train_dataset, val_dataset, test_dataset))
            
            return fold_datasets
    
    def analyze_splits(self, 
                      split: Union[DataSplit, CrossValidationSplit],
                      labels: Optional[List] = None) -> Dict[str, Any]:
        """Analyze the quality of data splits"""
        
        if isinstance(split, DataSplit):
            return self._analyze_single_split(split, labels)
        elif isinstance(split, CrossValidationSplit):
            return self._analyze_cross_validation_splits(split, labels)
    
    def _analyze_single_split(self, split: DataSplit, labels: Optional[List] = None) -> Dict[str, Any]:
        """Analyze a single data split"""
        
        analysis = {
            'split_info': split.split_info,
            'sample_counts': {
                'train': len(split.train_indices),
                'val': len(split.val_indices),
                'test': len(split.test_indices)
            }
        }
        
        if labels is not None:
            # Analyze label distribution
            train_labels = [labels[i] for i in split.train_indices]
            val_labels = [labels[i] for i in split.val_indices]
            test_labels = [labels[i] for i in split.test_indices]
            
            analysis['label_distribution'] = {
                'train': self._get_label_distribution(train_labels),
                'val': self._get_label_distribution(val_labels),
                'test': self._get_label_distribution(test_labels)
            }
            
            # Check stratification quality
            analysis['stratification_quality'] = self._check_stratification_quality(
                train_labels, val_labels, test_labels
            )
        
        return analysis
    
    def _analyze_cross_validation_splits(self, split: CrossValidationSplit, labels: Optional[List] = None) -> Dict[str, Any]:
        """Analyze cross-validation splits"""
        
        analysis = {
            'cv_info': split.cv_info,
            'fold_analyses': []
        }
        
        for fold_split in split.fold_splits:
            fold_analysis = self._analyze_single_split(fold_split, labels)
            analysis['fold_analyses'].append(fold_analysis)
        
        # Aggregate statistics across folds
        if labels is not None:
            analysis['aggregate_statistics'] = self._aggregate_cv_statistics(analysis['fold_analyses'])
        
        return analysis
    
    def _get_label_distribution(self, labels: List) -> Dict:
        """Get label distribution"""
        counter = Counter(labels)
        total = len(labels)
        
        return {
            'counts': dict(counter),
            'percentages': {label: count/total for label, count in counter.items()}
        }
    
    def _check_stratification_quality(self, 
                                    train_labels: List, 
                                    val_labels: List, 
                                    test_labels: List) -> Dict[str, float]:
        """Check the quality of stratification"""
        
        train_dist = self._get_label_distribution(train_labels)
        val_dist = self._get_label_distribution(val_labels)
        test_dist = self._get_label_distribution(test_labels)
        
        # Calculate distribution differences
        all_labels = set(train_dist['percentages'].keys()) | set(val_dist['percentages'].keys()) | set(test_dist['percentages'].keys())
        
        train_val_diff = sum(abs(train_dist['percentages'].get(label, 0) - val_dist['percentages'].get(label, 0)) for label in all_labels)
        train_test_diff = sum(abs(train_dist['percentages'].get(label, 0) - test_dist['percentages'].get(label, 0)) for label in all_labels)
        
        return {
            'train_val_difference': train_val_diff,
            'train_test_difference': train_test_diff,
            'overall_quality': 1.0 - (train_val_diff + train_test_diff) / 2
        }
    
    def _aggregate_cv_statistics(self, fold_analyses: List[Dict]) -> Dict:
        """Aggregate statistics across cross-validation folds"""
        
        # Calculate average stratification quality
        qualities = [analysis['stratification_quality']['overall_quality'] for analysis in fold_analyses]
        
        return {
            'mean_stratification_quality': np.mean(qualities),
            'std_stratification_quality': np.std(qualities),
            'min_stratification_quality': np.min(qualities),
            'max_stratification_quality': np.max(qualities)
        }

# Example usage
if __name__ == "__main__":
    # Example: Create sample data
    
    # Sample SEO data
    data = pd.DataFrame({
        'text': [f"Sample text {i}" for i in range(1000)],
        'domain': [f"domain_{i % 10}" for i in range(1000)],
        'keyword': [f"keyword_{i % 20}" for i in range(1000)],
        'content_type': [f"type_{i % 5}" for i in range(1000)],
        'labels': [i % 3 for i in range(1000)]  # 3 classes
    })
    
    # Configuration for standard splitting
    config = DataSplitConfig(
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        stratify_by='labels',
        random_state=42
    )
    
    # Create split manager
    split_manager = DataSplitManager(config)
    
    # Create standard splits
    splits = split_manager.create_splits(data, labels=data['labels'].values)
    
    # Analyze splits
    analysis = split_manager.analyze_splits(splits, labels=data['labels'].values)
    
    print("Standard Split Analysis:")
    print(f"Train samples: {analysis['sample_counts']['train']}")
    print(f"Val samples: {analysis['sample_counts']['val']}")
    print(f"Test samples: {analysis['sample_counts']['test']}")
    print(f"Stratification quality: {analysis['stratification_quality']['overall_quality']:.4f}")
    
    # Configuration for cross-validation
    cv_config = DataSplitConfig(
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        use_cross_validation=True,
        cv_folds=5,
        cv_strategy="stratified",
        stratify_by='labels',
        random_state=42
    )
    
    # Create cross-validation splits
    cv_split_manager = DataSplitManager(cv_config)
    cv_splits = cv_split_manager.create_splits(data, labels=data['labels'].values)
    
    # Analyze cross-validation splits
    cv_analysis = cv_split_manager.analyze_splits(cv_splits, labels=data['labels'].values)
    
    print("\nCross-Validation Analysis:")
    print(f"Number of folds: {len(cv_splits.fold_splits)}")
    print(f"Mean stratification quality: {cv_analysis['aggregate_statistics']['mean_stratification_quality']:.4f}")
    print(f"Std stratification quality: {cv_analysis['aggregate_statistics']['std_stratification_quality']:.4f}") 