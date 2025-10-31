from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import torch
from torch.utils.data import DataLoader, Dataset, random_split, Subset
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
from typing import Tuple, List, Optional, Dict, Any, Union
from sklearn.model_selection import KFold, StratifiedKFold
import pandas as pd
from pathlib import Path
import logging
from dataclasses import dataclass
import json

from typing import Any, List, Dict, Optional
import asyncio
logger = logging.getLogger(__name__)

@dataclass
class DataConfig:
    """Configuration for data loading and splitting."""
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    shuffle: bool = True
    drop_last: bool = False
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    random_seed: int = 42
    cross_validation_folds: int = 5
    stratified_splitting: bool = True

class SecurityDataset(Dataset):
    """Custom dataset for security-related data."""
    
    def __init__(self, data: np.ndarray, labels: np.ndarray, 
                 transform: Optional[callable] = None):
        """
        Initialize dataset with security data.
        
        Args:
            data: Input features (n_samples, n_features)
            labels: Target labels (n_samples,)
            transform: Optional transformation function
        """
        self.data = torch.FloatTensor(data)
        self.labels = torch.LongTensor(labels)
        self.transform = transform
        
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.data[idx]
        label = self.labels[idx]
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample, label

class TextSecurityDataset(Dataset):
    """Dataset for text-based security data."""
    
    def __init__(self, texts: List[str], labels: List[int], 
                 tokenizer, max_length: int = 512):
        """
        Initialize text dataset.
        
        Args:
            texts: List of text samples
            labels: List of labels
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.labels = torch.LongTensor(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': label
        }

class EfficientDataLoader:
    """Efficient data loading with proper splits and cross-validation."""
    
    def __init__(self, config: DataConfig):
        
    """__init__ function."""
self.config = config
        self.setup_random_seed()
        
    def setup_random_seed(self) -> None:
        """Set random seed for reproducibility."""
        torch.manual_seed(self.config.random_seed)
        np.random.seed(self.config.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.config.random_seed)
            torch.cuda.manual_seed_all(self.config.random_seed)
            
    def create_data_loaders(self, dataset: Dataset) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create train/validation/test data loaders.
        
        Args:
            dataset: PyTorch dataset
            
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        total_size = len(dataset)
        train_size = int(self.config.train_split * total_size)
        val_size = int(self.config.val_split * total_size)
        test_size = total_size - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(self.config.random_seed)
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=self.config.shuffle,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            drop_last=self.config.drop_last
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            drop_last=False
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            drop_last=False
        )
        
        logger.info(f"Created data loaders - Train: {len(train_dataset)}, "
                   f"Val: {len(val_dataset)}, Test: {len(test_dataset)}")
        
        return train_loader, val_loader, test_loader
    
    def create_cross_validation_loaders(self, dataset: Dataset, 
                                      labels: Optional[np.ndarray] = None) -> List[Tuple[DataLoader, DataLoader]]:
        """
        Create cross-validation data loaders.
        
        Args:
            dataset: PyTorch dataset
            labels: Labels for stratified splitting (optional)
            
        Returns:
            List of (train_loader, val_loader) tuples for each fold
        """
        total_size = len(dataset)
        indices = list(range(total_size))
        
        if self.config.stratified_splitting and labels is not None:
            kfold = StratifiedKFold(
                n_splits=self.config.cross_validation_folds,
                shuffle=True,
                random_state=self.config.random_seed
            )
            splits = kfold.split(indices, labels)
        else:
            kfold = KFold(
                n_splits=self.config.cross_validation_folds,
                shuffle=True,
                random_state=self.config.random_seed
            )
            splits = kfold.split(indices)
        
        cv_loaders = []
        
        for fold, (train_idx, val_idx) in enumerate(splits):
            train_sampler = SubsetRandomSampler(train_idx)
            val_sampler = SubsetRandomSampler(val_idx)
            
            train_loader = DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                sampler=train_sampler,
                num_workers=self.config.num_workers,
                pin_memory=self.config.pin_memory,
                drop_last=self.config.drop_last
            )
            
            val_loader = DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                sampler=val_sampler,
                num_workers=self.config.num_workers,
                pin_memory=self.config.pin_memory,
                drop_last=False
            )
            
            cv_loaders.append((train_loader, val_loader))
            logger.info(f"Fold {fold + 1}: Train {len(train_idx)}, Val {len(val_idx)}")
        
        return cv_loaders
    
    def load_from_csv(self, file_path: str, target_column: str, 
                     feature_columns: Optional[List[str]] = None) -> SecurityDataset:
        """
        Load data from CSV file.
        
        Args:
            file_path: Path to CSV file
            target_column: Name of target column
            feature_columns: List of feature column names (optional)
            
        Returns:
            SecurityDataset instance
        """
        df = pd.read_csv(file_path)
        
        if feature_columns is None:
            feature_columns = [col for col in df.columns if col != target_column]
        
        X = df[feature_columns].values
        y = df[target_column].values
        
        return SecurityDataset(X, y)
    
    def load_from_json(self, file_path: str, data_key: str = 'data', 
                      label_key: str = 'labels') -> SecurityDataset:
        """
        Load data from JSON file.
        
        Args:
            file_path: Path to JSON file
            data_key: Key for data in JSON
            label_key: Key for labels in JSON
            
        Returns:
            SecurityDataset instance
        """
        with open(file_path, 'r') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            data_dict = json.load(f)
        
        X = np.array(data_dict[data_key])
        y = np.array(data_dict[label_key])
        
        return SecurityDataset(X, y)
    
    def create_text_dataset(self, texts: List[str], labels: List[int], 
                           tokenizer, max_length: int = 512) -> TextSecurityDataset:
        """
        Create text dataset for transformer models.
        
        Args:
            texts: List of text samples
            labels: List of labels
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
            
        Returns:
            TextSecurityDataset instance
        """
        return TextSecurityDataset(texts, labels, tokenizer, max_length)

class DataTransforms:
    """Collection of data transformation functions."""
    
    @staticmethod
    def normalize_features(mean: np.ndarray, std: np.ndarray):
        """Create normalization transform."""
        def transform(x: torch.Tensor) -> torch.Tensor:
            return (x - mean) / std
        return transform
    
    @staticmethod
    def add_noise(noise_factor: float = 0.1):
        """Create noise addition transform."""
        def transform(x: torch.Tensor) -> torch.Tensor:
            noise = torch.randn_like(x) * noise_factor
            return x + noise
        return transform
    
    @staticmethod
    def random_mask(mask_prob: float = 0.1):
        """Create random masking transform."""
        def transform(x: torch.Tensor) -> torch.Tensor:
            mask = torch.rand_like(x) > mask_prob
            return x * mask
        return transform

def get_data_statistics(dataset: Dataset) -> Dict[str, Any]:
    """
    Calculate dataset statistics.
    
    Args:
        dataset: PyTorch dataset
        
    Returns:
        Dictionary with dataset statistics
    """
    if isinstance(dataset, SecurityDataset):
        data = dataset.data
        labels = dataset.labels
        
        stats = {
            'num_samples': len(dataset),
            'num_features': data.shape[1],
            'num_classes': len(torch.unique(labels)),
            'class_distribution': torch.bincount(labels).tolist(),
            'feature_mean': data.mean(dim=0).tolist(),
            'feature_std': data.std(dim=0).tolist(),
            'feature_min': data.min(dim=0)[0].tolist(),
            'feature_max': data.max(dim=0)[0].tolist()
        }
    else:
        stats = {
            'num_samples': len(dataset),
            'dataset_type': type(dataset).__name__
        }
    
    return stats

def save_data_config(config: DataConfig, file_path: str) -> None:
    """Save data configuration to file."""
    config_dict = {
        'batch_size': config.batch_size,
        'num_workers': config.num_workers,
        'pin_memory': config.pin_memory,
        'shuffle': config.shuffle,
        'drop_last': config.drop_last,
        'train_split': config.train_split,
        'val_split': config.val_split,
        'test_split': config.test_split,
        'random_seed': config.random_seed,
        'cross_validation_folds': config.cross_validation_folds,
        'stratified_splitting': config.stratified_splitting
    }
    
    with open(file_path, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        json.dump(config_dict, f, indent=2)

def load_data_config(file_path: str) -> DataConfig:
    """Load data configuration from file."""
    with open(file_path, 'r') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        config_dict = json.load(f)
    
    return DataConfig(**config_dict) 