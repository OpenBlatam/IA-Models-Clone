from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import torch
import torch.utils.data as data
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging
import json
from pathlib import Path
import os
from PIL import Image
import pickle
            import matplotlib.pyplot as plt
        from PIL import Image
from typing import Any, List, Dict, Optional
import asyncio
#!/usr/bin/env python3
"""
Base Dataset Class
Foundation class for all dataset implementations in the modular deep learning system.
"""


logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    """Base configuration for datasets."""
    # Dataset parameters
    dataset_name: str = "base_dataset"
    dataset_type: str = "base"
    data_path: str = ""
    split_ratio: List[float] = field(default_factory=lambda: [0.7, 0.15, 0.15])  # train, val, test
    
    # Data parameters
    input_size: Tuple[int, ...] = (224, 224)
    num_classes: int = 10
    num_channels: int = 3
    
    # Processing parameters
    normalize: bool = True
    augmentation: bool = False
    cache_data: bool = False
    shuffle: bool = True
    
    # Loading parameters
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    drop_last: bool = False
    
    # Augmentation parameters
    horizontal_flip: bool = True
    vertical_flip: bool = False
    rotation: float = 0.0
    brightness: float = 0.0
    contrast: float = 0.0
    saturation: float = 0.0
    hue: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'dataset_name': self.dataset_name,
            'dataset_type': self.dataset_type,
            'data_path': self.data_path,
            'split_ratio': self.split_ratio,
            'input_size': self.input_size,
            'num_classes': self.num_classes,
            'num_channels': self.num_channels,
            'normalize': self.normalize,
            'augmentation': self.augmentation,
            'cache_data': self.cache_data,
            'shuffle': self.shuffle,
            'batch_size': self.batch_size,
            'num_workers': self.num_workers,
            'pin_memory': self.pin_memory,
            'drop_last': self.drop_last,
            'horizontal_flip': self.horizontal_flip,
            'vertical_flip': self.vertical_flip,
            'rotation': self.rotation,
            'brightness': self.brightness,
            'contrast': self.contrast,
            'saturation': self.saturation,
            'hue': self.hue
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'DatasetConfig':
        """Create config from dictionary."""
        return cls(**config_dict)
    
    def save(self, filepath: str):
        """Save config to file."""
        with open(filepath, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'DatasetConfig':
        """Load config from file."""
        with open(filepath, 'r') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


class BaseDataset(data.Dataset, ABC):
    """Base dataset class for all data loading implementations."""
    
    def __init__(self, config: DatasetConfig, split: str = "train", transform: Optional[Callable] = None):
        
    """__init__ function."""
super().__init__()
        self.config = config
        self.split = split
        self.transform = transform
        self.data = []
        self.labels = []
        self.cached_data = {}
        
        # Validate split
        if split not in ["train", "val", "test"]:
            raise ValueError(f"Invalid split: {split}. Must be one of ['train', 'val', 'test']")
        
        # Load and prepare data
        self._load_data()
        self._prepare_data()
        
        logger.info(f"Initialized {self.config.dataset_name} ({self.split} split) with {len(self)} samples")
    
    @abstractmethod
    def _load_data(self) -> Any:
        """Load data from source. Must be implemented by subclasses."""
        pass
    
    def _prepare_data(self) -> Any:
        """Prepare data for training."""
        if self.config.cache_data:
            self._cache_data()
    
    def _cache_data(self) -> Any:
        """Cache data in memory for faster access."""
        logger.info("Caching data in memory...")
        
        for i in range(len(self.data)):
            if i not in self.cached_data:
                sample = self._load_sample(i)
                self.cached_data[i] = sample
        
        logger.info(f"Cached {len(self.cached_data)} samples")
    
    def _load_sample(self, index: int) -> Any:
        """Load a single sample. Can be overridden by subclasses."""
        return self.data[index]
    
    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.data)
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """Get a sample and its label."""
        if self.config.cache_data and index in self.cached_data:
            sample = self.cached_data[index]
        else:
            sample = self._load_sample(index)
        
        # Apply transformations
        if self.transform:
            sample = self.transform(sample)
        
        # Get label
        label = self.labels[index] if index < len(self.labels) else None
        
        return sample, label
    
    def get_sample_info(self, index: int) -> Dict[str, Any]:
        """Get information about a specific sample."""
        if index >= len(self):
            raise IndexError(f"Index {index} out of range")
        
        sample_info = {
            'index': index,
            'split': self.split,
            'dataset_name': self.config.dataset_name,
            'sample_shape': None,
            'label': self.labels[index] if index < len(self.labels) else None
        }
        
        # Get sample shape if possible
        try:
            sample = self._load_sample(index)
            if hasattr(sample, 'shape'):
                sample_info['sample_shape'] = sample.shape
            elif hasattr(sample, 'size'):
                sample_info['sample_shape'] = sample.size
        except Exception as e:
            logger.warning(f"Could not get sample shape for index {index}: {e}")
        
        return sample_info
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get information about the dataset."""
        dataset_info = {
            'dataset_name': self.config.dataset_name,
            'dataset_type': self.config.dataset_type,
            'split': self.split,
            'num_samples': len(self),
            'num_classes': self.config.num_classes,
            'input_size': self.config.input_size,
            'num_channels': self.config.num_channels,
            'config': self.config.to_dict()
        }
        
        # Add label distribution if available
        if self.labels:
            unique_labels, counts = np.unique(self.labels, return_counts=True)
            dataset_info['label_distribution'] = dict(zip(unique_labels.tolist(), counts.tolist()))
        
        return dataset_info
    
    def split_dataset(self, split_ratio: List[float] = None) -> Dict[str, 'BaseDataset']:
        """Split dataset into train/val/test sets."""
        if split_ratio is None:
            split_ratio = self.config.split_ratio
        
        if len(split_ratio) != 3 or sum(split_ratio) != 1.0:
            raise ValueError("Split ratio must have 3 values that sum to 1.0")
        
        total_samples = len(self)
        train_size = int(split_ratio[0] * total_samples)
        val_size = int(split_ratio[1] * total_samples)
        
        # Create splits
        splits = {
            'train': torch.utils.data.Subset(self, range(0, train_size)),
            'val': torch.utils.data.Subset(self, range(train_size, train_size + val_size)),
            'test': torch.utils.data.Subset(self, range(train_size + val_size, total_samples))
        }
        
        logger.info(f"Split dataset: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")
        
        return splits
    
    def save_dataset_info(self, filepath: str):
        """Save dataset information to file."""
        dataset_info = self.get_dataset_info()
        
        with open(filepath, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            json.dump(dataset_info, f, indent=2)
        
        logger.info(f"Dataset info saved to {filepath}")
    
    def get_class_weights(self) -> torch.Tensor:
        """Calculate class weights for imbalanced datasets."""
        if not self.labels:
            return torch.ones(self.config.num_classes)
        
        # Count samples per class
        class_counts = np.zeros(self.config.num_classes)
        for label in self.labels:
            if isinstance(label, (int, np.integer)):
                class_counts[label] += 1
            elif isinstance(label, torch.Tensor):
                class_counts[label.item()] += 1
        
        # Calculate weights (inverse frequency)
        total_samples = len(self.labels)
        class_weights = total_samples / (len(np.unique(self.labels)) * class_counts)
        
        # Handle zero counts
        class_weights[class_counts == 0] = 1.0
        
        return torch.FloatTensor(class_weights)
    
    def visualize_sample(self, index: int, save_path: Optional[str] = None):
        """Visualize a sample from the dataset."""
        sample, label = self[index]
        
        if isinstance(sample, torch.Tensor):
            sample = sample.numpy()
        
        if sample.ndim == 3 and sample.shape[0] in [1, 3, 4]:
            # Image data
            
            plt.figure(figsize=(8, 8))
            if sample.shape[0] == 1:
                plt.imshow(sample[0], cmap='gray')
            else:
                plt.imshow(np.transpose(sample, (1, 2, 0)))
            
            plt.title(f"Sample {index}, Label: {label}")
            plt.axis('off')
            
            if save_path:
                plt.savefig(save_path, bbox_inches='tight', dpi=150)
                plt.close()
            else:
                plt.show()
        else:
            logger.warning(f"Cannot visualize sample {index}: unsupported format")


class SimpleDataset(BaseDataset):
    """Simple dataset implementation for demonstration."""
    
    def _load_data(self) -> Any:
        """Load synthetic data for demonstration."""
        # Generate synthetic data
        num_samples = 1000
        input_size = np.prod(self.config.input_size)
        
        # Generate random data
        self.data = np.random.randn(num_samples, input_size).astype(np.float32)
        
        # Generate random labels
        self.labels = np.random.randint(0, self.config.num_classes, num_samples)
        
        logger.info(f"Loaded {num_samples} synthetic samples")


class ImageDataset(BaseDataset):
    """Base class for image datasets."""
    
    def __init__(self, config: DatasetConfig, split: str = "train", transform: Optional[Callable] = None):
        
    """__init__ function."""
super().__init__(config, split, transform)
    
    def _load_image(self, image_path: str) -> np.ndarray:
        """Load image from file."""
        try:
            image = Image.open(image_path).convert('RGB')
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            image = np.array(image)
            return image
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            # Return a black image as fallback
            return np.zeros((*self.config.input_size, self.config.num_channels), dtype=np.uint8)
    
    def _resize_image(self, image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """Resize image to target size."""
        
        pil_image = Image.fromarray(image)
        resized_image = pil_image.resize(target_size, Image.LANCZOS)
        return np.array(resized_image)
    
    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        """Normalize image to [0, 1] range."""
        if image.dtype == np.uint8:
            return image.astype(np.float32) / 255.0
        return image.astype(np.float32)


class TabularDataset(BaseDataset):
    """Base class for tabular datasets."""
    
    def __init__(self, config: DatasetConfig, split: str = "train", transform: Optional[Callable] = None):
        
    """__init__ function."""
super().__init__(config, split, transform)
    
    def _load_csv_data(self, filepath: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load data from CSV file."""
        try:
            df = pd.read_csv(filepath)
            
            # Separate features and labels
            if 'label' in df.columns:
                features = df.drop('label', axis=1).values
                labels = df['label'].values
            elif 'target' in df.columns:
                features = df.drop('target', axis=1).values
                labels = df['target'].values
            else:
                # Assume last column is the label
                features = df.iloc[:, :-1].values
                labels = df.iloc[:, -1].values
            
            return features.astype(np.float32), labels
        except Exception as e:
            logger.error(f"Error loading CSV data from {filepath}: {e}")
            return np.array([]), np.array([])
    
    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Normalize features using z-score normalization."""
        mean = np.mean(features, axis=0)
        std = np.std(features, axis=0)
        std[std == 0] = 1  # Avoid division by zero
        
        normalized_features = (features - mean) / std
        return normalized_features


class DataLoaderFactory:
    """Factory for creating data loaders."""
    
    @staticmethod
    def create_dataloader(dataset: BaseDataset, config: DatasetConfig) -> data.DataLoader:
        """Create a DataLoader for the given dataset."""
        return data.DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=config.shuffle if dataset.split == "train" else False,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            drop_last=config.drop_last
        )
    
    @staticmethod
    def create_dataloaders(datasets: Dict[str, BaseDataset], config: DatasetConfig) -> Dict[str, data.DataLoader]:
        """Create DataLoaders for multiple datasets."""
        dataloaders = {}
        
        for split_name, dataset in datasets.items():
            dataloaders[split_name] = DataLoaderFactory.create_dataloader(dataset, config)
        
        return dataloaders


# Example usage
if __name__ == "__main__":
    # Create dataset configuration
    config = DatasetConfig(
        dataset_name="test_dataset",
        dataset_type="synthetic",
        input_size=(28, 28),
        num_classes=10,
        num_channels=1,
        batch_size=32,
        num_workers=2
    )
    
    # Create dataset
    dataset = SimpleDataset(config, split="train")
    
    # Print dataset info
    print("Dataset Info:")
    print(json.dumps(dataset.get_dataset_info(), indent=2))
    
    # Create data loader
    dataloader = DataLoaderFactory.create_dataloader(dataset, config)
    
    # Test data loading
    for batch_idx, (data, labels) in enumerate(dataloader):
        print(f"Batch {batch_idx}: data shape={data.shape}, labels shape={labels.shape}")
        if batch_idx >= 2:  # Only test first few batches
            break
    
    # Test dataset splitting
    splits = dataset.split_dataset()
    print(f"\nDataset splits: {[f'{k}: {len(v)}' for k, v in splits.items()]}")
    
    # Save dataset info
    dataset.save_dataset_info("dataset_info.json")
    
    print("Dataset testing completed successfully!") 