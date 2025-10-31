from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import os
import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from dataclasses import dataclass, field
import random
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import av
from tqdm import tqdm
from typing import Any, List, Dict, Optional
import asyncio
"""
AI Video Data Loading Module
============================

This module provides a modular structure for data loading, preprocessing,
and dataset management for AI video generation.

Features:
- Custom dataset classes for video data
- Data preprocessing and augmentation
- Efficient data loading with caching
- Support for various video formats
- Batch processing and memory optimization
"""



# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""
    
    # Data paths
    data_dir: str
    metadata_file: Optional[str] = None
    
    # Video parameters
    frame_size: Tuple[int, int] = (256, 256)
    num_frames: int = 16
    fps: int = 30
    channels: int = 3
    
    # Data loading parameters
    batch_size: int = 8
    num_workers: int = 4
    pin_memory: bool = True
    shuffle: bool = True
    
    # Preprocessing parameters
    normalize: bool = True
    augment: bool = True
    cache_data: bool = False
    max_cache_size: int = 1000
    
    # Split parameters
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'data_dir': self.data_dir,
            'metadata_file': self.metadata_file,
            'frame_size': self.frame_size,
            'num_frames': self.num_frames,
            'fps': self.fps,
            'channels': self.channels,
            'batch_size': self.batch_size,
            'num_workers': self.num_workers,
            'pin_memory': self.pin_memory,
            'shuffle': self.shuffle,
            'normalize': self.normalize,
            'augment': self.augment,
            'cache_data': self.cache_data,
            'max_cache_size': self.max_cache_size,
            'train_split': self.train_split,
            'val_split': self.val_split,
            'test_split': self.test_split
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'DataConfig':
        """Create config from dictionary."""
        return cls(**config_dict)
    
    def save(self, filepath: str) -> None:
        """Save config to JSON file."""
        with open(filepath, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'DataConfig':
        """Load config from JSON file."""
        with open(filepath, 'r') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


class VideoTransform:
    """Video preprocessing and augmentation transforms."""
    
    def __init__(self, config: DataConfig):
        
    """__init__ function."""
self.config = config
        self.transforms = self._build_transforms()
    
    def _build_transforms(self) -> transforms.Compose:
        """Build transform pipeline."""
        transform_list = []
        
        # Resize frames
        transform_list.append(transforms.Resize(self.config.frame_size))
        
        # Convert to tensor
        transform_list.append(transforms.ToTensor())
        
        # Normalize if enabled
        if self.config.normalize:
            transform_list.append(transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ))
        
        return transforms.Compose(transform_list)
    
    def __call__(self, video: np.ndarray) -> torch.Tensor:
        """Apply transforms to video."""
        # video shape: (frames, height, width, channels)
        transformed_frames = []
        
        for frame in video:
            # Convert to PIL Image
            frame_pil = Image.fromarray(frame.astype(np.uint8))
            
            # Apply transforms
            frame_tensor = self.transforms(frame_pil)
            transformed_frames.append(frame_tensor)
        
        # Stack frames
        video_tensor = torch.stack(transformed_frames, dim=0)
        
        return video_tensor


class VideoAugmentation:
    """Video augmentation techniques."""
    
    def __init__(self, config: DataConfig):
        
    """__init__ function."""
self.config = config
    
    def temporal_augmentation(self, video: torch.Tensor) -> torch.Tensor:
        """Apply temporal augmentation (frame sampling, speed changes)."""
        if not self.config.augment:
            return video
        
        # Random temporal sampling
        if random.random() < 0.5:
            video = self._temporal_sampling(video)
        
        # Random speed change
        if random.random() < 0.3:
            video = self._speed_change(video)
        
        return video
    
    def spatial_augmentation(self, video: torch.Tensor) -> torch.Tensor:
        """Apply spatial augmentation (crop, flip, rotation)."""
        if not self.config.augment:
            return video
        
        # Random horizontal flip
        if random.random() < 0.5:
            video = torch.flip(video, dims=[3])  # Flip width dimension
        
        # Random crop
        if random.random() < 0.3:
            video = self._random_crop(video)
        
        # Random rotation
        if random.random() < 0.2:
            video = self._random_rotation(video)
        
        return video
    
    def _temporal_sampling(self, video: torch.Tensor) -> torch.Tensor:
        """Random temporal sampling."""
        frames, channels, height, width = video.shape
        num_frames = self.config.num_frames
        
        if frames > num_frames:
            # Random start index
            start_idx = random.randint(0, frames - num_frames)
            video = video[start_idx:start_idx + num_frames]
        elif frames < num_frames:
            # Repeat frames to reach desired length
            repeat_times = (num_frames + frames - 1) // frames
            video = video.repeat(repeat_times, 1, 1, 1)
            video = video[:num_frames]
        
        return video
    
    def _speed_change(self, video: torch.Tensor) -> torch.Tensor:
        """Random speed change."""
        frames = video.shape[0]
        speed_factor = random.uniform(0.5, 2.0)
        
        new_frames = int(frames * speed_factor)
        indices = torch.linspace(0, frames - 1, new_frames).long()
        
        return video[indices]
    
    def _random_crop(self, video: torch.Tensor) -> torch.Tensor:
        """Random spatial crop."""
        frames, channels, height, width = video.shape
        crop_size = min(height, width) // 2
        
        top = random.randint(0, height - crop_size)
        left = random.randint(0, width - crop_size)
        
        return video[:, :, top:top + crop_size, left:left + crop_size]
    
    def _random_rotation(self, video: torch.Tensor) -> torch.Tensor:
        """Random rotation."""
        angle = random.uniform(-15, 15)
        
        # Apply rotation to each frame
        rotated_frames = []
        for frame in video:
            rotated_frame = torch.rot90(frame, k=int(angle // 90))
            rotated_frames.append(rotated_frame)
        
        return torch.stack(rotated_frames, dim=0)


class BaseVideoDataset(ABC, Dataset):
    """Base class for video datasets."""
    
    def __init__(self, config: DataConfig, split: str = "train"):
        
    """__init__ function."""
super().__init__()
        self.config = config
        self.split = split
        
        # Initialize components
        self.transform = VideoTransform(config)
        self.augmentation = VideoAugmentation(config)
        
        # Load data
        self.data = self._load_data()
        
        logger.info(f"Loaded {len(self.data)} samples for {split} split")
    
    @abstractmethod
    def _load_data(self) -> List[Dict[str, Any]]:
        """Load dataset. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def _load_video(self, video_path: str) -> np.ndarray:
        """Load video from path. Must be implemented by subclasses."""
        pass
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get item by index."""
        sample = self.data[idx]
        
        # Load video
        video = self._load_video(sample['video_path'])
        
        # Apply transforms
        video_tensor = self.transform(video)
        
        # Apply augmentations for training
        if self.split == "train":
            video_tensor = self.augmentation.temporal_augmentation(video_tensor)
            video_tensor = self.augmentation.spatial_augmentation(video_tensor)
        
        # Prepare output
        output = {
            'video': video_tensor,
            'video_path': sample['video_path']
        }
        
        # Add metadata if available
        if 'prompt' in sample:
            output['prompt'] = sample['prompt']
        if 'label' in sample:
            output['label'] = sample['label']
        if 'metadata' in sample:
            output['metadata'] = sample['metadata']
        
        return output


class VideoFileDataset(BaseVideoDataset):
    """Dataset for video files with metadata."""
    
    def _load_data(self) -> List[Dict[str, Any]]:
        """Load video file paths and metadata."""
        data_dir = Path(self.config.data_dir)
        data = []
        
        # Load metadata if available
        metadata = None
        if self.config.metadata_file and Path(self.config.metadata_file).exists():
            metadata = pd.read_csv(self.config.metadata_file)
            logger.info(f"Loaded metadata with {len(metadata)} entries")
        
        # Find video files
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
        video_files = []
        
        for ext in video_extensions:
            video_files.extend(data_dir.rglob(f"*{ext}"))
        
        logger.info(f"Found {len(video_files)} video files")
        
        # Create data entries
        for video_path in video_files:
            entry = {'video_path': str(video_path)}
            
            # Add metadata if available
            if metadata is not None:
                # Try to match by filename
                filename = video_path.stem
                meta_row = metadata[metadata['filename'] == filename]
                if not meta_row.empty:
                    entry.update(meta_row.iloc[0].to_dict())
            
            data.append(entry)
        
        return data
    
    def _load_video(self, video_path: str) -> np.ndarray:
        """Load video using OpenCV."""
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        try:
            while True:
                ret, frame = cap.read()
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                if not ret:
                    break
                
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
                
                # Limit number of frames
                if len(frames) >= self.config.num_frames * 2:
                    break
        finally:
            cap.release()
        
        if not frames:
            raise ValueError(f"No frames found in video: {video_path}")
        
        # Convert to numpy array
        video = np.array(frames)
        
        # Ensure correct number of frames
        if len(video) > self.config.num_frames:
            # Sample frames evenly
            indices = np.linspace(0, len(video) - 1, self.config.num_frames, dtype=int)
            video = video[indices]
        elif len(video) < self.config.num_frames:
            # Repeat last frame to reach desired length
            last_frame = video[-1:]
            repeat_times = self.config.num_frames - len(video)
            video = np.concatenate([video, np.repeat(last_frame, repeat_times, axis=0)])
        
        return video


class CachedVideoDataset(BaseVideoDataset):
    """Dataset with video caching for faster loading."""
    
    def __init__(self, config: DataConfig, split: str = "train"):
        
    """__init__ function."""
self.cache = {}
        super().__init__(config, split)
    
    def _load_video(self, video_path: str) -> np.ndarray:
        """Load video with caching."""
        if video_path in self.cache:
            return self.cache[video_path]
        
        # Load video
        video = self._load_video_from_file(video_path)
        
        # Cache if enabled and cache not full
        if self.config.cache_data and len(self.cache) < self.config.max_cache_size:
            self.cache[video_path] = video
        
        return video
    
    def _load_video_from_file(self, video_path: str) -> np.ndarray:
        """Load video from file (same as VideoFileDataset)."""
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        try:
            while True:
                ret, frame = cap.read()
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                if not ret:
                    break
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
                
                if len(frames) >= self.config.num_frames * 2:
                    break
        finally:
            cap.release()
        
        if not frames:
            raise ValueError(f"No frames found in video: {video_path}")
        
        video = np.array(frames)
        
        if len(video) > self.config.num_frames:
            indices = np.linspace(0, len(video) - 1, self.config.num_frames, dtype=int)
            video = video[indices]
        elif len(video) < self.config.num_frames:
            last_frame = video[-1:]
            repeat_times = self.config.num_frames - len(video)
            video = np.concatenate([video, np.repeat(last_frame, repeat_times, axis=0)])
        
        return video


class DataLoaderFactory:
    """Factory class for creating data loaders."""
    
    _datasets = {
        'video_file': VideoFileDataset,
        'cached_video': CachedVideoDataset
    }
    
    @classmethod
    def create_dataset(cls, dataset_type: str, config: DataConfig, split: str = "train") -> BaseVideoDataset:
        """Create a dataset instance."""
        if dataset_type not in cls._datasets:
            raise ValueError(f"Unknown dataset type: {dataset_type}. Available: {list(cls._datasets.keys())}")
        
        dataset_class = cls._datasets[dataset_type]
        return dataset_class(config, split)
    
    @classmethod
    def create_data_loader(cls, dataset_type: str, config: DataConfig, split: str = "train") -> DataLoader:
        """Create a data loader instance."""
        dataset = cls.create_dataset(dataset_type, config, split)
        
        return DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=config.shuffle if split == "train" else False,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            drop_last=split == "train"
        )
    
    @classmethod
    def create_train_val_test_loaders(cls, dataset_type: str, config: DataConfig) -> Dict[str, DataLoader]:
        """Create train, validation, and test data loaders."""
        # Create full dataset
        full_dataset = cls.create_dataset(dataset_type, config, "full")
        
        # Calculate split sizes
        total_size = len(full_dataset)
        train_size = int(total_size * config.train_split)
        val_size = int(total_size * config.val_split)
        test_size = total_size - train_size - val_size
        
        # Split dataset
        train_dataset, val_dataset, test_dataset = random_split(
            full_dataset, [train_size, val_size, test_size]
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            drop_last=False
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            drop_last=False
        )
        
        return {
            'train': train_loader,
            'val': val_loader,
            'test': test_loader
        }
    
    @classmethod
    def get_available_datasets(cls) -> List[str]:
        """Get list of available dataset types."""
        return list(cls._datasets.keys())
    
    @classmethod
    def register_dataset(cls, name: str, dataset_class: type) -> None:
        """Register a new dataset type."""
        if not issubclass(dataset_class, BaseVideoDataset):
            raise ValueError(f"Dataset class must inherit from BaseVideoDataset")
        cls._datasets[name] = dataset_class


# Convenience functions
def create_dataset(dataset_type: str, config: DataConfig, split: str = "train") -> BaseVideoDataset:
    """Create a dataset instance."""
    return DataLoaderFactory.create_dataset(dataset_type, config, split)


def create_data_loader(dataset_type: str, config: DataConfig, split: str = "train") -> DataLoader:
    """Create a data loader instance."""
    return DataLoaderFactory.create_data_loader(dataset_type, config, split)


def create_train_val_test_loaders(dataset_type: str, config: DataConfig) -> Dict[str, DataLoader]:
    """Create train, validation, and test data loaders."""
    return DataLoaderFactory.create_train_val_test_loaders(dataset_type, config)


def get_dataset_info(dataset: BaseVideoDataset) -> Dict[str, Any]:
    """Get information about a dataset."""
    return {
        'dataset_type': dataset.__class__.__name__,
        'split': dataset.split,
        'size': len(dataset),
        'config': dataset.config.to_dict()
    }


if __name__ == "__main__":
    # Example usage
    config = DataConfig(
        data_dir="data/videos",
        frame_size=(64, 64),
        num_frames=8,
        batch_size=4
    )
    
    # Create data loaders
    loaders = create_train_val_test_loaders("video_file", config)
    
    print(f"Train batches: {len(loaders['train'])}")
    print(f"Val batches: {len(loaders['val'])}")
    print(f"Test batches: {len(loaders['test'])}")
    
    # Test data loading
    for batch in loaders['train']:
        print(f"Batch shape: {batch['video'].shape}")
        break 