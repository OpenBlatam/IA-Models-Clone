from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
BUFFER_SIZE = 1024

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Sampler, BatchSampler
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
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import warnings
from tqdm import tqdm
import h5py
import cv2
from PIL import Image
import albumentations as A
from torchvision import transforms
import psutil
import gc
import shutil
from typing import Any, List, Dict, Optional
import asyncio
#!/usr/bin/env python3
"""
Efficient Data Loading System
Advanced data loading implementation using PyTorch's DataLoader with optimizations.
"""



class DataType(Enum):
    """Types of data."""
    IMAGE = "image"
    TEXT = "text"
    AUDIO = "audio"
    VIDEO = "video"
    TABULAR = "tabular"
    MULTIMODAL = "multimodal"


class LoadingMode(Enum):
    """Data loading modes."""
    LAZY = "lazy"  # Load on demand
    EAGER = "eager"  # Pre-load all data
    STREAMING = "streaming"  # Stream from disk
    CACHED = "cached"  # Cache in memory


@dataclass
class DataLoaderConfig:
    """Configuration for efficient data loading."""
    # Basic settings
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2
    
    # Advanced settings
    drop_last: bool = False
    shuffle: bool = True
    collate_fn: Optional[Callable] = None
    
    # Memory optimization
    memory_efficient: bool = False
    max_memory_usage: float = 0.8  # Maximum memory usage (0.0-1.0)
    
    # Performance settings
    pin_memory_device: str = "cuda"
    non_blocking: bool = True
    generator_seed: Optional[int] = None
    
    # Custom settings
    custom_sampler: Optional[Sampler] = None
    custom_batch_sampler: Optional[BatchSampler] = None
    
    # Monitoring
    enable_monitoring: bool = True
    log_loading_stats: bool = True


class BaseDataset(Dataset):
    """Base dataset class with common functionality."""
    
    def __init__(self, data_path: str, transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 loading_mode: LoadingMode = LoadingMode.LAZY):
        self.data_path = Path(data_path)
        self.transform = transform
        self.target_transform = target_transform
        self.loading_mode = loading_mode
        self.logger = self._setup_logging()
        
        # Data storage
        self.data = None
        self.targets = None
        self.data_indices = []
        
        # Load data based on mode
        self._load_data()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(self.__class__.__name__)
    
    def _load_data(self) -> Any:
        """Load data based on loading mode."""
        if self.loading_mode == LoadingMode.EAGER:
            self._load_all_data()
        elif self.loading_mode == LoadingMode.LAZY:
            self._setup_lazy_loading()
        elif self.loading_mode == LoadingMode.STREAMING:
            self._setup_streaming()
        elif self.loading_mode == LoadingMode.CACHED:
            self._setup_cached_loading()
    
    def _load_all_data(self) -> Any:
        """Load all data into memory."""
        self.logger.info("Loading all data into memory...")
        # Implementation depends on data format
        pass
    
    def _setup_lazy_loading(self) -> Any:
        """Setup lazy loading (load on demand)."""
        self.logger.info("Setting up lazy loading...")
        # Implementation depends on data format
        pass
    
    def _setup_streaming(self) -> Any:
        """Setup streaming from disk."""
        self.logger.info("Setting up streaming loading...")
        # Implementation depends on data format
        pass
    
    def _setup_cached_loading(self) -> Any:
        """Setup cached loading with memory management."""
        self.logger.info("Setting up cached loading...")
        # Implementation depends on data format
        pass
    
    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.data_indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get item by index."""
        raise NotImplementedError("Subclasses must implement __getitem__")


class ImageDataset(BaseDataset):
    """Efficient image dataset with various loading strategies."""
    
    def __init__(self, data_path: str, image_size: Tuple[int, int] = (224, 224),
                 transform: Optional[Callable] = None,
                 loading_mode: LoadingMode = LoadingMode.LAZY,
                 cache_size: int = 1000):
        self.image_size = image_size
        self.cache_size = cache_size
        self.image_cache = {}
        
        # Default transform if none provided
        if transform is None:
            transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
        
        super().__init__(data_path, transform, loading_mode=loading_mode)
    
    def _setup_lazy_loading(self) -> Any:
        """Setup lazy loading for images."""
        # Find all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        self.data_indices = []
        
        for ext in image_extensions:
            self.data_indices.extend(list(self.data_path.glob(f"**/*{ext}")))
            self.data_indices.extend(list(self.data_path.glob(f"**/*{ext.upper()}")))
        
        self.logger.info(f"Found {len(self.data_indices)} images")
    
    def _load_image(self, image_path: Path) -> torch.Tensor:
        """Load and preprocess image."""
        try:
            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image
        except Exception as e:
            self.logger.warning(f"Error loading image {image_path}: {e}")
            # Return a placeholder image
            return torch.zeros(3, *self.image_size)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get image by index."""
        image_path = self.data_indices[idx]
        
        # Check cache first
        if image_path in self.image_cache:
            image = self.image_cache[image_path]
        else:
            # Load image
            image = self._load_image(image_path)
            
            # Cache if cache not full
            if len(self.image_cache) < self.cache_size:
                self.image_cache[image_path] = image
        
        # For now, return dummy target (in practice, you'd load from annotations)
        target = torch.tensor(0, dtype=torch.long)
        
        return image, target


class TextDataset(BaseDataset):
    """Efficient text dataset with tokenization."""
    
    def __init__(self, data_path: str, tokenizer: Optional[Callable] = None,
                 max_length: int = 512, loading_mode: LoadingMode = LoadingMode.LAZY):
        self.tokenizer = tokenizer
        self.max_length = max_length
        super().__init__(data_path, loading_mode=loading_mode)
    
    def _setup_lazy_loading(self) -> Any:
        """Setup lazy loading for text files."""
        # Find all text files
        text_extensions = {'.txt', '.json', '.csv'}
        self.data_indices = []
        
        for ext in text_extensions:
            self.data_indices.extend(list(self.data_path.glob(f"**/*{ext}")))
        
        self.logger.info(f"Found {len(self.data_indices)} text files")
    
    def _load_text(self, text_path: Path) -> str:
        """Load text from file."""
        try:
            with open(text_path, 'r', encoding='utf-8') as f:
                text = f.read()
            return text
        except Exception as e:
            self.logger.warning(f"Error loading text {text_path}: {e}")
            return ""
    
    def _tokenize_text(self, text: str) -> torch.Tensor:
        """Tokenize text."""
        if self.tokenizer:
            return self.tokenizer(text, max_length=self.max_length, truncation=True, 
                                padding='max_length', return_tensors='pt')
        else:
            # Simple character-level tokenization
            tokens = [ord(c) % 1000 for c in text[:self.max_length]]
            tokens = tokens + [0] * (self.max_length - len(tokens))
            return torch.tensor(tokens, dtype=torch.long)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get text by index."""
        text_path = self.data_indices[idx]
        text = self._load_text(text_path)
        tokens = self._tokenize_text(text)
        
        # For now, return dummy target
        target = torch.tensor(0, dtype=torch.long)
        
        return tokens, target


class HDF5Dataset(BaseDataset):
    """Efficient HDF5 dataset for large datasets."""
    
    def __init__(self, data_path: str, dataset_name: str = "data",
                 loading_mode: LoadingMode = LoadingMode.LAZY):
        self.dataset_name = dataset_name
        self.h5_file = None
        super().__init__(data_path, loading_mode=loading_mode)
    
    def _setup_lazy_loading(self) -> Any:
        """Setup lazy loading for HDF5 files."""
        # Open HDF5 file
        self.h5_file = h5py.File(self.data_path, 'r')
        self.data_indices = list(range(len(self.h5_file[self.dataset_name])))
        
        self.logger.info(f"Loaded HDF5 dataset with {len(self.data_indices)} samples")
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get item from HDF5 dataset."""
        if self.h5_file is None:
            raise RuntimeError("HDF5 file not opened")
        
        # Load data from HDF5
        data = self.h5_file[self.dataset_name][idx]
        data = torch.from_numpy(data).float()
        
        # For now, return dummy target
        target = torch.tensor(0, dtype=torch.long)
        
        return data, target
    
    def __del__(self) -> Any:
        """Close HDF5 file."""
        if self.h5_file is not None:
            self.h5_file.close()


class MemoryEfficientDataset(BaseDataset):
    """Memory-efficient dataset with dynamic loading."""
    
    def __init__(self, data_path: str, max_memory_usage: float = 0.8,
                 loading_mode: LoadingMode = LoadingMode.LAZY):
        self.max_memory_usage = max_memory_usage
        self.memory_monitor = MemoryMonitor()
        super().__init__(data_path, loading_mode=loading_mode)
    
    def _setup_cached_loading(self) -> Any:
        """Setup cached loading with memory management."""
        self.logger.info("Setting up memory-efficient cached loading...")
        
        # Monitor memory usage
        self.memory_monitor.start_monitoring()
        
        # Load data with memory constraints
        self._load_with_memory_constraints()
    
    def _load_with_memory_constraints(self) -> Any:
        """Load data while respecting memory constraints."""
        available_memory = psutil.virtual_memory().available
        max_allowed_memory = available_memory * self.max_memory_usage
        
        self.logger.info(f"Available memory: {available_memory / 1024**3:.2f} GB")
        self.logger.info(f"Max allowed memory: {max_allowed_memory / 1024**3:.2f} GB")
        
        # Implementation depends on data format
        pass


class MemoryMonitor:
    """Monitor memory usage during data loading."""
    
    def __init__(self) -> Any:
        self.memory_usage = []
        self.monitoring = False
    
    def start_monitoring(self) -> Any:
        """Start memory monitoring."""
        self.monitoring = True
        self.memory_usage = []
    
    def update(self) -> Any:
        """Update memory usage."""
        if self.monitoring:
            memory = psutil.virtual_memory()
            self.memory_usage.append({
                'timestamp': time.time(),
                'used': memory.used,
                'available': memory.available,
                'percent': memory.percent
            })
    
    def get_memory_stats(self) -> Dict[str, float]:
        """Get current memory statistics."""
        memory = psutil.virtual_memory()
        return {
            'used_gb': memory.used / 1024**3,
            'available_gb': memory.available / 1024**3,
            'percent': memory.percent
        }
    
    def check_memory_constraint(self, required_memory: int) -> bool:
        """Check if required memory is available."""
        available_memory = psutil.virtual_memory().available
        return available_memory >= required_memory


class EfficientDataLoader:
    """Efficient data loader with advanced features."""
    
    def __init__(self, dataset: Dataset, config: DataLoaderConfig):
        self.dataset = dataset
        self.config = config
        self.logger = self._setup_logging()
        
        # Create DataLoader
        self.dataloader = self._create_dataloader()
        
        # Performance monitoring
        self.performance_monitor = DataLoaderPerformanceMonitor()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def _create_dataloader(self) -> DataLoader:
        """Create optimized DataLoader."""
        # Determine optimal number of workers
        num_workers = self._get_optimal_workers()
        
        # Create sampler if needed
        sampler = self._create_sampler()
        
        # Create DataLoader
        dataloader = DataLoader(
            dataset=self.dataset,
            batch_size=self.config.batch_size,
            shuffle=self.config.shuffle if sampler is None else False,
            num_workers=num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=self.config.persistent_workers,
            prefetch_factor=self.config.prefetch_factor,
            drop_last=self.config.drop_last,
            collate_fn=self.config.collate_fn,
            sampler=sampler,
            batch_sampler=self.config.custom_batch_sampler,
            generator=self._create_generator()
        )
        
        self.logger.info(f"Created DataLoader with {num_workers} workers")
        return dataloader
    
    def _get_optimal_workers(self) -> int:
        """Get optimal number of workers."""
        if self.config.num_workers > 0:
            return self.config.num_workers
        
        # Auto-detect optimal number of workers
        cpu_count = mp.cpu_count()
        
        # Use 75% of CPU cores for data loading
        optimal_workers = max(1, int(cpu_count * 0.75))
        
        # Limit based on memory constraints
        memory_stats = psutil.virtual_memory()
        if memory_stats.available < 4 * 1024**3:  # Less than 4GB available
            optimal_workers = min(optimal_workers, 2)
        
        return optimal_workers
    
    def _create_sampler(self) -> Optional[Sampler]:
        """Create custom sampler if needed."""
        if self.config.custom_sampler:
            return self.config.custom_sampler
        
        # Could implement weighted sampling, etc.
        return None
    
    def _create_generator(self) -> Optional[torch.Generator]:
        """Create random generator if seed provided."""
        if self.config.generator_seed is not None:
            generator = torch.Generator()
            generator.manual_seed(self.config.generator_seed)
            return generator
        return None
    
    def __iter__(self) -> Iterator:
        """Iterate over DataLoader with monitoring."""
        if self.config.enable_monitoring:
            self.performance_monitor.start_iteration()
        
        for batch in self.dataloader:
            if self.config.enable_monitoring:
                self.performance_monitor.update_batch(batch)
            
            # Move to device if pin_memory is enabled
            if self.config.pin_memory and torch.cuda.is_available():
                batch = self._move_to_device(batch)
            
            yield batch
    
    def _move_to_device(self, batch: Union[torch.Tensor, Tuple, List]) -> Union[torch.Tensor, Tuple, List]:
        """Move batch to device."""
        device = torch.device(self.config.pin_memory_device)
        
        if isinstance(batch, torch.Tensor):
            return batch.to(device, non_blocking=self.config.non_blocking)
        elif isinstance(batch, (tuple, list)):
            return type(batch)(self._move_to_device(item) for item in batch)
        else:
            return batch
    
    def __len__(self) -> int:
        """Return number of batches."""
        return len(self.dataloader)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get DataLoader statistics."""
        stats = {
            'dataset_size': len(self.dataset),
            'num_batches': len(self.dataloader),
            'batch_size': self.config.batch_size,
            'num_workers': self.dataloader.num_workers,
            'pin_memory': self.config.pin_memory,
            'persistent_workers': self.config.persistent_workers
        }
        
        if self.config.enable_monitoring:
            stats.update(self.performance_monitor.get_stats())
        
        return stats


class DataLoaderPerformanceMonitor:
    """Monitor DataLoader performance."""
    
    def __init__(self) -> Any:
        self.iteration_start_time = None
        self.batch_times = []
        self.memory_usage = []
        self.cpu_usage = []
    
    def start_iteration(self) -> Any:
        """Start monitoring iteration."""
        self.iteration_start_time = time.time()
        self.batch_times = []
        self.memory_usage = []
        self.cpu_usage = []
    
    def update_batch(self, batch) -> Any:
        """Update batch statistics."""
        batch_time = time.time()
        self.batch_times.append(batch_time)
        
        # Monitor memory and CPU
        memory = psutil.virtual_memory()
        cpu = psutil.cpu_percent()
        
        self.memory_usage.append(memory.percent)
        self.cpu_usage.append(cpu)
    
    def get_stats(self) -> Dict[str, float]:
        """Get performance statistics."""
        if not self.batch_times:
            return {}
        
        total_time = time.time() - self.iteration_start_time
        avg_batch_time = np.mean(np.diff(self.batch_times)) if len(self.batch_times) > 1 else 0
        
        return {
            'total_time': total_time,
            'avg_batch_time': avg_batch_time,
            'batches_per_second': len(self.batch_times) / total_time if total_time > 0 else 0,
            'avg_memory_usage': np.mean(self.memory_usage) if self.memory_usage else 0,
            'avg_cpu_usage': np.mean(self.cpu_usage) if self.cpu_usage else 0
        }


class CustomCollateFn:
    """Custom collate function for different data types."""
    
    @staticmethod
    def image_collate(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Collate image batch."""
        images, targets = zip(*batch)
        return torch.stack(images), torch.stack(targets)
    
    @staticmethod
    def text_collate(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Collate text batch."""
        texts, targets = zip(*batch)
        return torch.stack(texts), torch.stack(targets)
    
    @staticmethod
    def variable_length_collate(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """Collate variable length sequences."""
        texts, targets = zip(*batch)
        return list(texts), torch.stack(targets)


class DataLoaderFactory:
    """Factory for creating efficient DataLoaders."""
    
    @staticmethod
    def create_dataloader(dataset: Dataset, config: DataLoaderConfig) -> EfficientDataLoader:
        """Create efficient DataLoader."""
        return EfficientDataLoader(dataset, config)
    
    @staticmethod
    def create_image_dataloader(data_path: str, config: DataLoaderConfig,
                               image_size: Tuple[int, int] = (224, 224)) -> EfficientDataLoader:
        """Create image DataLoader."""
        dataset = ImageDataset(data_path, image_size=image_size)
        return EfficientDataLoader(dataset, config)
    
    @staticmethod
    def create_text_dataloader(data_path: str, config: DataLoaderConfig,
                              max_length: int = 512) -> EfficientDataLoader:
        """Create text DataLoader."""
        dataset = TextDataset(data_path, max_length=max_length)
        return EfficientDataLoader(dataset, config)
    
    @staticmethod
    def create_hdf5_dataloader(data_path: str, config: DataLoaderConfig,
                               dataset_name: str = "data") -> EfficientDataLoader:
        """Create HDF5 DataLoader."""
        dataset = HDF5Dataset(data_path, dataset_name=dataset_name)
        return EfficientDataLoader(dataset, config)


def demonstrate_efficient_data_loading():
    """Demonstrate efficient data loading capabilities."""
    print("Efficient Data Loading Demonstration")
    print("=" * 50)
    
    # Create test data
    test_data_dir = Path("test_data")
    test_data_dir.mkdir(exist_ok=True)
    
    # Create dummy images
    for i in range(100):
        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        img_path = test_data_dir / f"image_{i:03d}.jpg"
        cv2.imwrite(str(img_path), img)
    
    # Create dummy text files
    for i in range(100):
        text = f"This is sample text {i} with some random content for testing data loading efficiency."
        text_path = test_data_dir / f"text_{i:03d}.txt"
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write(text)
    
    # Test different configurations
    configs = [
        DataLoaderConfig(
            batch_size=16,
            num_workers=2,
            pin_memory=True,
            persistent_workers=True
        ),
        DataLoaderConfig(
            batch_size=32,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2
        ),
        DataLoaderConfig(
            batch_size=64,
            num_workers=0,  # Single process
            pin_memory=False,
            persistent_workers=False
        )
    ]
    
    results = {}
    
    for i, config in enumerate(configs):
        print(f"\nTesting configuration {i+1}:")
        print(f"  Batch size: {config.batch_size}")
        print(f"  Workers: {config.num_workers}")
        print(f"  Pin memory: {config.pin_memory}")
        print(f"  Persistent workers: {config.persistent_workers}")
        
        try:
            # Create image dataset and dataloader
            dataset = ImageDataset(str(test_data_dir), image_size=(224, 224))
            dataloader = DataLoaderFactory.create_dataloader(dataset, config)
            
            # Test loading
            start_time = time.time()
            batch_count = 0
            
            for batch in dataloader:
                images, targets = batch
                batch_count += 1
                
                # Simulate some processing
                time.sleep(0.01)
                
                if batch_count >= 5:  # Test first 5 batches
                    break
            
            end_time = time.time()
            loading_time = end_time - start_time
            
            # Get stats
            stats = dataloader.get_stats()
            
            print(f"  Loading time: {loading_time:.4f}s")
            print(f"  Batches processed: {batch_count}")
            print(f"  Batches per second: {batch_count / loading_time:.2f}")
            print(f"  Memory usage: {stats.get('avg_memory_usage', 0):.1f}%")
            print(f"  CPU usage: {stats.get('avg_cpu_usage', 0):.1f}%")
            
            results[f"config_{i}"] = {
                'config': config,
                'loading_time': loading_time,
                'batch_count': batch_count,
                'batches_per_second': batch_count / loading_time,
                'stats': stats,
                'success': True
            }
            
        except Exception as e:
            print(f"  Error: {e}")
            results[f"config_{i}"] = {
                'config': config,
                'error': str(e),
                'success': False
            }
    
    # Cleanup
    shutil.rmtree(test_data_dir)
    
    return results


if __name__ == "__main__":
    # Demonstrate efficient data loading
    results = demonstrate_efficient_data_loading()
    print("\nEfficient data loading demonstration completed!") 