from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
BUFFER_SIZE = 1024

import asyncio
import time
import logging
import json
import pickle
import hashlib
from typing import Dict, Any, Optional, List, Union, Tuple, Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
import warnings
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
import torch
import torch.nn as nn
from torch.utils.data import (
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import h5py
import lmdb
import msgpack
from tqdm import tqdm
import psutil
import GPUtil
from .production_transformers import DeviceManager
from typing import Any, List, Dict, Optional
"""
🚀 Efficient Data Loading System - Production Ready
==================================================

Enterprise-grade data loading system with PyTorch DataLoader optimizations,
caching, prefetching, and production features for AI training.
"""


    DataLoader, Dataset, IterableDataset, 
    random_split, WeightedRandomSampler,
    SequentialSampler, RandomSampler
)

# Import our production engines

logger = logging.getLogger(__name__)

class DataFormat(Enum):
    """Supported data formats."""
    CSV = "csv"
    JSON = "json"
    HDF5 = "hdf5"
    LMDB = "lmdb"
    PARQUET = "parquet"
    PICKLE = "pickle"
    NUMPY = "numpy"

class CacheStrategy(Enum):
    """Cache strategies."""
    NONE = "none"
    MEMORY = "memory"
    DISK = "disk"
    HYBRID = "hybrid"

@dataclass
class DataLoaderConfig:
    """DataLoader configuration."""
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2
    drop_last: bool = False
    shuffle: bool = True
    
    # Advanced optimizations
    pin_memory_device: str = "cuda"
    generator: Optional[torch.Generator] = None
    timeout: int = 0
    
    # Caching
    cache_strategy: CacheStrategy = CacheStrategy.MEMORY
    cache_dir: str = "cache"
    cache_size_gb: float = 10.0
    
    # Performance
    pin_memory_batch_size: Optional[int] = None
    non_blocking: bool = True
    
    def __post_init__(self) -> Any:
        # Auto-detect optimal number of workers
        if self.num_workers == -1:
            self.num_workers = min(mp.cpu_count(), 8)
        
        # Auto-detect optimal batch size for pin memory
        if self.pin_memory_batch_size is None:
            self.pin_memory_batch_size = self.batch_size

class CachedDataset(Dataset):
    """Dataset with intelligent caching."""
    
    def __init__(self, dataset: Dataset, cache_strategy: CacheStrategy = CacheStrategy.MEMORY,
                 cache_dir: str = "cache", cache_size_gb: float = 10.0):
        
    """__init__ function."""
self.dataset = dataset
        self.cache_strategy = cache_strategy
        self.cache_dir = Path(cache_dir)
        self.cache_size_gb = cache_size_gb
        self.cache = {}
        self.cache_metadata = {}
        
        # Setup cache
        self._setup_cache()
    
    def _setup_cache(self) -> Any:
        """Setup caching infrastructure."""
        if self.cache_strategy == CacheStrategy.NONE:
            return
        
        # Create cache directory
        self.cache_dir.mkdir(exist_ok=True)
        
        # Calculate cache hash
        self.cache_hash = self._calculate_cache_hash()
        self.cache_file = self.cache_dir / f"cache_{self.cache_hash}.pkl"
        self.metadata_file = self.cache_dir / f"metadata_{self.cache_hash}.json"
        
        # Load existing cache
        self._load_cache()
    
    def _calculate_cache_hash(self) -> str:
        """Calculate hash for cache identification."""
        # Create hash from dataset properties
        dataset_info = {
            'length': len(self.dataset),
            'type': type(self.dataset).__name__,
            'timestamp': time.time()
        }
        
        return hashlib.md5(json.dumps(dataset_info, sort_keys=True).encode()).hexdigest()[:8]
    
    def _load_cache(self) -> Any:
        """Load cache from disk."""
        if self.cache_strategy in [CacheStrategy.DISK, CacheStrategy.HYBRID]:
            if self.cache_file.exists() and self.metadata_file.exists():
                try:
                    # Load metadata
                    with open(self.metadata_file, 'r') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                        self.cache_metadata = json.load(f)
                    
                    # Load cache data
                    with open(self.cache_file, 'rb') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                        self.cache = pickle.load(f)
                    
                    logger.info(f"Loaded cache: {len(self.cache)} items")
                except Exception as e:
                    logger.warning(f"Failed to load cache: {e}")
                    self.cache = {}
                    self.cache_metadata = {}
    
    def _save_cache(self) -> Any:
        """Save cache to disk."""
        if self.cache_strategy in [CacheStrategy.DISK, CacheStrategy.HYBRID]:
            try:
                # Save metadata
                metadata = {
                    'cache_strategy': self.cache_strategy.value,
                    'cache_size': len(self.cache),
                    'timestamp': time.time()
                }
                
                with open(self.metadata_file, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                    json.dump(metadata, f)
                
                # Save cache data
                with open(self.cache_file, 'wb') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                    pickle.dump(self.cache, f)
                
                logger.info(f"Saved cache: {len(self.cache)} items")
            except Exception as e:
                logger.error(f"Failed to save cache: {e}")
    
    def __len__(self) -> Any:
        return len(self.dataset)
    
    def __getitem__(self, idx) -> Optional[Dict[str, Any]]:
        # Check cache first
        if idx in self.cache:
            return self.cache[idx]
        
        # Load from dataset
        item = self.dataset[idx]
        
        # Cache the item
        if self.cache_strategy != CacheStrategy.NONE:
            self.cache[idx] = item
            
            # Check cache size limit
            if len(self.cache) > self.cache_size_gb * 1e6:  # Rough estimate
                # Remove oldest items
                oldest_keys = sorted(self.cache.keys())[:1000]
                for key in oldest_keys:
                    del self.cache[key]
        
        return item
    
    def __del__(self) -> Any:
        """Save cache on destruction."""
        if self.cache_strategy in [CacheStrategy.DISK, CacheStrategy.HYBRID]:
            self._save_cache()

class StreamingDataset(IterableDataset):
    """Streaming dataset for large datasets."""
    
    def __init__(self, data_path: str, data_format: DataFormat, 
                 chunk_size: int = 1000, shuffle: bool = True):
        
    """__init__ function."""
self.data_path = data_path
        self.data_format = data_format
        self.chunk_size = chunk_size
        self.shuffle = shuffle
        self.worker_info = torch.utils.data.get_worker_info()
    
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate over dataset chunks."""
        worker_id = 0
        num_workers = 1
        
        if self.worker_info is not None:
            worker_id = self.worker_info.id
            num_workers = self.worker_info.num_workers
        
        # Calculate chunk range for this worker
        total_chunks = self._get_total_chunks()
        chunks_per_worker = total_chunks // num_workers
        start_chunk = worker_id * chunks_per_worker
        end_chunk = start_chunk + chunks_per_worker if worker_id < num_workers - 1 else total_chunks
        
        # Iterate over chunks
        for chunk_idx in range(start_chunk, end_chunk):
            chunk_data = self._load_chunk(chunk_idx)
            
            if self.shuffle:
                np.random.shuffle(chunk_data)
            
            for item in chunk_data:
                yield item
    
    def _get_total_chunks(self) -> int:
        """Get total number of chunks."""
        if self.data_format == DataFormat.HDF5:
            with h5py.File(self.data_path, 'r') as f:
                total_size = len(f['data'])
                return (total_size + self.chunk_size - 1) // self.chunk_size
        else:
            # Estimate for other formats
            return 100  # Default estimate
    
    def _load_chunk(self, chunk_idx: int) -> List[Dict[str, Any]]:
        """Load a specific chunk."""
        start_idx = chunk_idx * self.chunk_size
        end_idx = start_idx + self.chunk_size
        
        if self.data_format == DataFormat.HDF5:
            with h5py.File(self.data_path, 'r') as f:
                chunk_data = f['data'][start_idx:end_idx]
                return [{'data': item} for item in chunk_data]
        elif self.data_format == DataFormat.LMDB:
            with lmdb.open(self.data_path, readonly=True) as env:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                with env.begin() as txn:
                    chunk_data = []
                    for i in range(start_idx, end_idx):
                        key = f"{i:08d}".encode()
                        value = txn.get(key)
                        if value is not None:
                            item = msgpack.unpackb(value, raw=False)
                            chunk_data.append(item)
                    return chunk_data
        else:
            raise ValueError(f"Unsupported format for streaming: {self.data_format}")

class OptimizedTextDataset(Dataset):
    """Optimized text dataset with advanced features."""
    
    def __init__(self, texts: List[str], labels: List[int], 
                 tokenizer=None, max_length: int = 512,
                 truncation: str = 'longest_first',
                 padding: str = 'max_length',
                 return_tensors: str = 'pt',
                 cache_encodings: bool = True):
        
    """__init__ function."""
self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.truncation = truncation
        self.padding = padding
        self.return_tensors = return_tensors
        self.cache_encodings = cache_encodings
        
        # Cache for tokenized encodings
        self.encodings_cache = {}
        
        # Validate inputs
        self._validate_inputs()
        
        # Pre-tokenize if caching is enabled
        if self.cache_encodings and self.tokenizer:
            self._pre_tokenize()
    
    def _validate_inputs(self) -> bool:
        """Validate input data."""
        if len(self.texts) != len(self.labels):
            raise ValueError("Texts and labels must have the same length")
        
        if not self.texts:
            raise ValueError("Texts list cannot be empty")
    
    def _pre_tokenize(self) -> Any:
        """Pre-tokenize all texts for caching."""
        logger.info("Pre-tokenizing texts for caching...")
        
        for i, text in enumerate(tqdm(self.texts, desc="Tokenizing")):
            encoding = self.tokenizer(
                text,
                truncation=self.truncation,
                padding=self.padding,
                max_length=self.max_length,
                return_tensors=self.return_tensors
            )
            self.encodings_cache[i] = encoding
    
    def __len__(self) -> Any:
        return len(self.texts)
    
    def __getitem__(self, idx) -> Optional[Dict[str, Any]]:
        # Check cache first
        if idx in self.encodings_cache:
            encoding = self.encodings_cache[idx]
        else:
            # Tokenize on-the-fly
            encoding = self.tokenizer(
                self.texts[idx],
                truncation=self.truncation,
                padding=self.padding,
                max_length=self.max_length,
                return_tensors=self.return_tensors
            )
        
        # Prepare output
        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }
        
        # Add token_type_ids if available
        if 'token_type_ids' in encoding:
            item['token_type_ids'] = encoding['token_type_ids'].flatten()
        
        return item

class DataLoaderFactory:
    """Factory for creating optimized DataLoaders."""
    
    def __init__(self, device_manager: DeviceManager):
        
    """__init__ function."""
self.device_manager = device_manager
        self.logger = logging.getLogger(f"{__name__}.DataLoaderFactory")
    
    def create_dataloader(self, dataset: Dataset, config: DataLoaderConfig,
                         sampler=None) -> DataLoader:
        """Create optimized DataLoader."""
        # Auto-optimize configuration
        config = self._optimize_config(config)
        
        # Create sampler if not provided
        if sampler is None:
            sampler = self._create_sampler(dataset, config)
        
        # Create DataLoader
        dataloader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            sampler=sampler,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            persistent_workers=config.persistent_workers,
            prefetch_factor=config.prefetch_factor,
            drop_last=config.drop_last,
            timeout=config.timeout,
            generator=config.generator
        )
        
        self.logger.info(f"Created DataLoader: {len(dataloader)} batches, "
                        f"{config.num_workers} workers, pin_memory={config.pin_memory}")
        
        return dataloader
    
    def _optimize_config(self, config: DataLoaderConfig) -> DataLoaderConfig:
        """Optimize configuration based on system resources."""
        # Auto-detect optimal number of workers
        if config.num_workers == -1:
            cpu_count = mp.cpu_count()
            memory_gb = psutil.virtual_memory().total / (1024**3)
            
            # Heuristic: 1 worker per 2GB RAM, max 8 workers
            optimal_workers = min(cpu_count, int(memory_gb / 2), 8)
            config.num_workers = max(1, optimal_workers)
        
        # Auto-detect optimal batch size for pin memory
        if config.pin_memory and config.pin_memory_batch_size is None:
            gpu_memory = self._get_gpu_memory()
            if gpu_memory:
                # Heuristic: batch size based on GPU memory
                optimal_batch_size = min(config.batch_size, int(gpu_memory / 2))
                config.pin_memory_batch_size = optimal_batch_size
        
        # Enable persistent workers for better performance
        if config.num_workers > 0:
            config.persistent_workers = True
        
        return config
    
    def _get_gpu_memory(self) -> Optional[float]:
        """Get available GPU memory in GB."""
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                return gpus[0].memoryFree / 1024  # Convert to GB
        except:
            pass
        return None
    
    def _create_sampler(self, dataset: Dataset, config: DataLoaderConfig):
        """Create appropriate sampler."""
        if config.shuffle:
            if hasattr(dataset, 'weights'):
                # Weighted random sampling
                return WeightedRandomSampler(
                    dataset.weights,
                    len(dataset),
                    replacement=True
                )
            else:
                # Regular random sampling
                return RandomSampler(dataset)
        else:
            return SequentialSampler(dataset)
    
    def create_distributed_dataloader(self, dataset: Dataset, config: DataLoaderConfig,
                                    world_size: int, rank: int) -> DataLoader:
        """Create distributed DataLoader."""
        # Create distributed sampler
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=config.shuffle
        )
        
        # Create DataLoader
        return self.create_dataloader(dataset, config, sampler)
    
    def create_weighted_dataloader(self, dataset: Dataset, config: DataLoaderConfig,
                                 weights: List[float]) -> DataLoader:
        """Create DataLoader with weighted sampling."""
        # Create weighted sampler
        sampler = WeightedRandomSampler(
            weights,
            len(dataset),
            replacement=True
        )
        
        # Create DataLoader
        return self.create_dataloader(dataset, config, sampler)

class DataLoaderManager:
    """Manager for efficient data loading operations."""
    
    def __init__(self, device_manager: DeviceManager):
        
    """__init__ function."""
self.device_manager = device_manager
        self.factory = DataLoaderFactory(device_manager)
        self.logger = logging.getLogger(f"{__name__}.DataLoaderManager")
        self.cache = {}
    
    async def load_dataset(self, data_path: str, data_format: DataFormat,
                          config: DataLoaderConfig, **kwargs) -> Tuple[Dataset, DataLoader]:
        """Load dataset and create DataLoader."""
        # Check cache
        cache_key = f"{data_path}_{data_format.value}_{hash(config)}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Load dataset
        dataset = await self._load_dataset(data_path, data_format, **kwargs)
        
        # Apply caching if configured
        if config.cache_strategy != CacheStrategy.NONE:
            dataset = CachedDataset(
                dataset,
                cache_strategy=config.cache_strategy,
                cache_dir=config.cache_dir,
                cache_size_gb=config.cache_size_gb
            )
        
        # Create DataLoader
        dataloader = self.factory.create_dataloader(dataset, config)
        
        # Cache result
        self.cache[cache_key] = (dataset, dataloader)
        
        return dataset, dataloader
    
    async def _load_dataset(self, data_path: str, data_format: DataFormat, **kwargs) -> Dataset:
        """Load dataset based on format."""
        if data_format == DataFormat.CSV:
            return await self._load_csv_dataset(data_path, **kwargs)
        elif data_format == DataFormat.JSON:
            return await self._load_json_dataset(data_path, **kwargs)
        elif data_format == DataFormat.HDF5:
            return await self._load_hdf5_dataset(data_path, **kwargs)
        elif data_format == DataFormat.LMDB:
            return await self._load_lmdb_dataset(data_path, **kwargs)
        elif data_format == DataFormat.PARQUET:
            return await self._load_parquet_dataset(data_path, **kwargs)
        else:
            raise ValueError(f"Unsupported data format: {data_format}")
    
    async def _load_csv_dataset(self, data_path: str, **kwargs) -> Dataset:
        """Load CSV dataset."""
        # Use ThreadPoolExecutor for I/O operations
        loop = asyncio.get_event_loop()
        
        def load_csv():
            
    """load_csv function."""
df = pd.read_csv(data_path)
            texts = df['text'].tolist()
            labels = df['label'].tolist()
            return OptimizedTextDataset(texts, labels, **kwargs)
        
        return await loop.run_in_executor(None, load_csv)
    
    async def _load_json_dataset(self, data_path: str, **kwargs) -> Dataset:
        """Load JSON dataset."""
        loop = asyncio.get_event_loop()
        
        def load_json():
            
    """load_json function."""
with open(data_path, 'r') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                data = json.load(f)
            texts = [item['text'] for item in data]
            labels = [item['label'] for item in data]
            return OptimizedTextDataset(texts, labels, **kwargs)
        
        return await loop.run_in_executor(None, load_json)
    
    async def _load_hdf5_dataset(self, data_path: str, **kwargs) -> Dataset:
        """Load HDF5 dataset."""
        loop = asyncio.get_event_loop()
        
        def load_hdf5():
            
    """load_hdf5 function."""
with h5py.File(data_path, 'r') as f:
                texts = f['texts'][:].astype(str)
                labels = f['labels'][:]
            return OptimizedTextDataset(texts.tolist(), labels.tolist(), **kwargs)
        
        return await loop.run_in_executor(None, load_hdf5)
    
    async def _load_lmdb_dataset(self, data_path: str, **kwargs) -> Dataset:
        """Load LMDB dataset."""
        # For large datasets, use streaming
        return StreamingDataset(data_path, DataFormat.LMDB, **kwargs)
    
    async def _load_parquet_dataset(self, data_path: str, **kwargs) -> Dataset:
        """Load Parquet dataset."""
        loop = asyncio.get_event_loop()
        
        def load_parquet():
            
    """load_parquet function."""
df = pd.read_parquet(data_path)
            texts = df['text'].tolist()
            labels = df['label'].tolist()
            return OptimizedTextDataset(texts, labels, **kwargs)
        
        return await loop.run_in_executor(None, load_parquet)
    
    def split_dataset(self, dataset: Dataset, train_ratio: float = 0.8,
                     val_ratio: float = 0.1, test_ratio: float = 0.1,
                     random_seed: int = 42) -> Tuple[Dataset, Dataset, Dataset]:
        """Split dataset into train/val/test."""
        total_size = len(dataset)
        
        # Calculate sizes
        train_size = int(total_size * train_ratio)
        val_size = int(total_size * val_ratio)
        test_size = total_size - train_size - val_size
        
        # Split dataset
        train_dataset, val_dataset, test_dataset = random_split(
            dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(random_seed)
        )
        
        return train_dataset, val_dataset, test_dataset
    
    def create_dataloaders(self, train_dataset: Dataset, val_dataset: Dataset,
                          test_dataset: Dataset, config: DataLoaderConfig) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create train/val/test DataLoaders."""
        # Create DataLoaders
        train_loader = self.factory.create_dataloader(train_dataset, config)
        
        # Validation and test should not shuffle
        val_config = DataLoaderConfig(**vars(config))
        val_config.shuffle = False
        val_loader = self.factory.create_dataloader(val_dataset, val_config)
        
        test_config = DataLoaderConfig(**vars(config))
        test_config.shuffle = False
        test_loader = self.factory.create_dataloader(test_dataset, test_config)
        
        return train_loader, val_loader, test_loader
    
    def get_dataloader_stats(self, dataloader: DataLoader) -> Dict[str, Any]:
        """Get DataLoader statistics."""
        total_batches = len(dataloader)
        total_samples = total_batches * dataloader.batch_size
        
        # Estimate memory usage
        sample_batch = next(iter(dataloader))
        batch_size_bytes = sum(tensor.element_size() * tensor.nelement() 
                              for tensor in sample_batch.values() 
                              if isinstance(tensor, torch.Tensor))
        
        estimated_memory_mb = (batch_size_bytes * total_batches) / (1024 * 1024)
        
        return {
            'total_batches': total_batches,
            'total_samples': total_samples,
            'batch_size': dataloader.batch_size,
            'num_workers': dataloader.num_workers,
            'pin_memory': dataloader.pin_memory,
            'estimated_memory_mb': estimated_memory_mb,
            'persistent_workers': dataloader.persistent_workers,
            'prefetch_factor': dataloader.prefetch_factor
        }

# Performance monitoring
class DataLoaderProfiler:
    """Profile DataLoader performance."""
    
    def __init__(self) -> Any:
        self.logger = logging.getLogger(f"{__name__}.DataLoaderProfiler")
        self.metrics = {}
    
    def profile_dataloader(self, dataloader: DataLoader, num_batches: int = 10) -> Dict[str, float]:
        """Profile DataLoader performance."""
        self.logger.info(f"Profiling DataLoader for {num_batches} batches...")
        
        # Warm up
        warmup_iter = iter(dataloader)
        for _ in range(min(2, num_batches)):
            next(warmup_iter)
        
        # Profile
        start_time = time.time()
        batch_times = []
        
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break
            
            batch_start = time.time()
            # Process batch (just move to device for timing)
            batch = {k: v.to('cpu') if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            batch_time = time.time() - batch_start
            batch_times.append(batch_time)
        
        total_time = time.time() - start_time
        
        # Calculate metrics
        avg_batch_time = np.mean(batch_times)
        std_batch_time = np.std(batch_times)
        throughput = num_batches / total_time
        
        metrics = {
            'total_time': total_time,
            'avg_batch_time': avg_batch_time,
            'std_batch_time': std_batch_time,
            'throughput_batches_per_sec': throughput,
            'throughput_samples_per_sec': throughput * dataloader.batch_size
        }
        
        self.metrics = metrics
        self.logger.info(f"Profiling results: {metrics}")
        
        return metrics
    
    def optimize_dataloader_config(self, current_config: DataLoaderConfig,
                                 target_throughput: float) -> DataLoaderConfig:
        """Optimize DataLoader configuration based on profiling."""
        optimized_config = DataLoaderConfig(**vars(current_config))
        
        # Simple optimization heuristics
        if self.metrics.get('throughput_samples_per_sec', 0) < target_throughput:
            # Increase workers if CPU-bound
            if optimized_config.num_workers < mp.cpu_count():
                optimized_config.num_workers = min(
                    optimized_config.num_workers + 2,
                    mp.cpu_count()
                )
            
            # Increase batch size if memory allows
            if optimized_config.batch_size < 128:
                optimized_config.batch_size = min(
                    optimized_config.batch_size * 2,
                    128
                )
        
        return optimized_config

# Factory functions
async def create_data_loader_manager(device_manager: DeviceManager) -> DataLoaderManager:
    """Create a DataLoader manager instance."""
    return DataLoaderManager(device_manager)

async def create_data_loader_factory(device_manager: DeviceManager) -> DataLoaderFactory:
    """Create a DataLoader factory instance."""
    return DataLoaderFactory(device_manager)

# Quick usage functions
async def quick_dataloader(
    data_path: str,
    data_format: DataFormat = DataFormat.CSV,
    batch_size: int = 32,
    num_workers: int = 4
) -> Tuple[Dataset, DataLoader]:
    """Quick DataLoader creation."""
    device_manager = DeviceManager()
    manager = await create_data_loader_manager(device_manager)
    
    config = DataLoaderConfig(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True
    )
    
    return await manager.load_dataset(data_path, data_format, config)

async def quick_dataloader_split(
    data_path: str,
    data_format: DataFormat = DataFormat.CSV,
    batch_size: int = 32,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Quick DataLoader creation with train/val/test split."""
    device_manager = DeviceManager()
    manager = await create_data_loader_manager(device_manager)
    
    config = DataLoaderConfig(
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    
    # Load dataset
    dataset, _ = await manager.load_dataset(data_path, data_format, config)
    
    # Split dataset
    train_dataset, val_dataset, test_dataset = manager.split_dataset(
        dataset, train_ratio, val_ratio, test_ratio
    )
    
    # Create DataLoaders
    return manager.create_dataloaders(train_dataset, val_dataset, test_dataset, config)

# Example usage
if __name__ == "__main__":
    async def demo():
        
    """demo function."""
# Quick DataLoader example
        dataset, dataloader = await quick_dataloader(
            data_path="data/sentiment_dataset.csv",
            data_format=DataFormat.CSV,
            batch_size=16
        )
        
        print(f"Dataset size: {len(dataset)}")
        print(f"DataLoader batches: {len(dataloader)}")
        
        # Test batch loading
        for i, batch in enumerate(dataloader):
            print(f"Batch {i}: {batch.keys()}")
            if i >= 2:  # Show first 3 batches
                break
        
        # Profile performance
        profiler = DataLoaderProfiler()
        metrics = profiler.profile_dataloader(dataloader, num_batches=5)
        print(f"Performance metrics: {metrics}")
    
    asyncio.run(demo()) 