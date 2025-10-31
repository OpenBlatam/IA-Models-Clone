from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
BUFFER_SIZE = 1024

import asyncio
import json
import logging
import os
import pickle
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Iterator
from functools import lru_cache
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import (
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data._utils.collate import default_collate
import torch.multiprocessing as mp
from torchvision import transforms
import structlog
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import dask.dataframe as dd
import vaex
from numba import jit
import psutil
import gc
from typing import Any, List, Dict, Optional
"""
Efficient Data Loading System for Cybersecurity Applications

This module provides optimized data loading capabilities using PyTorch's DataLoader
with features specifically designed for cybersecurity applications:
- Custom datasets for different data types
- Efficient data preprocessing and augmentation
- Memory optimization and caching
- Multi-process data loading
- Performance monitoring and profiling
- Security-focused data validation
"""


    DataLoader, Dataset, TensorDataset, IterableDataset,
    WeightedRandomSampler, SequentialSampler, RandomSampler,
    BatchSampler, Subset, ConcatDataset
)

# Configure structured logging
logger = structlog.get_logger(__name__)


@dataclass
class DataLoaderConfig:
    """Configuration for efficient data loading."""
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2
    drop_last: bool = False
    shuffle: bool = True
    collate_fn: Optional[Callable] = None
    sampler: Optional[torch.utils.data.Sampler] = None
    batch_sampler: Optional[torch.utils.data.BatchSampler] = None
    timeout: int = 0
    worker_init_fn: Optional[Callable] = None
    multiprocessing_context: str = "spawn"
    generator: Optional[torch.Generator] = None
    
    # Memory optimization
    pin_memory_device: str = ""
    memory_format: torch.memory_format = torch.contiguous_format
    
    # Caching
    enable_caching: bool = True
    cache_dir: str = "./cache"
    cache_size: int = 1000
    
    # Performance monitoring
    enable_monitoring: bool = True
    monitor_interval: int = 100
    
    # Security
    validate_data: bool = True
    max_sequence_length: int = 512
    sanitize_inputs: bool = True


class BaseCybersecurityDataset(Dataset, ABC):
    """Abstract base class for cybersecurity datasets."""
    
    def __init__(self, data_path: str, config: DataLoaderConfig):
        
    """__init__ function."""
self.data_path = data_path
        self.config = config
        self.data = []
        self.labels = []
        self.metadata = {}
        self._load_data()
        self._validate_data()
    
    @abstractmethod
    def _load_data(self) -> Any:
        """Load and preprocess the dataset."""
        pass
    
    def _validate_data(self) -> bool:
        """Validate data integrity and security."""
        if not self.config.validate_data:
            return
        
        # Check for empty dataset
        if len(self.data) == 0:
            raise ValueError("Dataset is empty")
        
        # Check for data consistency
        if len(self.data) != len(self.labels):
            raise ValueError("Data and labels have different lengths")
        
        # Check for malicious content
        if self.config.sanitize_inputs:
            self._sanitize_data()
    
    def _sanitize_data(self) -> Any:
        """Sanitize data to prevent security issues."""
        sanitized_data = []
        sanitized_labels = []
        
        for i, (data_item, label) in enumerate(zip(self.data, self.labels)):
            try:
                sanitized_item = self._sanitize_item(data_item)
                sanitized_data.append(sanitized_item)
                sanitized_labels.append(label)
            except Exception as e:
                logger.warning(f"Removing malicious item at index {i}: {e}")
                continue
        
        self.data = sanitized_data
        self.labels = sanitized_labels
    
    def _sanitize_item(self, item: Any) -> Any:
        """Sanitize a single data item."""
        if isinstance(item, str):
            # Remove potentially dangerous characters
            dangerous_chars = ['<script>', 'javascript:', 'data:', 'vbscript:']
            for char in dangerous_chars:
                if char.lower() in item.lower():
                    raise ValueError(f"Potentially dangerous content detected: {char}")
            
            # Limit string length
            if len(item) > self.config.max_sequence_length:
                item = item[:self.config.max_sequence_length]
        
        elif isinstance(item, (list, tuple)):
            item = [self._sanitize_item(subitem) for subitem in item]
        
        elif isinstance(item, dict):
            item = {k: self._sanitize_item(v) for k, v in item.items()}
        
        return item
    
    def __len__(self) -> Any:
        return len(self.data)
    
    def __getitem__(self, idx) -> Optional[Dict[str, Any]]:
        return self.data[idx], self.labels[idx]
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get dataset metadata."""
        return {
            "size": len(self.data),
            "data_path": self.data_path,
            "config": self.config.__dict__,
            **self.metadata
        }


class ThreatDetectionDataset(BaseCybersecurityDataset):
    """Optimized dataset for threat detection."""
    
    def __init__(self, data_path: str, config: DataLoaderConfig, tokenizer=None):
        
    """__init__ function."""
self.tokenizer = tokenizer
        super().__init__(data_path, config)
    
    def _load_data(self) -> Any:
        """Load threat detection dataset with optimizations."""
        try:
            # Use efficient data loading based on file size
            file_size = os.path.getsize(self.data_path)
            
            if file_size > 100 * 1024 * 1024:  # > 100MB
                # Use Dask for large files
                df = dd.read_csv(self.data_path).compute()
            elif file_size > 10 * 1024 * 1024:  # > 10MB
                # Use Vaex for medium files
                df = vaex.read_csv(self.data_path).to_pandas_df()
            else:
                # Use pandas for small files
                df = pd.read_csv(self.data_path)
            
            texts = df['text'].tolist()
            labels = df['label'].tolist()
            
            # Pre-tokenize if tokenizer is provided
            if self.tokenizer:
                self._preprocess_with_tokenizer(texts, labels)
            else:
                self.data = texts
                self.labels = labels
            
            # Store metadata
            self.metadata.update({
                "num_classes": len(set(labels)),
                "class_distribution": pd.Series(labels).value_counts().to_dict(),
                "avg_text_length": np.mean([len(str(t)) for t in texts]),
                "file_size_mb": file_size / (1024 * 1024)
            })
            
        except Exception as e:
            logger.error("Failed to load threat detection dataset", error=str(e))
            raise
    
    def _preprocess_with_tokenizer(self, texts: List[str], labels: List[int]):
        """Preprocess texts with tokenizer for efficiency."""
        # Batch tokenization for better performance
        batch_size = 1000
        tokenized_data = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize batch
            encodings = self.tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=self.config.max_sequence_length,
                return_tensors='pt'
            )
            
            # Convert to list of dictionaries
            for j in range(len(batch_texts)):
                tokenized_data.append({
                    'input_ids': encodings['input_ids'][j],
                    'attention_mask': encodings['attention_mask'][j]
                })
        
        self.data = tokenized_data
        self.labels = labels


class AnomalyDetectionDataset(BaseCybersecurityDataset):
    """Optimized dataset for anomaly detection."""
    
    def _load_data(self) -> Any:
        """Load anomaly detection dataset with optimizations."""
        try:
            df = pd.read_csv(self.data_path)
            
            # Parse JSON features efficiently
            features = []
            labels = []
            
            for _, row in df.iterrows():
                try:
                    feature_vector = json.loads(row['features'])
                    features.append(feature_vector)
                    labels.append(row['label'])
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"Skipping malformed row: {e}")
                    continue
            
            # Convert to tensors
            self.data = torch.tensor(features, dtype=torch.float32)
            self.labels = torch.tensor(labels, dtype=torch.long)
            
            # Store metadata
            self.metadata.update({
                "feature_dim": self.data.shape[1],
                "num_samples": len(self.data),
                "anomaly_ratio": (self.labels == 1).float().mean().item()
            })
            
        except Exception as e:
            logger.error("Failed to load anomaly detection dataset", error=str(e))
            raise


class NetworkTrafficDataset(BaseCybersecurityDataset):
    """Optimized dataset for network traffic analysis."""
    
    def _load_data(self) -> Any:
        """Load network traffic dataset with optimizations."""
        try:
            # Load network traffic data
            df = pd.read_csv(self.data_path)
            
            # Extract features
            feature_columns = [col for col in df.columns if col not in ['label', 'timestamp']]
            features = df[feature_columns].values
            labels = df['label'].values
            
            # Normalize features
            scaler = StandardScaler()
            features_normalized = scaler.fit_transform(features)
            
            # Convert to tensors
            self.data = torch.tensor(features_normalized, dtype=torch.float32)
            self.labels = torch.tensor(labels, dtype=torch.long)
            
            # Store scaler for inference
            self.metadata['scaler'] = scaler
            self.metadata.update({
                "feature_dim": features.shape[1],
                "num_samples": len(features),
                "feature_columns": feature_columns
            })
            
        except Exception as e:
            logger.error("Failed to load network traffic dataset", error=str(e))
            raise


class MalwareDataset(BaseCybersecurityDataset):
    """Optimized dataset for malware classification."""
    
    def _load_data(self) -> Any:
        """Load malware dataset with optimizations."""
        try:
            df = pd.read_csv(self.data_path)
            
            # Extract binary features and API calls
            binary_features = []
            api_sequences = []
            labels = []
            
            for _, row in df.iterrows():
                # Binary features
                binary_feat = json.loads(row['binary_features'])
                binary_features.append(binary_feat)
                
                # API call sequence
                api_seq = json.loads(row['api_calls'])
                api_sequences.append(api_seq)
                
                labels.append(row['label'])
            
            # Convert to tensors
            self.data = {
                'binary_features': torch.tensor(binary_features, dtype=torch.float32),
                'api_sequences': api_sequences  # Keep as list for variable length
            }
            self.labels = torch.tensor(labels, dtype=torch.long)
            
            # Store metadata
            self.metadata.update({
                "num_classes": len(set(labels)),
                "binary_feature_dim": len(binary_features[0]),
                "avg_api_sequence_length": np.mean([len(seq) for seq in api_sequences])
            })
            
        except Exception as e:
            logger.error("Failed to load malware dataset", error=str(e))
            raise


class CachedDataset(Dataset):
    """Dataset wrapper with caching capabilities."""
    
    def __init__(self, dataset: Dataset, cache_dir: str, cache_size: int = 1000):
        
    """__init__ function."""
self.dataset = dataset
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_size = cache_size
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Load existing cache
        self._load_cache()
    
    def _load_cache(self) -> Any:
        """Load existing cache from disk."""
        cache_file = self.cache_dir / "cache_metadata.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                    cache_data = pickle.load(f)
                    self.cache = cache_data.get('cache', {})
                    self.cache_hits = cache_data.get('hits', 0)
                    self.cache_misses = cache_data.get('misses', 0)
                logger.info(f"Loaded cache with {len(self.cache)} items")
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
    
    def _save_cache(self) -> Any:
        """Save cache to disk."""
        cache_file = self.cache_dir / "cache_metadata.pkl"
        try:
            cache_data = {
                'cache': self.cache,
                'hits': self.cache_hits,
                'misses': self.cache_misses
            }
            with open(cache_file, 'wb') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                pickle.dump(cache_data, f)
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
    
    def __len__(self) -> Any:
        return len(self.dataset)
    
    def __getitem__(self, idx) -> Optional[Dict[str, Any]]:
        # Check cache first
        if idx in self.cache:
            self.cache_hits += 1
            return self.cache[idx]
        
        # Load from dataset
        self.cache_misses += 1
        item = self.dataset[idx]
        
        # Cache the item
        if len(self.cache) < self.cache_size:
            self.cache[idx] = item
        
        return item
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0
        
        return {
            "cache_size": len(self.cache),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": hit_rate,
            "total_requests": total_requests
        }


class DataAugmentation:
    """Data augmentation for cybersecurity datasets."""
    
    @staticmethod
    def augment_text(text: str, augmentation_prob: float = 0.3) -> str:
        """Augment text data for threat detection."""
        if np.random.random() > augmentation_prob:
            return text
        
        augmented_text = text
        
        # Random character substitution
        if np.random.random() < 0.1:
            chars = list(augmented_text)
            if len(chars) > 1:
                idx1, idx2 = np.random.choice(len(chars), 2, replace=False)
                chars[idx1], chars[idx2] = chars[idx2], chars[idx1]
                augmented_text = ''.join(chars)
        
        # Random word insertion
        if np.random.random() < 0.05:
            words = augmented_text.split()
            if len(words) > 0:
                insert_word = np.random.choice(words)
                insert_pos = np.random.randint(0, len(words) + 1)
                words.insert(insert_pos, insert_word)
                augmented_text = ' '.join(words)
        
        # Random word deletion
        if np.random.random() < 0.05:
            words = augmented_text.split()
            if len(words) > 1:
                del_pos = np.random.randint(0, len(words))
                words.pop(del_pos)
                augmented_text = ' '.join(words)
        
        return augmented_text
    
    @staticmethod
    def augment_features(features: np.ndarray, noise_factor: float = 0.01) -> np.ndarray:
        """Augment numerical features for anomaly detection."""
        if np.random.random() > 0.5:
            return features
        
        # Add Gaussian noise
        noise = np.random.normal(0, noise_factor, features.shape)
        augmented_features = features + noise
        
        # Ensure features stay in valid range
        augmented_features = np.clip(augmented_features, 0, 1)
        
        return augmented_features


class CustomCollateFn:
    """Custom collate functions for different data types."""
    
    @staticmethod
    def threat_detection_collate(batch) -> Any:
        """Collate function for threat detection data."""
        texts, labels = zip(*batch)
        
        # Handle tokenized and non-tokenized texts
        if isinstance(texts[0], dict):
            # Already tokenized
            input_ids = torch.stack([item['input_ids'] for item in texts])
            attention_masks = torch.stack([item['attention_mask'] for item in texts])
            
            return {
                'input_ids': input_ids,
                'attention_mask': attention_masks,
                'labels': torch.tensor(labels)
            }
        else:
            # Non-tokenized texts
            return {
                'texts': texts,
                'labels': torch.tensor(labels)
            }
    
    @staticmethod
    def anomaly_detection_collate(batch) -> Any:
        """Collate function for anomaly detection data."""
        features, labels = zip(*batch)
        
        # Stack features
        features_tensor = torch.stack(features)
        labels_tensor = torch.stack(labels)
        
        return features_tensor, labels_tensor
    
    @staticmethod
    def malware_collate(batch) -> Any:
        """Collate function for malware data."""
        data, labels = zip(*batch)
        
        # Handle binary features and API sequences
        binary_features = torch.stack([item['binary_features'] for item in data])
        api_sequences = [item['api_sequences'] for item in data]
        labels_tensor = torch.stack(labels)
        
        return {
            'binary_features': binary_features,
            'api_sequences': api_sequences,
            'labels': labels_tensor
        }


class DataLoaderFactory:
    """Factory for creating optimized DataLoaders."""
    
    @staticmethod
    def create_dataloader(
        dataset: Dataset,
        config: DataLoaderConfig,
        dataset_type: str = "generic"
    ) -> DataLoader:
        """Create an optimized DataLoader."""
        
        # Apply caching if enabled
        if config.enable_caching:
            dataset = CachedDataset(dataset, config.cache_dir, config.cache_size)
        
        # Select appropriate collate function
        if config.collate_fn is None:
            if dataset_type == "threat_detection":
                config.collate_fn = CustomCollateFn.threat_detection_collate
            elif dataset_type == "anomaly_detection":
                config.collate_fn = CustomCollateFn.anomaly_detection_collate
            elif dataset_type == "malware":
                config.collate_fn = CustomCollateFn.malware_collate
        
        # Create sampler
        sampler = config.sampler
        if sampler is None:
            if config.shuffle:
                sampler = RandomSampler(dataset, generator=config.generator)
            else:
                sampler = SequentialSampler(dataset)
        
        # Create DataLoader
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=config.batch_size,
            sampler=sampler,
            batch_sampler=config.batch_sampler,
            num_workers=config.num_workers,
            collate_fn=config.collate_fn,
            pin_memory=config.pin_memory,
            drop_last=config.drop_last,
            timeout=config.timeout,
            worker_init_fn=config.worker_init_fn,
            multiprocessing_context=config.multiprocessing_context,
            persistent_workers=config.persistent_workers,
            prefetch_factor=config.prefetch_factor,
            generator=config.generator
        )
        
        return dataloader


class DataLoaderMonitor:
    """Monitor DataLoader performance and memory usage."""
    
    def __init__(self, dataloader: DataLoader, config: DataLoaderConfig):
        
    """__init__ function."""
self.dataloader = dataloader
        self.config = config
        self.metrics = {
            "load_times": [],
            "memory_usage": [],
            "batch_sizes": [],
            "errors": []
        }
        self.start_time = None
    
    def start_monitoring(self) -> Any:
        """Start monitoring the DataLoader."""
        self.start_time = time.time()
        logger.info("Started DataLoader monitoring")
    
    def record_batch_load(self, batch_load_time: float, batch_size: int, memory_usage: float):
        """Record batch loading metrics."""
        self.metrics["load_times"].append(batch_load_time)
        self.metrics["batch_sizes"].append(batch_size)
        self.metrics["memory_usage"].append(memory_usage)
    
    def record_error(self, error: Exception):
        """Record DataLoader errors."""
        self.metrics["errors"].append({
            "error": str(error),
            "timestamp": time.time()
        })
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get performance report."""
        if not self.metrics["load_times"]:
            return {"error": "No metrics recorded"}
        
        avg_load_time = np.mean(self.metrics["load_times"])
        avg_memory_usage = np.mean(self.metrics["memory_usage"])
        total_batches = len(self.metrics["load_times"])
        error_count = len(self.metrics["errors"])
        
        return {
            "total_batches": total_batches,
            "avg_load_time_ms": avg_load_time * 1000,
            "avg_memory_usage_mb": avg_memory_usage,
            "error_count": error_count,
            "throughput_batches_per_sec": total_batches / (time.time() - self.start_time) if self.start_time else 0,
            "errors": self.metrics["errors"][-10:]  # Last 10 errors
        }


class MemoryOptimizedDataLoader:
    """Memory-optimized DataLoader with automatic garbage collection."""
    
    def __init__(self, dataloader: DataLoader, max_memory_usage: float = 0.8):
        
    """__init__ function."""
self.dataloader = dataloader
        self.max_memory_usage = max_memory_usage
        self.monitor = DataLoaderMonitor(dataloader, DataLoaderConfig())
    
    def __iter__(self) -> Any:
        self.monitor.start_monitoring()
        
        for batch_idx, batch in enumerate(self.dataloader):
            start_time = time.time()
            
            # Check memory usage
            memory_usage = psutil.virtual_memory().percent / 100
            if memory_usage > self.max_memory_usage:
                logger.warning(f"High memory usage: {memory_usage:.2%}")
                gc.collect()
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            # Record metrics
            load_time = time.time() - start_time
            batch_size = len(batch[0]) if isinstance(batch, (list, tuple)) else batch.shape[0]
            self.monitor.record_batch_load(load_time, batch_size, memory_usage)
            
            yield batch
    
    def __len__(self) -> Any:
        return len(self.dataloader)
    
    def get_performance_report(self) -> Optional[Dict[str, Any]]:
        return self.monitor.get_performance_report()


class AsyncDataLoader:
    """Asynchronous DataLoader for non-blocking data loading."""
    
    def __init__(self, dataloader: DataLoader, max_queue_size: int = 10):
        
    """__init__ function."""
self.dataloader = dataloader
        self.max_queue_size = max_queue_size
        self.queue = asyncio.Queue(maxsize=max_queue_size)
        self.producer_task = None
        self.consumer_task = None
    
    async def start(self) -> Any:
        """Start the async DataLoader."""
        self.producer_task = asyncio.create_task(self._producer())
    
    async def stop(self) -> Any:
        """Stop the async DataLoader."""
        if self.producer_task:
            self.producer_task.cancel()
            try:
                await self.producer_task
            except asyncio.CancelledError:
                pass
    
    async def _producer(self) -> Any:
        """Produce batches and put them in the queue."""
        try:
            for batch in self.dataloader:
                await self.queue.put(batch)
        except Exception as e:
            logger.error(f"Error in producer: {e}")
        finally:
            await self.queue.put(None)  # Sentinel value
    
    async def __aiter__(self) -> Any:
        """Async iterator."""
        await self.start()
        try:
            while True:
                batch = await self.queue.get()
                if batch is None:  # Sentinel value
                    break
                yield batch
        finally:
            await self.stop()


class DataLoaderBenchmark:
    """Benchmark DataLoader performance."""
    
    @staticmethod
    def benchmark_dataloader(
        dataloader: DataLoader,
        num_batches: int = 100,
        warmup_batches: int = 10
    ) -> Dict[str, Any]:
        """Benchmark DataLoader performance."""
        
        # Warmup
        logger.info("Warming up DataLoader...")
        warmup_iter = iter(dataloader)
        for _ in range(warmup_batches):
            try:
                next(warmup_iter)
            except StopIteration:
                break
        
        # Benchmark
        logger.info("Starting benchmark...")
        start_time = time.time()
        batch_times = []
        memory_usage = []
        
        iter_dataloader = iter(dataloader)
        for i in range(num_batches):
            batch_start = time.time()
            
            try:
                batch = next(iter_dataloader)
                batch_time = time.time() - batch_start
                batch_times.append(batch_time)
                
                # Record memory usage
                mem_usage = psutil.virtual_memory().percent
                memory_usage.append(mem_usage)
                
                if i % 10 == 0:
                    logger.info(f"Processed {i+1}/{num_batches} batches")
                    
            except StopIteration:
                logger.warning(f"DataLoader exhausted after {i} batches")
                break
        
        total_time = time.time() - start_time
        
        # Calculate statistics
        avg_batch_time = np.mean(batch_times)
        std_batch_time = np.std(batch_times)
        avg_memory_usage = np.mean(memory_usage)
        
        return {
            "total_batches": len(batch_times),
            "total_time_seconds": total_time,
            "avg_batch_time_ms": avg_batch_time * 1000,
            "std_batch_time_ms": std_batch_time * 1000,
            "throughput_batches_per_sec": len(batch_times) / total_time,
            "avg_memory_usage_percent": avg_memory_usage,
            "max_memory_usage_percent": max(memory_usage),
            "min_memory_usage_percent": min(memory_usage)
        }


# Utility functions
def create_balanced_sampler(dataset: Dataset, labels: List[int]) -> WeightedRandomSampler:
    """Create a balanced sampler for imbalanced datasets."""
    class_counts = np.bincount(labels)
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[label] for label in labels]
    
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )


def split_dataset(dataset: Dataset, train_ratio: float = 0.8, val_ratio: float = 0.1) -> Tuple[Dataset, Dataset, Dataset]:
    """Split dataset into train, validation, and test sets."""
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    return train_dataset, val_dataset, test_dataset


@lru_cache(maxsize=128)
def get_dataset_info(dataset_path: str) -> Dict[str, Any]:
    """Get dataset information with caching."""
    try:
        df = pd.read_csv(dataset_path)
        return {
            "size": len(df),
            "columns": df.columns.tolist(),
            "memory_usage_mb": df.memory_usage(deep=True).sum() / (1024 * 1024),
            "dtypes": df.dtypes.to_dict()
        }
    except Exception as e:
        logger.error(f"Failed to get dataset info: {e}")
        return {}


def optimize_dataloader_config(
    dataset_size: int,
    available_memory_gb: float,
    num_cpus: int
) -> DataLoaderConfig:
    """Optimize DataLoader configuration based on system resources."""
    
    # Calculate optimal batch size
    memory_per_sample_mb = 0.1  # Estimate
    max_batch_size = int((available_memory_gb * 1024 * 0.5) / memory_per_sample_mb)
    optimal_batch_size = min(64, max_batch_size)
    
    # Calculate optimal number of workers
    optimal_workers = min(num_cpus - 1, 8)  # Leave one CPU free
    
    return DataLoaderConfig(
        batch_size=optimal_batch_size,
        num_workers=optimal_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )


# Example usage
if __name__ == "__main__":
    # Example: Create optimized DataLoader for threat detection
    config = DataLoaderConfig(
        batch_size=32,
        num_workers=4,
        pin_memory=True,
        enable_caching=True
    )
    
    # Create dataset
    dataset = ThreatDetectionDataset("data/threats.csv", config)
    
    # Create DataLoader
    dataloader = DataLoaderFactory.create_dataloader(
        dataset, config, "threat_detection"
    )
    
    # Benchmark performance
    benchmark_results = DataLoaderBenchmark.benchmark_dataloader(dataloader)
    print("Benchmark Results:", benchmark_results) 