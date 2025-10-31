#!/usr/bin/env python3
"""
Async Data Loading Pipeline for SEO Evaluation System
High-performance async data loading with prefetching and caching
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, IterableDataset
import asyncio
import aiofiles
import aiohttp
import numpy as np
import pandas as pd
import logging
import time
import threading
import queue
from typing import Dict, Any, Optional, List, Tuple, Union, AsyncGenerator, Iterator
from dataclasses import dataclass, field
from contextlib import asynccontextmanager
import warnings
from pathlib import Path
import json
import pickle
import hashlib
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from functools import lru_cache
import weakref
from collections import OrderedDict, deque

warnings.filterwarnings("ignore")

@dataclass
class AsyncDataConfig:
    """Configuration for async data loading."""
    # Data Loading
    batch_size: int = 32
    num_workers: int = 4
    prefetch_factor: int = 2
    pin_memory: bool = True
    persistent_workers: bool = True
    drop_last: bool = False
    
    # Async Settings
    enable_async_loading: bool = True
    async_buffer_size: int = 1000
    async_timeout: float = 30.0
    enable_prefetching: bool = True
    prefetch_batches: int = 4
    
    # Caching
    enable_data_caching: bool = True
    cache_size: int = 1000
    cache_ttl: int = 3600
    enable_disk_caching: bool = True
    cache_dir: str = "./cache"
    
    # Optimization
    enable_compression: bool = True
    enable_parallel_processing: bool = True
    max_parallel_workers: int = 8
    enable_memory_mapping: bool = True
    
    # Monitoring
    enable_loading_monitoring: bool = True
    monitor_interval: float = 1.0

class AsyncDataLoader:
    """High-performance async data loader with prefetching and caching."""
    
    def __init__(self, config: AsyncDataConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.cache_manager = DataCacheManager(config)
        self.prefetch_queue = asyncio.Queue(maxsize=config.prefetch_batches)
        self.loading_stats = LoadingStats()
        
        # Initialize async components
        self._setup_async_components()
        
    def _setup_async_components(self):
        """Setup async components."""
        if self.config.enable_disk_caching:
            Path(self.config.cache_dir).mkdir(parents=True, exist_ok=True)
            
        if self.config.enable_loading_monitoring:
            self.loading_stats.start_monitoring()
    
    async def load_dataset(self, dataset: Dataset) -> AsyncGenerator[torch.Tensor, None]:
        """Load dataset asynchronously with prefetching."""
        if not self.config.enable_async_loading:
            # Fallback to synchronous loading
            for batch in self._sync_loader(dataset):
                yield batch
            return
            
        # Start prefetching
        prefetch_task = asyncio.create_task(self._prefetch_data(dataset))
        
        try:
            while True:
                try:
                    # Get batch from prefetch queue with timeout
                    batch = await asyncio.wait_for(
                        self.prefetch_queue.get(), 
                        timeout=self.config.async_timeout
                    )
                    
                    if batch is None:  # End of dataset
                        break
                        
                    yield batch
                    
                except asyncio.TimeoutError:
                    self.logger.warning("Timeout waiting for data batch")
                    break
                    
        finally:
            # Cancel prefetching task
            prefetch_task.cancel()
            try:
                await prefetch_task
            except asyncio.CancelledError:
                pass
    
    async def _prefetch_data(self, dataset: Dataset):
        """Prefetch data batches asynchronously."""
        try:
            # Create data loader for prefetching
            loader = DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                num_workers=self.config.num_workers,
                pin_memory=self.config.pin_memory,
                persistent_workers=self.config.persistent_workers,
                drop_last=self.config.drop_last,
                prefetch_factor=self.config.prefetch_factor
            )
            
            # Prefetch batches
            for batch in loader:
                # Check cache first
                cache_key = self._generate_cache_key(batch)
                cached_batch = await self.cache_manager.get_cached_batch(cache_key)
                
                if cached_batch is not None:
                    await self.prefetch_queue.put(cached_batch)
                else:
                    # Process batch asynchronously
                    processed_batch = await self._process_batch_async(batch)
                    
                    # Cache processed batch
                    await self.cache_manager.cache_batch(cache_key, processed_batch)
                    
                    await self.prefetch_queue.put(processed_batch)
                    
            # Signal end of dataset
            await self.prefetch_queue.put(None)
            
        except Exception as e:
            self.logger.error(f"Error in prefetching: {e}")
            await self.prefetch_queue.put(None)
    
    async def _process_batch_async(self, batch: torch.Tensor) -> torch.Tensor:
        """Process batch asynchronously."""
        if self.config.enable_parallel_processing:
            # Process in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor(max_workers=self.config.max_parallel_workers) as executor:
                processed_batch = await loop.run_in_executor(
                    executor, 
                    self._process_batch_sync, 
                    batch
                )
        else:
            processed_batch = self._process_batch_sync(batch)
            
        return processed_batch
    
    def _process_batch_sync(self, batch: torch.Tensor) -> torch.Tensor:
        """Process batch synchronously."""
        # Apply data preprocessing
        if self.config.enable_compression:
            batch = self._compress_batch(batch)
            
        return batch
    
    def _compress_batch(self, batch: torch.Tensor) -> torch.Tensor:
        """Compress batch to reduce memory usage."""
        # Convert to half precision if not already
        if batch.dtype == torch.float32:
            batch = batch.half()
            
        return batch
    
    def _generate_cache_key(self, batch: torch.Tensor) -> str:
        """Generate cache key for batch."""
        # Create hash of batch content
        batch_hash = hashlib.md5(batch.numpy().tobytes()).hexdigest()
        return f"batch_{batch_hash}"
    
    def _sync_loader(self, dataset: Dataset) -> Iterator[torch.Tensor]:
        """Synchronous data loader fallback."""
        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=self.config.persistent_workers,
            drop_last=self.config.drop_last
        )
        
        for batch in loader:
            yield batch

class DataCacheManager:
    """Manage data caching for async loading."""
    
    def __init__(self, config: AsyncDataConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.memory_cache = OrderedDict()
        self.cache_timestamps = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
    async def get_cached_batch(self, key: str) -> Optional[torch.Tensor]:
        """Get cached batch asynchronously."""
        if not self.config.enable_data_caching:
            return None
            
        # Check memory cache first
        if key in self.memory_cache:
            # Check TTL
            if time.time() - self.cache_timestamps[key] > self.config.cache_ttl:
                del self.memory_cache[key]
                del self.cache_timestamps[key]
                self.cache_misses += 1
                return None
                
            # Move to end (LRU)
            batch = self.memory_cache.pop(key)
            self.memory_cache[key] = batch
            self.cache_timestamps[key] = time.time()
            
            self.cache_hits += 1
            self.logger.debug(f"Memory cache hit: {key}")
            return batch
        
        # Check disk cache
        if self.config.enable_disk_caching:
            disk_batch = await self._load_from_disk_cache(key)
            if disk_batch is not None:
                # Add to memory cache
                self._add_to_memory_cache(key, disk_batch)
                self.cache_hits += 1
                self.logger.debug(f"Disk cache hit: {key}")
                return disk_batch
        
        self.cache_misses += 1
        return None
    
    async def cache_batch(self, key: str, batch: torch.Tensor):
        """Cache batch asynchronously."""
        if not self.config.enable_data_caching:
            return
            
        # Add to memory cache
        self._add_to_memory_cache(key, batch)
        
        # Add to disk cache
        if self.config.enable_disk_caching:
            await self._save_to_disk_cache(key, batch)
    
    def _add_to_memory_cache(self, key: str, batch: torch.Tensor):
        """Add batch to memory cache."""
        if len(self.memory_cache) >= self.config.cache_size:
            # Remove oldest entry
            oldest_key = next(iter(self.memory_cache))
            del self.memory_cache[oldest_key]
            del self.cache_timestamps[oldest_key]
            
        self.memory_cache[key] = batch
        self.cache_timestamps[key] = time.time()
    
    async def _save_to_disk_cache(self, key: str, batch: torch.Tensor):
        """Save batch to disk cache."""
        try:
            cache_file = Path(self.config.cache_dir) / f"{key}.pkl"
            
            # Save asynchronously
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor() as executor:
                await loop.run_in_executor(
                    executor,
                    self._save_batch_sync,
                    cache_file,
                    batch
                )
                
        except Exception as e:
            self.logger.error(f"Error saving to disk cache: {e}")
    
    def _save_batch_sync(self, cache_file: Path, batch: torch.Tensor):
        """Save batch synchronously."""
        with open(cache_file, 'wb') as f:
            pickle.dump(batch, f)
    
    async def _load_from_disk_cache(self, key: str) -> Optional[torch.Tensor]:
        """Load batch from disk cache."""
        try:
            cache_file = Path(self.config.cache_dir) / f"{key}.pkl"
            
            if not cache_file.exists():
                return None
                
            # Check file age
            file_age = time.time() - cache_file.stat().st_mtime
            if file_age > self.config.cache_ttl:
                cache_file.unlink()  # Remove expired cache
                return None
                
            # Load asynchronously
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor() as executor:
                batch = await loop.run_in_executor(
                    executor,
                    self._load_batch_sync,
                    cache_file
                )
                
            return batch
            
        except Exception as e:
            self.logger.error(f"Error loading from disk cache: {e}")
            return None
    
    def _load_batch_sync(self, cache_file: Path) -> torch.Tensor:
        """Load batch synchronously."""
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0
        
        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate,
            'memory_cache_size': len(self.memory_cache),
            'total_requests': total_requests
        }
    
    def clear_cache(self):
        """Clear all caches."""
        self.memory_cache.clear()
        self.cache_timestamps.clear()
        
        if self.config.enable_disk_caching:
            cache_dir = Path(self.config.cache_dir)
            if cache_dir.exists():
                for cache_file in cache_dir.glob("*.pkl"):
                    cache_file.unlink()
                    
        self.logger.info("All caches cleared")

class LoadingStats:
    """Monitor data loading statistics."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.monitoring = False
        self.monitor_thread = None
        self.stats = {
            'batches_loaded': 0,
            'total_bytes_loaded': 0,
            'loading_time': 0.0,
            'avg_batch_time': 0.0,
            'throughput': 0.0
        }
        self.start_time = None
        
    def start_monitoring(self):
        """Start monitoring."""
        if not self.monitoring:
            self.monitoring = True
            self.start_time = time.time()
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
            self.logger.info("Loading statistics monitoring started")
    
    def stop_monitoring(self):
        """Stop monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        self.logger.info("Loading statistics monitoring stopped")
    
    def _monitor_loop(self):
        """Monitoring loop."""
        while self.monitoring:
            try:
                # Log statistics periodically
                self.logger.info(f"Loading stats: {self.stats}")
                time.sleep(5.0)  # Log every 5 seconds
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(5.0)
    
    def record_batch_loaded(self, batch_size: int, loading_time: float):
        """Record batch loading statistics."""
        self.stats['batches_loaded'] += 1
        self.stats['total_bytes_loaded'] += batch_size
        self.stats['loading_time'] += loading_time
        
        # Update averages
        if self.stats['batches_loaded'] > 0:
            self.stats['avg_batch_time'] = self.stats['loading_time'] / self.stats['batches_loaded']
            
        # Calculate throughput
        if self.stats['loading_time'] > 0:
            self.stats['throughput'] = self.stats['total_bytes_loaded'] / self.stats['loading_time']

class AsyncDataset(Dataset):
    """Async-compatible dataset wrapper."""
    
    def __init__(self, base_dataset: Dataset, config: AsyncDataConfig):
        self.base_dataset = base_dataset
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def __len__(self) -> int:
        return len(self.base_dataset)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.base_dataset[idx]
    
    async def get_batch_async(self, indices: List[int]) -> torch.Tensor:
        """Get batch asynchronously."""
        if self.config.enable_parallel_processing:
            # Process in thread pool
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor(max_workers=self.config.max_parallel_workers) as executor:
                batch_data = await asyncio.gather(*[
                    loop.run_in_executor(executor, self.__getitem__, idx)
                    for idx in indices
                ])
        else:
            batch_data = [self.__getitem__(idx) for idx in indices]
            
        return torch.stack(batch_data)

# Utility functions
@asynccontextmanager
async def async_data_context(config: AsyncDataConfig):
    """Context manager for async data loading."""
    loader = AsyncDataLoader(config)
    try:
        yield loader
    finally:
        if config.enable_loading_monitoring:
            loader.loading_stats.stop_monitoring()

def create_async_loader(dataset: Dataset, config: AsyncDataConfig) -> AsyncDataLoader:
    """Create async data loader for dataset."""
    return AsyncDataLoader(config)

async def load_data_async(dataset: Dataset, config: AsyncDataConfig) -> AsyncGenerator[torch.Tensor, None]:
    """Load data asynchronously."""
    async with async_data_context(config) as loader:
        async for batch in loader.load_dataset(dataset):
            yield batch






