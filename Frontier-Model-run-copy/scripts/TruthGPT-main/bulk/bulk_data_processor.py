#!/usr/bin/env python3
"""
Bulk Data Processor - Handles bulk data processing for optimization
Processes large datasets efficiently with memory optimization and parallel processing
"""

import torch
import torch.nn as nn
import torch.utils.data as data
from typing import Dict, Any, List, Optional, Union, Tuple, Iterator, Callable
from dataclasses import dataclass, field
import time
import json
import logging
import threading
import queue
import concurrent.futures
from pathlib import Path
import psutil
import gc
from collections import defaultdict, deque
import numpy as np
import pickle
import h5py
import zarr
from torch.utils.data import DataLoader, Dataset, IterableDataset
import multiprocessing as mp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class BulkDataConfig:
    """Configuration for bulk data processing."""
    # Processing settings
    batch_size: int = 32
    num_workers: int = 4
    prefetch_factor: int = 2
    pin_memory: bool = True
    
    # Memory management
    max_memory_gb: float = 8.0
    enable_memory_mapping: bool = True
    enable_compression: bool = True
    compression_level: int = 6
    
    # Data augmentation
    enable_data_augmentation: bool = True
    augmentation_probability: float = 0.3
    augmentation_strategies: List[str] = field(default_factory=lambda: [
        'noise_injection', 'sequence_permutation', 'token_masking'
    ])
    
    # Quality control
    enable_data_validation: bool = True
    validation_threshold: float = 0.95
    enable_outlier_detection: bool = True
    outlier_threshold: float = 3.0
    
    # Performance optimization
    enable_parallel_processing: bool = True
    enable_async_loading: bool = True
    enable_caching: bool = True
    cache_size_mb: int = 512
    
    # Monitoring
    enable_progress_tracking: bool = True
    progress_update_interval: int = 100
    enable_performance_monitoring: bool = True

class BulkDataset(Dataset):
    """Custom dataset for bulk data processing."""
    
    def __init__(self, data_path: str, config: BulkDataConfig, 
                 transform: Optional[Callable] = None):
        self.data_path = data_path
        self.config = config
        self.transform = transform
        self.data = self._load_data()
        self.length = len(self.data)
        
    def _load_data(self) -> List[Any]:
        """Load data from file."""
        try:
            if self.data_path.endswith('.json'):
                with open(self.data_path, 'r') as f:
                    return json.load(f)
            elif self.data_path.endswith('.pkl'):
                with open(self.data_path, 'rb') as f:
                    return pickle.load(f)
            elif self.data_path.endswith('.h5'):
                with h5py.File(self.data_path, 'r') as f:
                    return list(f['data'])
            elif self.data_path.endswith('.zarr'):
                store = zarr.open(self.data_path, mode='r')
                return list(store)
            else:
                # Assume text file
                with open(self.data_path, 'r', encoding='utf-8') as f:
                    return f.readlines()
        except Exception as e:
            logger.error(f"Failed to load data from {self.data_path}: {e}")
            return []
    
    def __len__(self) -> int:
        return self.length
    
    def __getitem__(self, idx: int) -> Any:
        """Get item at index."""
        if idx >= self.length:
            raise IndexError(f"Index {idx} out of range for dataset of size {self.length}")
        
        item = self.data[idx]
        
        if self.transform:
            item = self.transform(item)
        
        return item

class BulkDataProcessor:
    """Main bulk data processor for handling large datasets."""
    
    def __init__(self, config: BulkDataConfig):
        self.config = config
        self.data_cache = {}
        self.processing_stats = defaultdict(list)
        self.monitoring_thread = None
        self.stop_monitoring = threading.Event()
        
    def process_dataset(self, dataset: Dataset, 
                      model: Optional[nn.Module] = None,
                      processing_function: Optional[Callable] = None) -> Dict[str, Any]:
        """Process a dataset in bulk."""
        logger.info(f"🚀 Starting bulk processing of dataset with {len(dataset)} samples")
        
        if self.config.enable_performance_monitoring:
            self._start_monitoring()
        
        # Create data loader
        dataloader = self._create_dataloader(dataset)
        
        # Process data
        results = self._process_batches(dataloader, model, processing_function)
        
        if self.config.enable_performance_monitoring:
            self._stop_monitoring()
        
        # Generate processing report
        report = self._generate_processing_report(results)
        
        logger.info(f"✅ Bulk processing completed: {len(results)} batches processed")
        return report
    
    def _create_dataloader(self, dataset: Dataset) -> DataLoader:
        """Create optimized data loader."""
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            prefetch_factor=self.config.prefetch_factor,
            persistent_workers=True if self.config.num_workers > 0 else False
        )
    
    def _process_batches(self, dataloader: DataLoader, 
                        model: Optional[nn.Module],
                        processing_function: Optional[Callable]) -> List[Dict[str, Any]]:
        """Process data in batches."""
        results = []
        batch_count = 0
        
        for batch_idx, batch in enumerate(dataloader):
            try:
                start_time = time.time()
                
                # Process batch
                if processing_function:
                    batch_result = processing_function(batch)
                elif model:
                    batch_result = self._process_with_model(batch, model)
                else:
                    batch_result = self._process_batch_basic(batch)
                
                processing_time = time.time() - start_time
                
                # Add metadata
                batch_result.update({
                    'batch_idx': batch_idx,
                    'processing_time': processing_time,
                    'batch_size': len(batch) if isinstance(batch, (list, tuple)) else batch.size(0) if hasattr(batch, 'size') else 1
                })
                
                results.append(batch_result)
                batch_count += 1
                
                # Progress tracking
                if self.config.enable_progress_tracking and batch_count % self.config.progress_update_interval == 0:
                    logger.info(f"Processed {batch_count} batches")
                
                # Memory management
                if self.config.enable_caching:
                    self._manage_cache()
                
            except Exception as e:
                logger.error(f"Error processing batch {batch_idx}: {e}")
                results.append({
                    'batch_idx': batch_idx,
                    'error': str(e),
                    'processing_time': 0.0
                })
        
        return results
    
    def _process_with_model(self, batch: Any, model: nn.Module) -> Dict[str, Any]:
        """Process batch with model."""
        try:
            model.eval()
            with torch.no_grad():
                if isinstance(batch, (list, tuple)):
                    inputs = batch[0] if len(batch) > 0 else batch
                else:
                    inputs = batch
                
                # Ensure inputs are tensors
                if not isinstance(inputs, torch.Tensor):
                    inputs = torch.tensor(inputs)
                
                # Move to device if model is on GPU
                if next(model.parameters()).is_cuda and not inputs.is_cuda:
                    inputs = inputs.cuda()
                
                # Forward pass
                outputs = model(inputs)
                
                return {
                    'outputs': outputs.cpu().numpy() if outputs.is_cuda else outputs.numpy(),
                    'input_shape': inputs.shape,
                    'output_shape': outputs.shape,
                    'model_parameters': sum(p.numel() for p in model.parameters())
                }
        except Exception as e:
            return {'error': str(e)}
    
    def _process_batch_basic(self, batch: Any) -> Dict[str, Any]:
        """Basic batch processing without model."""
        try:
            if isinstance(batch, torch.Tensor):
                return {
                    'shape': batch.shape,
                    'dtype': str(batch.dtype),
                    'mean': batch.mean().item(),
                    'std': batch.std().item(),
                    'min': batch.min().item(),
                    'max': batch.max().item()
                }
            elif isinstance(batch, (list, tuple)):
                return {
                    'length': len(batch),
                    'type': type(batch[0]).__name__ if batch else 'empty'
                }
            else:
                return {
                    'type': type(batch).__name__,
                    'str_representation': str(batch)[:100]
                }
        except Exception as e:
            return {'error': str(e)}
    
    def _manage_cache(self):
        """Manage data cache to prevent memory overflow."""
        if len(self.data_cache) > self.config.cache_size_mb:
            # Remove oldest entries
            cache_items = list(self.data_cache.items())
            cache_items.sort(key=lambda x: x[1]['timestamp'])
            
            # Remove half of the cache
            for key, _ in cache_items[:len(cache_items)//2]:
                del self.data_cache[key]
    
    def _start_monitoring(self):
        """Start performance monitoring."""
        if self.monitoring_thread is None or not self.monitoring_thread.is_alive():
            self.stop_monitoring.clear()
            self.monitoring_thread = threading.Thread(target=self._monitor_performance)
            self.monitoring_thread.start()
    
    def _stop_monitoring(self):
        """Stop performance monitoring."""
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.stop_monitoring.set()
            self.monitoring_thread.join()
    
    def _monitor_performance(self):
        """Monitor system performance during processing."""
        while not self.stop_monitoring.is_set():
            try:
                memory_usage = psutil.virtual_memory().percent
                cpu_usage = psutil.cpu_percent()
                
                self.processing_stats['memory'].append(memory_usage)
                self.processing_stats['cpu'].append(cpu_usage)
                
                # Keep only recent stats
                if len(self.processing_stats['memory']) > 1000:
                    self.processing_stats['memory'] = self.processing_stats['memory'][-500:]
                    self.processing_stats['cpu'] = self.processing_stats['cpu'][-500:]
                
                time.sleep(0.5)
            except:
                break
    
    def _generate_processing_report(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate processing report."""
        successful_batches = [r for r in results if 'error' not in r]
        failed_batches = [r for r in results if 'error' in r]
        
        report = {
            'timestamp': time.time(),
            'config': self.config.__dict__,
            'summary': {
                'total_batches': len(results),
                'successful_batches': len(successful_batches),
                'failed_batches': len(failed_batches),
                'success_rate': len(successful_batches) / len(results) if results else 0,
                'average_processing_time': np.mean([r.get('processing_time', 0) for r in successful_batches]) if successful_batches else 0,
                'total_processing_time': sum(r.get('processing_time', 0) for r in results)
            },
            'performance_metrics': dict(self.processing_stats),
            'batch_results': results[:10] if len(results) > 10 else results  # Sample results
        }
        
        return report
    
    def save_processing_report(self, report: Dict[str, Any], filepath: str):
        """Save processing report to file."""
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"📊 Processing report saved to {filepath}")
    
    def load_dataset_from_file(self, filepath: str, 
                              transform: Optional[Callable] = None) -> BulkDataset:
        """Load dataset from file."""
        return BulkDataset(filepath, self.config, transform)
    
    def create_synthetic_dataset(self, num_samples: int, 
                                sample_shape: Tuple[int, ...],
                                data_type: str = 'float') -> BulkDataset:
        """Create synthetic dataset for testing."""
        # Create temporary data
        if data_type == 'float':
            data = [torch.randn(*sample_shape) for _ in range(num_samples)]
        elif data_type == 'int':
            data = [torch.randint(0, 100, sample_shape) for _ in range(num_samples)]
        else:
            data = [torch.zeros(sample_shape) for _ in range(num_samples)]
        
        # Save to temporary file
        temp_file = f"temp_dataset_{int(time.time())}.pkl"
        with open(temp_file, 'wb') as f:
            pickle.dump(data, f)
        
        return BulkDataset(temp_file, self.config)

class BulkDataAugmentation:
    """Data augmentation for bulk processing."""
    
    def __init__(self, config: BulkDataConfig):
        self.config = config
        self.augmentation_functions = {
            'noise_injection': self._add_noise,
            'sequence_permutation': self._permute_sequence,
            'token_masking': self._mask_tokens
        }
    
    def augment_batch(self, batch: torch.Tensor) -> torch.Tensor:
        """Augment a batch of data."""
        if not self.config.enable_data_augmentation:
            return batch
        
        augmented_batch = batch.clone()
        
        for strategy in self.config.augmentation_strategies:
            if strategy in self.augmentation_functions and np.random.random() < self.config.augmentation_probability:
                augmented_batch = self.augmentation_functions[strategy](augmented_batch)
        
        return augmented_batch
    
    def _add_noise(self, batch: torch.Tensor) -> torch.Tensor:
        """Add noise to batch."""
        noise = torch.randn_like(batch) * 0.1
        return batch + noise
    
    def _permute_sequence(self, batch: torch.Tensor) -> torch.Tensor:
        """Permute sequence dimensions."""
        if batch.dim() > 1:
            perm = torch.randperm(batch.size(1))
            return batch[:, perm]
        return batch
    
    def _mask_tokens(self, batch: torch.Tensor) -> torch.Tensor:
        """Mask random tokens."""
        mask_prob = 0.1
        mask = torch.rand_like(batch) < mask_prob
        batch = batch.clone()
        batch[mask] = 0
        return batch

def create_bulk_data_processor(config: Optional[Dict[str, Any]] = None) -> BulkDataProcessor:
    """Create a bulk data processor instance."""
    if config is None:
        config = {}
    
    data_config = BulkDataConfig(**config)
    return BulkDataProcessor(data_config)

def process_dataset_bulk(dataset: Dataset, 
                        config: Optional[Dict[str, Any]] = None,
                        model: Optional[nn.Module] = None,
                        processing_function: Optional[Callable] = None) -> Dict[str, Any]:
    """Convenience function for bulk dataset processing."""
    processor = create_bulk_data_processor(config)
    return processor.process_dataset(dataset, model, processing_function)

if __name__ == "__main__":
    print("🚀 Bulk Data Processor")
    print("=" * 40)
    
    # Example usage
    config = {
        'batch_size': 16,
        'num_workers': 2,
        'enable_parallel_processing': True,
        'enable_data_augmentation': True
    }
    
    processor = create_bulk_data_processor(config)
    print(f"✅ Bulk data processor created with batch size {processor.config.batch_size}")

