#!/usr/bin/env python3
"""
Memory Optimization Module for SEO Evaluation System
Advanced memory management with dynamic allocation and efficient caching
"""

import torch
import torch.nn as nn
import torch.cuda.amp as amp
import gc
import psutil
import os
import logging
import time
import asyncio
from typing import Dict, Any, Optional, List, Tuple, Union, Callable
from dataclasses import dataclass, field
from contextlib import contextmanager
import warnings
from pathlib import Path
import json
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
from functools import lru_cache, wraps
import weakref
from collections import OrderedDict

warnings.filterwarnings("ignore")

@dataclass
class MemoryConfig:
    """Configuration for memory optimization."""
    # Memory Management
    max_memory_usage: float = 0.8  # Maximum memory usage as fraction of total
    enable_dynamic_batching: bool = True
    enable_memory_pooling: bool = True
    enable_gradient_checkpointing: bool = True
    enable_memory_efficient_attention: bool = True
    enable_flash_attention: bool = True
    
    # Caching
    enable_model_caching: bool = True
    enable_data_caching: bool = True
    cache_size: int = 1000
    cache_ttl: int = 3600  # Time to live in seconds
    
    # Monitoring
    enable_memory_monitoring: bool = True
    memory_check_interval: float = 1.0  # seconds
    enable_memory_profiling: bool = True
    
    # Optimization
    enable_memory_compression: bool = True
    enable_memory_defragmentation: bool = True
    enable_automatic_gc: bool = True
    gc_threshold: float = 0.7  # Trigger GC when memory usage exceeds this threshold

class MemoryOptimizer:
    """Advanced memory optimization for SEO evaluation system."""
    
    def __init__(self, config: MemoryConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.memory_monitor = MemoryMonitor(config)
        self.cache_manager = CacheManager(config)
        self.dynamic_batcher = DynamicBatcher(config)
        
        # Initialize memory optimization
        self._setup_memory_optimization()
        
    def _setup_memory_optimization(self):
        """Setup memory optimization features."""
        if torch.cuda.is_available():
            # Enable memory pooling
            if self.config.enable_memory_pooling:
                torch.cuda.empty_cache()
                torch.cuda.set_per_process_memory_fraction(self.config.max_memory_usage)
                
            # Enable memory efficient attention
            if self.config.enable_memory_efficient_attention:
                try:
                    from transformers.models.bert.modeling_bert import BertSelfAttention
                    BertSelfAttention._use_memory_efficient_attention = True
                except ImportError:
                    self.logger.warning("Memory efficient attention not available")
                    
            # Enable flash attention
            if self.config.enable_flash_attention:
                try:
                    from transformers.models.bert.modeling_bert import BertSelfAttention
                    BertSelfAttention._use_flash_attention = True
                except ImportError:
                    self.logger.warning("Flash attention not available")
    
    @contextmanager
    def memory_context(self, model: nn.Module = None):
        """Context manager for memory optimization."""
        try:
            # Pre-optimization
            if model and self.config.enable_gradient_checkpointing:
                model.gradient_checkpointing_enable()
                
            # Start memory monitoring
            if self.config.enable_memory_monitoring:
                self.memory_monitor.start_monitoring()
                
            yield
            
        finally:
            # Post-optimization cleanup
            if self.config.enable_automatic_gc:
                self._cleanup_memory()
                
            # Stop memory monitoring
            if self.config.enable_memory_monitoring:
                self.memory_monitor.stop_monitoring()
    
    def _cleanup_memory(self):
        """Clean up memory and trigger garbage collection."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        if self.config.enable_memory_defragmentation:
            gc.collect()
            
        # Check if we need to trigger GC
        current_memory = self.memory_monitor.get_memory_usage()
        if current_memory > self.config.gc_threshold:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def optimize_model_memory(self, model: nn.Module) -> nn.Module:
        """Optimize model memory usage."""
        if self.config.enable_gradient_checkpointing:
            model.gradient_checkpointing_enable()
            
        if self.config.enable_memory_compression:
            model = self._compress_model(model)
            
        return model
    
    def _compress_model(self, model: nn.Module) -> nn.Module:
        """Compress model to reduce memory usage."""
        # Convert to half precision if not already
        if not next(model.parameters()).dtype == torch.float16:
            model = model.half()
            
        return model
    
    def get_optimal_batch_size(self, model: nn.Module, sample_input: torch.Tensor) -> int:
        """Dynamically determine optimal batch size based on available memory."""
        return self.dynamic_batcher.calculate_optimal_batch_size(model, sample_input)
    
    def cache_model(self, model: nn.Module, key: str):
        """Cache model for reuse."""
        if self.config.enable_model_caching:
            self.cache_manager.cache_model(model, key)
    
    def get_cached_model(self, key: str) -> Optional[nn.Module]:
        """Retrieve cached model."""
        if self.config.enable_model_caching:
            return self.cache_manager.get_model(key)
        return None

class MemoryMonitor:
    """Monitor memory usage in real-time."""
    
    def __init__(self, config: MemoryConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.monitoring = False
        self.monitor_thread = None
        self.memory_history = []
        self.max_history_size = 1000
        
    def start_monitoring(self):
        """Start memory monitoring."""
        if not self.monitoring:
            self.monitoring = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
            self.logger.info("Memory monitoring started")
    
    def stop_monitoring(self):
        """Stop memory monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        self.logger.info("Memory monitoring stopped")
    
    def _monitor_loop(self):
        """Memory monitoring loop."""
        while self.monitoring:
            try:
                memory_info = self.get_memory_usage()
                self.memory_history.append(memory_info)
                
                # Keep history size manageable
                if len(self.memory_history) > self.max_history_size:
                    self.memory_history.pop(0)
                    
                # Check for memory issues
                if memory_info['usage_percent'] > 0.9:
                    self.logger.warning(f"High memory usage: {memory_info['usage_percent']:.2%}")
                    
                time.sleep(self.config.memory_check_interval)
                
            except Exception as e:
                self.logger.error(f"Error in memory monitoring: {e}")
                time.sleep(self.config.memory_check_interval)
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage."""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        # System memory
        system_memory = psutil.virtual_memory()
        
        # GPU memory if available
        gpu_memory = {}
        if torch.cuda.is_available():
            gpu_memory = {
                'gpu_allocated': torch.cuda.memory_allocated() / 1024**3,  # GB
                'gpu_reserved': torch.cuda.memory_reserved() / 1024**3,   # GB
                'gpu_max_allocated': torch.cuda.max_memory_allocated() / 1024**3  # GB
            }
        
        return {
            'rss': memory_info.rss / 1024**3,  # GB
            'vms': memory_info.vms / 1024**3,  # GB
            'usage_percent': system_memory.percent / 100,
            'available': system_memory.available / 1024**3,  # GB
            **gpu_memory
        }
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics."""
        if not self.memory_history:
            return {}
            
        usage_percentages = [h['usage_percent'] for h in self.memory_history]
        gpu_allocated = [h.get('gpu_allocated', 0) for h in self.memory_history]
        
        return {
            'current_usage': self.memory_history[-1],
            'avg_usage_percent': sum(usage_percentages) / len(usage_percentages),
            'max_usage_percent': max(usage_percentages),
            'avg_gpu_allocated': sum(gpu_allocated) / len(gpu_allocated) if gpu_allocated else 0,
            'max_gpu_allocated': max(gpu_allocated) if gpu_allocated else 0,
            'history_length': len(self.memory_history)
        }

class CacheManager:
    """Manage model and data caching."""
    
    def __init__(self, config: MemoryConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.model_cache = OrderedDict()
        self.data_cache = OrderedDict()
        self.cache_timestamps = {}
        
    def cache_model(self, model: nn.Module, key: str):
        """Cache a model."""
        if len(self.model_cache) >= self.config.cache_size:
            # Remove oldest entry
            oldest_key = next(iter(self.model_cache))
            del self.model_cache[oldest_key]
            del self.cache_timestamps[oldest_key]
            
        self.model_cache[key] = model
        self.cache_timestamps[key] = time.time()
        self.logger.debug(f"Cached model: {key}")
    
    def get_model(self, key: str) -> Optional[nn.Module]:
        """Retrieve a cached model."""
        if key in self.model_cache:
            # Check TTL
            if time.time() - self.cache_timestamps[key] > self.config.cache_ttl:
                del self.model_cache[key]
                del self.cache_timestamps[key]
                return None
                
            # Move to end (LRU)
            model = self.model_cache.pop(key)
            self.model_cache[key] = model
            self.cache_timestamps[key] = time.time()
            
            self.logger.debug(f"Retrieved cached model: {key}")
            return model
            
        return None
    
    def cache_data(self, data: Any, key: str):
        """Cache data."""
        if len(self.data_cache) >= self.config.cache_size:
            # Remove oldest entry
            oldest_key = next(iter(self.data_cache))
            del self.data_cache[oldest_key]
            del self.cache_timestamps[oldest_key]
            
        self.data_cache[key] = data
        self.cache_timestamps[key] = time.time()
        self.logger.debug(f"Cached data: {key}")
    
    def get_data(self, key: str) -> Optional[Any]:
        """Retrieve cached data."""
        if key in self.data_cache:
            # Check TTL
            if time.time() - self.cache_timestamps[key] > self.config.cache_ttl:
                del self.data_cache[key]
                del self.cache_timestamps[key]
                return None
                
            # Move to end (LRU)
            data = self.data_cache.pop(key)
            self.data_cache[key] = data
            self.cache_timestamps[key] = time.time()
            
            self.logger.debug(f"Retrieved cached data: {key}")
            return data
            
        return None
    
    def clear_cache(self):
        """Clear all caches."""
        self.model_cache.clear()
        self.data_cache.clear()
        self.cache_timestamps.clear()
        self.logger.info("All caches cleared")

class DynamicBatcher:
    """Dynamic batch size calculation based on available memory."""
    
    def __init__(self, config: MemoryConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def calculate_optimal_batch_size(self, model: nn.Module, sample_input: torch.Tensor) -> int:
        """Calculate optimal batch size based on available memory."""
        if not self.config.enable_dynamic_batching:
            return 32  # Default batch size
            
        try:
            # Get available memory
            memory_info = psutil.virtual_memory()
            available_memory = memory_info.available * self.config.max_memory_usage
            
            # Estimate memory per sample
            with torch.no_grad():
                sample_output = model(sample_input.unsqueeze(0))
                sample_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
                
            # Calculate optimal batch size
            if sample_memory > 0:
                optimal_batch_size = int(available_memory / sample_memory)
            else:
                # Fallback calculation
                optimal_batch_size = int(available_memory / (1024**3)) * 8  # Rough estimate
                
            # Apply reasonable bounds
            optimal_batch_size = max(1, min(optimal_batch_size, 512))
            
            self.logger.info(f"Calculated optimal batch size: {optimal_batch_size}")
            return optimal_batch_size
            
        except Exception as e:
            self.logger.warning(f"Error calculating optimal batch size: {e}")
            return 32  # Fallback to default

# Utility functions
def memory_optimized(func: Callable) -> Callable:
    """Decorator for memory optimization."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        optimizer = MemoryOptimizer(MemoryConfig())
        with optimizer.memory_context():
            return func(*args, **kwargs)
    return wrapper

def monitor_memory_usage(func: Callable) -> Callable:
    """Decorator to monitor memory usage of a function."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        monitor = MemoryMonitor(MemoryConfig())
        start_memory = monitor.get_memory_usage()
        
        result = func(*args, **kwargs)
        
        end_memory = monitor.get_memory_usage()
        memory_diff = {
            'rss_diff': end_memory['rss'] - start_memory['rss'],
            'vms_diff': end_memory['vms'] - start_memory['vms'],
            'usage_diff': end_memory['usage_percent'] - start_memory['usage_percent']
        }
        
        logging.getLogger(__name__).info(f"Memory usage for {func.__name__}: {memory_diff}")
        return result
    return wrapper






