import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
import logging
import time
import threading
import asyncio
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import lru_cache, wraps
import pickle
import hashlib
import json
from pathlib import Path
from dataclasses import dataclass, field
from collections import defaultdict, deque
import gc
import psutil
import os
from datetime import datetime, timedelta
import warnings
import tracemalloc
from contextlib import contextmanager
import torch.cuda.amp as amp
from torch.cuda.amp import autocast, GradScaler
warnings.filterwarnings('ignore')

# Import our existing components
from custom_nn_modules import (
    FacebookContentAnalysisTransformer, MultiModalFacebookAnalyzer,
    TemporalEngagementPredictor, AdaptiveContentOptimizer, FacebookDiffusionUNet
)


@dataclass
class EnhancedPerformanceConfig:
    """Enhanced configuration for performance optimization"""
    # Caching
    enable_caching: bool = True
    cache_size: int = 50000  # Increased cache size
    cache_ttl_seconds: int = 7200  # 2 hours
    enable_distributed_cache: bool = False
    cache_persistence: bool = True
    
    # Parallel processing
    max_workers: int = min(8, os.cpu_count() or 4)
    use_multiprocessing: bool = True
    batch_processing: bool = True
    batch_size: int = 64  # Increased batch size
    enable_async_processing: bool = True
    
    # Memory management
    enable_memory_optimization: bool = True
    max_memory_usage_gb: float = 16.0  # Increased memory limit
    garbage_collection_threshold: float = 0.75
    enable_memory_profiling: bool = True
    memory_cleanup_interval: int = 300  # 5 minutes
    
    # GPU optimization
    enable_gpu_optimization: bool = True
    mixed_precision: bool = True
    gradient_checkpointing: bool = True
    memory_efficient_attention: bool = True
    enable_tensor_cores: bool = True
    cuda_graphs: bool = False  # For repeated operations
    
    # Monitoring and profiling
    enable_performance_monitoring: bool = True
    log_performance_metrics: bool = True
    profile_execution: bool = True
    enable_telemetry: bool = True
    metrics_export_interval: int = 60  # 1 minute
    
    # Advanced optimizations
    enable_model_compression: bool = False
    enable_quantization: bool = False
    enable_pruning: bool = False
    enable_kernel_fusion: bool = True


class EnhancedPerformanceCache:
    """Enhanced high-performance caching system with advanced features"""
    
    def __init__(self, config: EnhancedPerformanceConfig):
        self.config = config
        self.max_size = config.cache_size
        self.ttl_seconds = config.cache_ttl_seconds
        self.cache = {}
        self.access_times = {}
        self.access_counts = defaultdict(int)
        self.lock = threading.RLock()
        
        # Performance metrics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.total_requests = 0
        
        # Memory tracking
        self.cache_memory_usage = 0
        self.max_memory_usage = 0
        
        # Start cleanup thread
        self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self.cleanup_thread.start()
        
        # Load persistent cache if enabled
        if config.cache_persistence:
            self._load_persistent_cache()
    
    def _load_persistent_cache(self):
        """Load cache from disk"""
        cache_file = Path("cache/performance_cache.pkl")
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)
                    self.cache = data.get('cache', {})
                    self.access_times = data.get('access_times', {})
                    self.access_counts = data.get('access_counts', defaultdict(int))
                logging.info(f"Loaded persistent cache with {len(self.cache)} entries")
            except Exception as e:
                logging.warning(f"Failed to load persistent cache: {e}")
    
    def _save_persistent_cache(self):
        """Save cache to disk"""
        if not self.config.cache_persistence:
            return
            
        cache_file = Path("cache/performance_cache.pkl")
        cache_file.parent.mkdir(exist_ok=True)
        
        try:
            data = {
                'cache': self.cache,
                'access_times': self.access_times,
                'access_counts': dict(self.access_counts)
            }
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            logging.warning(f"Failed to save persistent cache: {e}")
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache with enhanced statistics"""
        self.total_requests += 1
        
        with self.lock:
            if key in self.cache:
                current_time = time.time()
                if current_time - self.access_times[key] <= self.ttl_seconds:
                    self.hits += 1
                    self.access_times[key] = current_time
                    self.access_counts[key] += 1
                    return self.cache[key]
                else:
                    # Expired, remove it
                    del self.cache[key]
                    del self.access_times[key]
                    del self.access_counts[key]
                    self.evictions += 1
        
        self.misses += 1
        return None
    
    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> bool:
        """Set item in cache with memory tracking"""
        ttl = ttl_seconds or self.ttl_seconds
        
        with self.lock:
            # Check if we need to evict items
            if len(self.cache) >= self.max_size:
                self._evict_least_valuable()
            
            # Estimate memory usage
            try:
                item_size = len(pickle.dumps(value))
                self.cache_memory_usage += item_size
                self.max_memory_usage = max(self.max_memory_usage, self.cache_memory_usage)
            except:
                item_size = 0
            
            self.cache[key] = value
            self.access_times[key] = time.time()
            self.access_counts[key] = 1
            
            return True
    
    def _evict_least_valuable(self):
        """Evict items based on value (access count and recency)"""
        if not self.cache:
            return
            
        # Calculate value score for each item
        current_time = time.time()
        item_scores = {}
        
        for key in self.cache:
            age = current_time - self.access_times[key]
            access_count = self.access_counts[key]
            # Value = access_count / (age + 1) - lower is worse
            item_scores[key] = access_count / (age + 1)
        
        # Find worst items to evict
        items_to_evict = len(self.cache) - self.max_size + 1
        worst_items = sorted(item_scores.items(), key=lambda x: x[1])[:items_to_evict]
        
        for key, _ in worst_items:
            try:
                item_size = len(pickle.dumps(self.cache[key]))
                self.cache_memory_usage -= item_size
            except:
                pass
            
            del self.cache[key]
            del self.access_times[key]
            del self.access_counts[key]
            self.evictions += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        with self.lock:
            hit_rate = self.hits / max(self.total_requests, 1)
            memory_usage_mb = self.cache_memory_usage / (1024 * 1024)
            max_memory_mb = self.max_memory_usage / (1024 * 1024)
            
            return {
                'hits': self.hits,
                'misses': self.misses,
                'evictions': self.evictions,
                'total_requests': self.total_requests,
                'hit_rate': hit_rate,
                'cache_size': len(self.cache),
                'memory_usage_mb': memory_usage_mb,
                'max_memory_usage_mb': max_memory_mb,
                'efficiency': hit_rate * (1 - memory_usage_mb / max(16.0, 1))
            }
    
    def _cleanup_loop(self):
        """Background cleanup loop with enhanced logic"""
        while True:
            time.sleep(60)  # Check every minute
            self._cleanup_expired()
            
            # Save cache periodically
            if self.config.cache_persistence:
                self._save_persistent_cache()
    
    def _cleanup_expired(self):
        """Remove expired entries with memory cleanup"""
        current_time = time.time()
        expired_keys = []
        
        with self.lock:
            for key, access_time in self.access_times.items():
                if current_time - access_time > self.ttl_seconds:
                    expired_keys.append(key)
            
            # Remove expired items
            for key in expired_keys:
                try:
                    item_size = len(pickle.dumps(self.cache[key]))
                    self.cache_memory_usage -= item_size
                except:
                    pass
                
                del self.cache[key]
                del self.access_times[key]
                del self.access_counts[key]
                self.evictions += 1


class MemoryManager:
    """Advanced memory management system"""
    
    def __init__(self, config: EnhancedPerformanceConfig):
        self.config = config
        self.memory_threshold = config.max_memory_usage_gb * 1024 * 1024 * 1024  # Convert to bytes
        self.last_cleanup = time.time()
        self.cleanup_interval = config.memory_cleanup_interval
        
        # Memory profiling
        if config.enable_memory_profiling:
            tracemalloc.start()
            self.snapshot = tracemalloc.take_snapshot()
    
    def check_memory_usage(self) -> Dict[str, float]:
        """Check current memory usage"""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        # System memory
        system_memory = psutil.virtual_memory()
        
        # GPU memory if available
        gpu_memory = self._get_gpu_memory_info()
        
        return {
            'process_rss_gb': memory_info.rss / (1024**3),
            'process_vms_gb': memory_info.vms / (1024**3),
            'system_available_gb': system_memory.available / (1024**3),
            'system_used_gb': system_memory.used / (1024**3),
            'system_percent': system_memory.percent,
            'gpu_memory_used_gb': gpu_memory.get('used_gb', 0),
            'gpu_memory_total_gb': gpu_memory.get('total_gb', 0)
        }
    
    def _get_gpu_memory_info(self) -> Dict[str, float]:
        """Get GPU memory information"""
        if not torch.cuda.is_available():
            return {}
        
        try:
            gpu_memory = torch.cuda.memory_stats()
            allocated = gpu_memory.get('allocated_bytes.all.current', 0)
            reserved = gpu_memory.get('reserved_bytes.all.current', 0)
            total = torch.cuda.get_device_properties(0).total_memory
            
            return {
                'used_gb': allocated / (1024**3),
                'reserved_gb': reserved / (1024**3),
                'total_gb': total / (1024**3)
            }
        except:
            return {}
    
    def should_cleanup_memory(self) -> bool:
        """Check if memory cleanup is needed"""
        current_time = time.time()
        memory_info = self.check_memory_usage()
        
        # Check if we're over threshold or cleanup interval has passed
        over_threshold = memory_info['process_rss_gb'] > self.config.max_memory_usage_gb
        cleanup_due = current_time - self.last_cleanup > self.cleanup_interval
        
        return over_threshold or cleanup_due
    
    def cleanup_memory(self) -> Dict[str, Any]:
        """Perform memory cleanup"""
        if not self.should_cleanup_memory():
            return {'status': 'no_cleanup_needed'}
        
        # Take memory snapshot before cleanup
        if self.config.enable_memory_profiling:
            before_snapshot = tracemalloc.take_snapshot()
        
        # Force garbage collection
        collected = gc.collect()
        
        # Clear PyTorch cache if GPU memory is high
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Take memory snapshot after cleanup
        if self.config.enable_memory_profiling:
            after_snapshot = tracemalloc.take_snapshot()
            top_stats = after_snapshot.compare_to(before_snapshot, 'lineno')
        
        self.last_cleanup = time.time()
        
        # Get memory info after cleanup
        after_memory = self.check_memory_usage()
        
        return {
            'status': 'cleanup_completed',
            'garbage_collected': collected,
            'memory_after_cleanup_gb': after_memory['process_rss_gb'],
            'top_memory_changes': top_stats[:5] if self.config.enable_memory_profiling else []
        }


class GPUOptimizer:
    """Advanced GPU optimization system"""
    
    def __init__(self, config: EnhancedPerformanceConfig):
        self.config = config
        self.scaler = GradScaler() if config.mixed_precision else None
        self.cuda_graphs = {}
        
        if config.enable_gpu_optimization and torch.cuda.is_available():
            self._setup_gpu_optimizations()
    
    def _setup_gpu_optimizations(self):
        """Setup GPU optimizations"""
        # Enable memory efficient attention if available
        if self.config.memory_efficient_attention:
            try:
                torch.backends.cuda.enable_flash_sdp(True)
                torch.backends.cuda.enable_mem_efficient_sdp(True)
                torch.backends.cuda.enable_math_sdp(True)
            except:
                pass
        
        # Enable tensor cores for mixed precision
        if self.config.enable_tensor_cores and self.config.mixed_precision:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    
    @contextmanager
    def autocast_context(self):
        """Context manager for automatic mixed precision"""
        if self.config.mixed_precision and self.scaler:
            with autocast():
                yield
        else:
            yield
    
    def optimize_model(self, model: nn.Module) -> nn.Module:
        """Apply GPU optimizations to model"""
        if not self.config.enable_gpu_optimization:
            return model
        
        # Enable gradient checkpointing
        if self.config.gradient_checkpointing:
            model.gradient_checkpointing_enable()
        
        # Move to GPU
        if torch.cuda.is_available():
            model = model.cuda()
            
            # Compile model if available (PyTorch 2.0+)
            try:
                if hasattr(torch, 'compile'):
                    model = torch.compile(model, mode='max-autotune')
            except:
                pass
        
        return model
    
    def create_cuda_graph(self, name: str, func: Callable, *args, **kwargs):
        """Create CUDA graph for repeated operations"""
        if not self.config.cuda_graphs or not torch.cuda.is_available():
            return func(*args, **kwargs)
        
        if name not in self.cuda_graphs:
            # Warmup
            for _ in range(3):
                func(*args, **kwargs)
            
            torch.cuda.synchronize()
            
            # Create graph
            with torch.cuda.graph(self.cuda_graphs[name]):
                result = func(*args, **kwargs)
            
            return result
        else:
            # Replay graph
            with torch.cuda.graph(self.cuda_graphs[name]):
                return func(*args, **kwargs)


class PerformanceProfiler:
    """Advanced performance profiling system"""
    
    def __init__(self, config: EnhancedPerformanceConfig):
        self.config = config
        self.profiles = {}
        self.current_profile = None
        self.start_times = {}
        
        # Performance metrics
        self.operation_times = defaultdict(list)
        self.memory_usage = defaultdict(list)
        self.gpu_usage = defaultdict(list)
    
    @contextmanager
    def profile_operation(self, operation_name: str):
        """Context manager for profiling operations"""
        if not self.config.profile_execution:
            yield
            return
        
        start_time = time.time()
        start_memory = self._get_memory_usage()
        start_gpu = self._get_gpu_usage()
        
        try:
            yield
        finally:
            end_time = time.time()
            end_memory = self._get_memory_usage()
            end_gpu = self._get_gpu_usage()
            
            duration = end_time - start_time
            memory_diff = end_memory - start_memory
            gpu_diff = end_gpu - start_gpu
            
            self.operation_times[operation_name].append(duration)
            self.memory_usage[operation_name].append(memory_diff)
            self.gpu_usage[operation_name].append(gpu_diff)
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        except:
            return 0.0
    
    def _get_gpu_usage(self) -> float:
        """Get current GPU memory usage in MB"""
        if not torch.cuda.is_available():
            return 0.0
        
        try:
            return torch.cuda.memory_allocated() / (1024 * 1024)
        except:
            return 0.0
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        report = {
            'operation_times': {},
            'memory_usage': {},
            'gpu_usage': {},
            'summary': {}
        }
        
        for op_name in self.operation_times:
            times = self.operation_times[op_name]
            memory = self.memory_usage[op_name]
            gpu = self.gpu_usage[op_name]
            
            report['operation_times'][op_name] = {
                'count': len(times),
                'mean': np.mean(times),
                'std': np.std(times),
                'min': np.min(times),
                'max': np.max(times),
                'total': np.sum(times)
            }
            
            report['memory_usage'][op_name] = {
                'mean': np.mean(memory),
                'max': np.max(memory)
            }
            
            report['gpu_usage'][op_name] = {
                'mean': np.mean(gpu),
                'max': np.max(gpu)
            }
        
        # Summary statistics
        all_times = [t for times in self.operation_times.values() for t in times]
        if all_times:
            report['summary'] = {
                'total_operations': sum(len(times) for times in self.operation_times.values()),
                'total_time': sum(all_times),
                'average_time': np.mean(all_times),
                'slowest_operation': max(all_times),
                'fastest_operation': min(all_times)
            }
        
        return report


class EnhancedPerformanceOptimizationEngine:
    """Enhanced performance optimization engine with advanced features"""
    
    def __init__(self, config: EnhancedPerformanceConfig):
        self.config = config
        self.cache = EnhancedPerformanceCache(config)
        self.memory_manager = MemoryManager(config)
        self.gpu_optimizer = GPUOptimizer(config)
        self.profiler = PerformanceProfiler(config)
        
        # Thread pool for parallel processing
        self.thread_pool = ThreadPoolExecutor(max_workers=config.max_workers)
        self.process_pool = ProcessPoolExecutor(max_workers=config.max_workers) if config.use_multiprocessing else None
        
        # Performance monitoring
        self.metrics = {}
        self.last_metrics_export = time.time()
        
        # Start monitoring thread
        if config.enable_performance_monitoring:
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitoring_thread.start()
    
    def optimize_content_batch(self, contents: List[str], content_type: str = "Post") -> List[Dict[str, Any]]:
        """Optimize a batch of content with enhanced performance"""
        if not contents:
            return []
        
        # Check if we should cleanup memory
        if self.memory_manager.should_cleanup_memory():
            cleanup_result = self.memory_manager.cleanup_memory()
            logging.info(f"Memory cleanup: {cleanup_result}")
        
        # Process in batches
        batch_size = self.config.batch_size
        results = []
        
        with self.profiler.profile_operation("batch_optimization"):
            for i in range(0, len(contents), batch_size):
                batch = contents[i:i + batch_size]
                
                # Check cache first
                cached_results = self._get_cached_batch(batch, content_type)
                uncached_batch = [content for j, content in enumerate(batch) if cached_results[j] is None]
                
                if uncached_batch:
                    # Process uncached items
                    batch_results = self._process_batch(uncached_batch, content_type)
                    
                    # Cache results
                    for content, result in zip(uncached_batch, batch_results):
                        cache_key = self._generate_cache_key(content, content_type)
                        self.cache.set(cache_key, result)
                    
                    # Combine cached and new results
                    for j, content in enumerate(batch):
                        if cached_results[j] is not None:
                            results.append(cached_results[j])
                        else:
                            results.append(batch_results[uncached_batch.index(content)])
                else:
                    results.extend(cached_results)
        
        return results
    
    def _get_cached_batch(self, contents: List[str], content_type: str) -> List[Optional[Dict[str, Any]]]:
        """Get cached results for a batch of content"""
        cached_results = []
        
        for content in contents:
            cache_key = self._generate_cache_key(content, content_type)
            cached_result = self.cache.get(cache_key)
            cached_results.append(cached_result)
        
        return cached_results
    
    def _process_batch(self, contents: List[str], content_type: str) -> List[Dict[str, Any]]:
        """Process a batch of uncached content"""
        if self.config.enable_async_processing:
            return self._process_batch_async(contents, content_type)
        else:
            return self._process_batch_sync(contents, content_type)
    
    def _process_batch_sync(self, contents: List[str], content_type: str) -> List[Dict[str, Any]]:
        """Process batch synchronously"""
        results = []
        
        for content in contents:
            # Simulate content optimization (replace with actual logic)
            result = {
                'content': content,
                'engagement_score': np.random.uniform(0.1, 0.9),
                'viral_potential': np.random.uniform(0.1, 0.9),
                'optimization_suggestions': [
                    'Use more engaging language',
                    'Include relevant hashtags',
                    'Add call-to-action'
                ],
                'processing_time': time.time()
            }
            results.append(result)
        
        return results
    
    def _process_batch_async(self, contents: List[str], content_type: str) -> List[Dict[str, Any]]:
        """Process batch asynchronously using thread pool"""
        futures = []
        
        for content in contents:
            future = self.thread_pool.submit(self._process_single_content, content, content_type)
            futures.append(future)
        
        # Collect results
        results = []
        for future in futures:
            try:
                result = future.result(timeout=30)  # 30 second timeout
                results.append(result)
            except Exception as e:
                logging.error(f"Error processing content: {e}")
                results.append({
                    'error': str(e),
                    'processing_time': time.time()
                })
        
        return results
    
    def _process_single_content(self, content: str, content_type: str) -> Dict[str, Any]:
        """Process a single content item"""
        # Simulate content optimization (replace with actual logic)
        time.sleep(0.1)  # Simulate processing time
        
        return {
            'content': content,
            'engagement_score': np.random.uniform(0.1, 0.9),
            'viral_potential': np.random.uniform(0.1, 0.9),
            'optimization_suggestions': [
                'Use more engaging language',
                'Include relevant hashtags',
                'Add call-to-action'
            ],
            'processing_time': time.time()
        }
    
    def _generate_cache_key(self, content: str, content_type: str) -> str:
        """Generate cache key for content"""
        content_hash = hashlib.md5(f"{content}:{content_type}".encode()).hexdigest()
        return f"content_opt_{content_hash}"
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        stats = {
            'cache_stats': self.cache.get_stats(),
            'memory_info': self.memory_manager.check_memory_usage(),
            'performance_report': self.profiler.get_performance_report(),
            'gpu_info': self.gpu_optimizer._get_gpu_memory_info() if torch.cuda.is_available() else {},
            'system_info': {
                'cpu_count': os.cpu_count(),
                'python_version': f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
                'pytorch_version': torch.__version__,
                'cuda_available': torch.cuda.is_available(),
                'cuda_version': torch.version.cuda if torch.cuda.is_available() else None
            }
        }
        
        return stats
    
    def _monitoring_loop(self):
        """Background monitoring loop"""
        while True:
            try:
                # Export metrics if enabled
                current_time = time.time()
                if current_time - self.last_metrics_export > self.config.metrics_export_interval:
                    self._export_metrics()
                    self.last_metrics_export = current_time
                
                # Check system health
                self._check_system_health()
                
                time.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logging.error(f"Error in monitoring loop: {e}")
                time.sleep(30)  # Wait longer on error
    
    def _export_metrics(self):
        """Export performance metrics"""
        if not self.config.enable_telemetry:
            return
        
        try:
            metrics = self.get_system_stats()
            metrics['timestamp'] = datetime.now().isoformat()
            
            # Save to file
            metrics_file = Path("logs/performance_metrics.json")
            metrics_file.parent.mkdir(exist_ok=True)
            
            with open(metrics_file, 'a') as f:
                f.write(json.dumps(metrics) + '\n')
                
        except Exception as e:
            logging.error(f"Failed to export metrics: {e}")
    
    def _check_system_health(self):
        """Check system health and log warnings"""
        memory_info = self.memory_manager.check_memory_usage()
        
        # Check memory usage
        if memory_info['process_rss_gb'] > self.config.max_memory_usage_gb * 0.9:
            logging.warning(f"High memory usage: {memory_info['process_rss_gb']:.2f} GB")
        
        # Check system memory
        if memory_info['system_percent'] > 90:
            logging.warning(f"High system memory usage: {memory_info['system_percent']:.1f}%")
        
        # Check GPU memory
        if memory_info['gpu_memory_used_gb'] > 0:
            gpu_usage_percent = (memory_info['gpu_memory_used_gb'] / max(memory_info['gpu_memory_total_gb'], 1)) * 100
            if gpu_usage_percent > 90:
                logging.warning(f"High GPU memory usage: {gpu_usage_percent:.1f}%")
    
    def cleanup(self):
        """Cleanup resources"""
        if self.thread_pool:
            self.thread_pool.shutdown(wait=True)
        
        if self.process_pool:
            self.process_pool.shutdown(wait=True)
        
        # Save cache
        if self.config.cache_persistence:
            self.cache._save_persistent_cache()


# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create configuration
    config = EnhancedPerformanceConfig(
        enable_caching=True,
        enable_memory_optimization=True,
        enable_gpu_optimization=True,
        enable_performance_monitoring=True,
        profile_execution=True
    )
    
    # Initialize engine
    engine = EnhancedPerformanceOptimizationEngine(config)
    
    try:
        # Test batch processing
        test_contents = [
            "Check out our amazing new product!",
            "Don't miss this incredible opportunity",
            "Transform your life with our solution",
            "Join thousands of satisfied customers"
        ]
        
        print("🚀 Testing Enhanced Performance Engine...")
        
        # Process content
        results = engine.optimize_content_batch(test_contents, "Post")
        
        print(f"✅ Processed {len(results)} content items")
        for i, result in enumerate(results):
            print(f"  {i+1}. Engagement: {result['engagement_score']:.3f}, Viral: {result['viral_potential']:.3f}")
        
        # Get system stats
        stats = engine.get_system_stats()
        print(f"\n📊 Cache Hit Rate: {stats['cache_stats']['hit_rate']:.3f}")
        print(f"💾 Memory Usage: {stats['memory_info']['process_rss_gb']:.2f} GB")
        
        # Test memory cleanup
        print("\n🧹 Testing memory cleanup...")
        cleanup_result = engine.memory_manager.cleanup_memory()
        print(f"Cleanup result: {cleanup_result}")
        
    finally:
        # Cleanup
        engine.cleanup()
        print("\n✨ Enhanced Performance Engine test completed!")
