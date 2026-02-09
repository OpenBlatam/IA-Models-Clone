"""
🚀 ASYNC BATCH PROCESSOR v6.0.0 - ULTRA-FAST BATCHING
======================================================

High-performance async batch processing utilities:
- ⚡ Parallel batch processing with configurable concurrency
- 🔄 Smart batching with dynamic batch sizes
- 📊 Performance monitoring and optimization
- 🧵 Worker pool management
- 💾 Memory-efficient processing
- 🎯 Adaptive batch sizing
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Callable, TypeVar, Generic, Tuple
import uuid
import gc
import psutil
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing

# Configure logging
logger = logging.getLogger(__name__)

# =============================================================================
# 🎯 BATCH PROCESSING TYPES
# =============================================================================

class BatchStrategy(Enum):
    """Batch processing strategies."""
    SEQUENTIAL = "sequential"      # Process one by one
    PARALLEL = "parallel"         # Process all in parallel
    CHUNKED = "chunked"          # Process in chunks
    ADAPTIVE = "adaptive"        # Adaptive batch sizing
    STREAMING = "streaming"      # Stream processing

class BatchStatus(Enum):
    """Batch processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

# =============================================================================
# 🎯 BATCH CONFIGURATION
# =============================================================================

@dataclass
class BatchConfig:
    """Configuration for batch processing."""
    strategy: BatchStrategy = BatchStrategy.ADAPTIVE
    initial_batch_size: int = 100
    max_batch_size: int = 1000
    min_batch_size: int = 10
    max_concurrency: int = min(32, multiprocessing.cpu_count() * 2)
    timeout: float = 300.0
    retry_attempts: int = 3
    enable_memory_optimization: bool = True
    enable_performance_monitoring: bool = True
    memory_threshold_mb: int = 1024
    adaptive_threshold: float = 0.8  # 80% success rate for adaptive sizing
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'strategy': self.strategy.value,
            'initial_batch_size': self.initial_batch_size,
            'max_batch_size': self.max_batch_size,
            'min_batch_size': self.min_batch_size,
            'max_concurrency': self.max_concurrency,
            'timeout': self.timeout,
            'retry_attempts': self.retry_attempts,
            'enable_memory_optimization': self.enable_memory_optimization,
            'enable_performance_monitoring': self.enable_performance_monitoring,
            'memory_threshold_mb': self.memory_threshold_mb,
            'adaptive_threshold': self.adaptive_threshold
        }

# =============================================================================
# 🎯 BATCH PROCESSOR INTERFACES
# =============================================================================

T = TypeVar('T')
R = TypeVar('R')

class BatchProcessor(ABC, Generic[T, R]):
    """Abstract batch processor interface."""
    
    @abstractmethod
    async def process_item(self, item: T) -> R:
        """Process a single item."""
        pass
    
    @abstractmethod
    async def process_batch(self, items: List[T]) -> List[R]:
        """Process a batch of items."""
        pass

class AsyncBatchProcessor:
    """High-performance async batch processor."""
    
    def __init__(self, config: BatchConfig):
        self.config = config
        self.processor_id = str(uuid.uuid4())
        self.start_time = time.time()
        
        # Performance tracking
        self.total_items_processed = 0
        self.total_batches_processed = 0
        self.total_processing_time = 0.0
        self.successful_items = 0
        self.failed_items = 0
        
        # Batch history for adaptive sizing
        self.batch_history: List[Dict[str, Any]] = []
        
        # Worker pools
        self.thread_pool: Optional[ThreadPoolExecutor] = None
        self.process_pool: Optional[ProcessPoolExecutor] = None
        
        # Initialize worker pools
        self._initialize_worker_pools()
        
        logger.info(f"🚀 Async Batch Processor initialized with ID: {self.processor_id}")
    
    def _initialize_worker_pools(self) -> None:
        """Initialize worker thread and process pools."""
        try:
            self.thread_pool = ThreadPoolExecutor(
                max_workers=self.config.max_concurrency,
                thread_name_prefix="AsyncBatch"
            )
            logger.debug(f"🚀 Thread pool initialized with {self.config.max_concurrency} workers")
        except Exception as e:
            logger.warning(f"⚠️ Failed to initialize thread pool: {e}")
        
        try:
            self.process_pool = ProcessPoolExecutor(
                max_workers=min(self.config.max_concurrency // 2, multiprocessing.cpu_count())
            )
            logger.debug(f"🚀 Process pool initialized")
        except Exception as e:
            logger.warning(f"⚠️ Failed to initialize process pool: {e}")
    
    async def process_items(
        self, 
        items: List[T], 
        processor: Callable[[T], R],
        strategy: Optional[BatchStrategy] = None
    ) -> List[R]:
        """Process items using the specified strategy."""
        if not items:
            return []
        
        strategy = strategy or self.config.strategy
        start_time = time.time()
        
        try:
            if strategy == BatchStrategy.SEQUENTIAL:
                results = await self._process_sequential(items, processor)
            elif strategy == BatchStrategy.PARALLEL:
                results = await self._process_parallel(items, processor)
            elif strategy == BatchStrategy.CHUNKED:
                results = await self._process_chunked(items, processor)
            elif strategy == BatchStrategy.ADAPTIVE:
                results = await self._process_adaptive(items, processor)
            elif strategy == BatchStrategy.STREAMING:
                results = await self._process_streaming(items, processor)
            else:
                raise ValueError(f"Unknown batch strategy: {strategy}")
            
            # Update performance metrics
            processing_time = time.time() - start_time
            self.total_processing_time += processing_time
            self.total_items_processed += len(items)
            self.total_batches_processed += 1
            
            # Memory optimization
            if self.config.enable_memory_optimization:
                self._optimize_memory()
            
            logger.debug(f"🚀 Processed {len(items)} items in {processing_time:.3f}s using {strategy.value} strategy")
            return results
            
        except Exception as e:
            logger.error(f"❌ Batch processing failed: {e}")
            raise
    
    async def _process_sequential(self, items: List[T], processor: Callable[[T], R]) -> List[R]:
        """Process items sequentially."""
        results = []
        for item in items:
            try:
                if asyncio.iscoroutinefunction(processor):
                    result = await processor(item)
                else:
                    result = processor(item)
                results.append(result)
                self.successful_items += 1
            except Exception as e:
                logger.warning(f"⚠️ Item processing failed: {e}")
                results.append(None)
                self.failed_items += 1
        
        return results
    
    async def _process_parallel(self, items: List[T], processor: Callable[[T], R]) -> List[R]:
        """Process all items in parallel."""
        if asyncio.iscoroutinefunction(processor):
            # Async processor - use asyncio.gather
            tasks = [processor(item) for item in items]
            results = await asyncio.gather(*tasks, return_exceptions=True)
        else:
            # Sync processor - use thread pool
            if self.thread_pool:
                loop = asyncio.get_event_loop()
                tasks = [
                    loop.run_in_executor(self.thread_pool, processor, item)
                    for item in items
                ]
                results = await asyncio.gather(*tasks, return_exceptions=True)
            else:
                # Fallback to sequential
                return await self._process_sequential(items, processor)
        
        # Count successes and failures
        for result in results:
            if isinstance(result, Exception):
                self.failed_items += 1
            else:
                self.successful_items += 1
        
        return results
    
    async def _process_chunked(self, items: List[T], processor: Callable[[T], R]) -> List[R]:
        """Process items in chunks."""
        batch_size = self.config.initial_batch_size
        results = []
        
        for i in range(0, len(items), batch_size):
            chunk = items[i:i + batch_size]
            chunk_results = await self._process_parallel(chunk, processor)
            results.extend(chunk_results)
            
            # Memory optimization between chunks
            if self.config.enable_memory_optimization:
                self._optimize_memory()
        
        return results
    
    async def _process_adaptive(self, items: List[T], processor: Callable[[T], R]) -> List[R]:
        """Process items with adaptive batch sizing."""
        current_batch_size = self.config.initial_batch_size
        results = []
        
        for i in range(0, len(items), current_batch_size):
            chunk = items[i:i + current_batch_size]
            
            # Process chunk
            start_time = time.time()
            chunk_results = await self._process_parallel(chunk, processor)
            chunk_time = time.time() - start_time
            
            # Record batch performance
            success_rate = sum(1 for r in chunk_results if not isinstance(r, Exception)) / len(chunk_results)
            self.batch_history.append({
                'batch_size': current_batch_size,
                'processing_time': chunk_time,
                'success_rate': success_rate,
                'timestamp': time.time()
            })
            
            # Keep only recent history
            if len(self.batch_history) > 20:
                self.batch_history = self.batch_history[-20:]
            
            # Adjust batch size based on performance
            current_batch_size = self._adjust_batch_size(current_batch_size, success_rate, chunk_time)
            
            results.extend(chunk_results)
            
            # Memory optimization between batches
            if self.config.enable_memory_optimization:
                self._optimize_memory()
        
        return results
    
    async def _process_streaming(self, items: List[T], processor: Callable[[T], R]) -> List[R]:
        """Process items in streaming fashion."""
        results = []
        semaphore = asyncio.Semaphore(self.config.max_concurrency)
        
        async def process_with_semaphore(item: T) -> R:
            async with semaphore:
                if asyncio.iscoroutinefunction(processor):
                    return await processor(item)
                else:
                    if self.thread_pool:
                        loop = asyncio.get_event_loop()
                        return await loop.run_in_executor(self.thread_pool, processor, item)
                    else:
                        return processor(item)
        
        # Create tasks for all items
        tasks = [process_with_semaphore(item) for item in items]
        
        # Process as they complete
        for completed_task in asyncio.as_completed(tasks):
            try:
                result = await completed_task
                results.append(result)
                self.successful_items += 1
            except Exception as e:
                logger.warning(f"⚠️ Streaming item processing failed: {e}")
                results.append(None)
                self.failed_items += 1
        
        return results
    
    def _adjust_batch_size(self, current_size: int, success_rate: float, processing_time: float) -> int:
        """Adjust batch size based on performance metrics."""
        if success_rate >= self.config.adaptive_threshold:
            # Good performance - increase batch size
            new_size = min(current_size * 1.2, self.config.max_batch_size)
        else:
            # Poor performance - decrease batch size
            new_size = max(current_size * 0.8, self.config.min_batch_size)
        
        return int(new_size)
    
    def _optimize_memory(self) -> None:
        """Optimize memory usage."""
        current_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        if current_memory > self.config.memory_threshold_mb:
            # Force garbage collection
            collected = gc.collect()
            logger.debug(f"🧹 Memory optimization: collected {collected} objects")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        uptime = time.time() - self.start_time
        
        return {
            'processor_id': self.processor_id,
            'uptime_seconds': uptime,
            'total_items_processed': self.total_items_processed,
            'total_batches_processed': self.total_batches_processed,
            'total_processing_time': self.total_processing_time,
            'successful_items': self.successful_items,
            'failed_items': self.failed_items,
            'success_rate': (
                self.successful_items / max(1, self.total_items_processed) * 100
            ),
            'items_per_second': (
                self.total_items_processed / uptime if uptime > 0 else 0.0
            ),
            'avg_batch_time': (
                self.total_processing_time / max(1, self.total_batches_processed)
            ),
            'current_batch_size': self.config.initial_batch_size,
            'batch_history_length': len(self.batch_history)
        }
    
    async def shutdown(self) -> None:
        """Shutdown the batch processor."""
        logger.info("🔄 Shutting down Async Batch Processor...")
        
        # Shutdown worker pools
        if self.thread_pool:
            self.thread_pool.shutdown(wait=True)
        
        if self.process_pool:
            self.process_pool.shutdown(wait=True)
        
        logger.info("✅ Async Batch Processor shutdown complete")

# =============================================================================
# 🚀 SPECIALIZED BATCH PROCESSORS
# =============================================================================

class DataBatchProcessor(AsyncBatchProcessor):
    """Specialized batch processor for data processing tasks."""
    
    async def process_data_batch(
        self, 
        data_items: List[Dict[str, Any]], 
        transform_func: Callable[[Dict[str, Any]], Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Process data items in batches."""
        return await self.process_items(data_items, transform_func)
    
    async def process_with_validation(
        self, 
        items: List[T], 
        processor: Callable[[T], R],
        validator: Callable[[R], bool]
    ) -> List[R]:
        """Process items with validation."""
        results = await self.process_items(items, processor)
        
        # Validate results
        valid_results = []
        for result in results:
            if result is not None and validator(result):
                valid_results.append(result)
            else:
                self.failed_items += 1
        
        return valid_results

class MLBatchProcessor(AsyncBatchProcessor):
    """Specialized batch processor for machine learning tasks."""
    
    async def process_ml_batch(
        self, 
        features: List[Any], 
        model: Callable[[Any], Any],
        batch_size: Optional[int] = None
    ) -> List[Any]:
        """Process ML features in optimized batches."""
        if batch_size:
            self.config.initial_batch_size = batch_size
        
        return await self.process_items(features, model, BatchStrategy.CHUNKED)
    
    async def process_with_batching(
        self, 
        items: List[T], 
        processor: Callable[[T], R],
        optimal_batch_size: int
    ) -> List[R]:
        """Process with optimal batch size for ML models."""
        original_batch_size = self.config.initial_batch_size
        self.config.initial_batch_size = optimal_batch_size
        
        try:
            return await self.process_items(items, processor, BatchStrategy.CHUNKED)
        finally:
            self.config.initial_batch_size = original_batch_size

# =============================================================================
# 🚀 FACTORY FUNCTIONS
# =============================================================================

def create_batch_processor(config: Optional[BatchConfig] = None) -> AsyncBatchProcessor:
    """Create a batch processor."""
    if config is None:
        config = BatchConfig()
    return AsyncBatchProcessor(config)

def create_data_batch_processor(config: Optional[BatchConfig] = None) -> DataBatchProcessor:
    """Create a data batch processor."""
    if config is None:
        config = BatchConfig()
    return DataBatchProcessor(config)

def create_ml_batch_processor(config: Optional[BatchConfig] = None) -> MLBatchProcessor:
    """Create an ML batch processor."""
    if config is None:
        config = BatchConfig()
    return MLBatchProcessor(config)

def create_optimized_batch_config(**kwargs) -> BatchConfig:
    """Create an optimized batch configuration."""
    config = BatchConfig()
    
    # Apply optimizations based on system capabilities
    cpu_count = multiprocessing.cpu_count()
    if cpu_count > 16:
        config.max_concurrency = 64
        config.max_batch_size = 2000
    elif cpu_count > 8:
        config.max_concurrency = 32
        config.max_batch_size = 1000
    else:
        config.max_concurrency = 16
        config.max_batch_size = 500
    
    # Apply custom settings
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return config

# =============================================================================
# 🌟 EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "BatchStrategy",
    "BatchStatus",
    
    # Configuration
    "BatchConfig",
    
    # Main processors
    "AsyncBatchProcessor",
    "DataBatchProcessor",
    "MLBatchProcessor",
    
    # Factory functions
    "create_batch_processor",
    "create_data_batch_processor",
    "create_ml_batch_processor",
    "create_optimized_batch_config"
]


