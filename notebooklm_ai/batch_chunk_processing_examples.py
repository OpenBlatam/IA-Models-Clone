from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

# Constants
BUFFER_SIZE = 1024

import asyncio
import time
import logging
import json
import pickle
import gzip
import sqlite3
import threading
from typing import List, Dict, Any, Optional, Iterator, AsyncIterator, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from contextlib import asynccontextmanager
from collections import defaultdict, deque
import weakref
import gc
from pathlib import Path
import hashlib
from datetime import datetime, timedelta
import signal
import sys
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from functools import partial
import psutil
import tracemalloc
from typing import Any, List, Dict, Optional
"""
Batch and Chunk Processing for Large Target Lists

This module provides comprehensive examples for batch and chunk processing of large target lists
to manage resource utilization effectively. It includes memory management, progress tracking,
error handling, and performance optimization techniques.

Key Features:
- Intelligent chunking strategies for large datasets
- Memory-efficient processing with generators
- Progress tracking and reporting
- Error handling and recovery
- Resource utilization monitoring
- Parallel processing with controlled concurrency
- Result aggregation and persistence
"""


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ChunkStrategy(Enum):
    """Chunking strategies for different use cases"""
    FIXED_SIZE = "fixed_size"
    MEMORY_BASED = "memory_based"
    TIME_BASED = "time_based"
    ADAPTIVE = "adaptive"
    WEIGHTED = "weighted"


class ProcessingMode(Enum):
    """Processing modes for batch operations"""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    STREAMING = "streaming"
    HYBRID = "hybrid"


@dataclass
class ChunkConfig:
    """Configuration for chunk processing"""
    chunk_size: int = 1000
    max_memory_mb: int = 512
    max_processing_time: float = 300.0  # 5 minutes
    max_concurrent_chunks: int = 4
    enable_memory_monitoring: bool = True
    enable_progress_tracking: bool = True
    enable_error_recovery: bool = True
    enable_result_persistence: bool = True
    result_format: str = "json"  # json, pickle, sqlite
    compression_enabled: bool = False
    cleanup_interval: int = 10  # chunks


@dataclass
class ProcessingStats:
    """Statistics for batch processing"""
    total_targets: int = 0
    processed_targets: int = 0
    successful_targets: int = 0
    failed_targets: int = 0
    skipped_targets: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    total_processing_time: float = 0.0
    avg_processing_time: float = 0.0
    memory_peak_mb: float = 0.0
    memory_current_mb: float = 0.0
    chunks_processed: int = 0
    chunks_failed: int = 0
    errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert stats to dictionary"""
        return {
            "total_targets": self.total_targets,
            "processed_targets": self.processed_targets,
            "successful_targets": self.successful_targets,
            "failed_targets": self.failed_targets,
            "skipped_targets": self.skipped_targets,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "total_processing_time": self.total_processing_time,
            "avg_processing_time": self.avg_processing_time,
            "memory_peak_mb": self.memory_peak_mb,
            "memory_current_mb": self.memory_current_mb,
            "chunks_processed": self.chunks_processed,
            "chunks_failed": self.chunks_failed,
            "error_count": len(self.errors),
            "success_rate": self.successful_targets / self.total_targets if self.total_targets > 0 else 0
        }


class MemoryMonitor:
    """Memory usage monitoring and management"""
    
    def __init__(self, max_memory_mb: int = 512):
        
    """__init__ function."""
self.max_memory_mb = max_memory_mb
        self.peak_memory_mb = 0.0
        self.current_memory_mb = 0.0
        self.memory_history: List[Tuple[datetime, float]] = []
        self.monitoring_enabled = True
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            self.current_memory_mb = memory_mb
            
            if memory_mb > self.peak_memory_mb:
                self.peak_memory_mb = memory_mb
            
            if self.monitoring_enabled:
                self.memory_history.append((datetime.now(), memory_mb))
            
            return memory_mb
        except Exception as e:
            logger.warning(f"Failed to get memory usage: {e}")
            return 0.0
    
    def is_memory_limit_exceeded(self) -> bool:
        """Check if memory limit is exceeded"""
        return self.get_memory_usage() > self.max_memory_mb
    
    def force_garbage_collection(self) -> Any:
        """Force garbage collection"""
        try:
            collected = gc.collect()
            logger.debug(f"Garbage collection freed {collected} objects")
        except Exception as e:
            logger.warning(f"Garbage collection failed: {e}")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics"""
        return {
            "current_mb": self.current_memory_mb,
            "peak_mb": self.peak_memory_mb,
            "limit_mb": self.max_memory_mb,
            "usage_percent": (self.current_memory_mb / self.max_memory_mb) * 100,
            "history_count": len(self.memory_history)
        }


class ProgressTracker:
    """Progress tracking and reporting"""
    
    def __init__(self, total_items: int, enable_tracking: bool = True):
        
    """__init__ function."""
self.total_items = total_items
        self.processed_items = 0
        self.successful_items = 0
        self.failed_items = 0
        self.skipped_items = 0
        self.start_time = datetime.now()
        self.last_update_time = self.start_time
        self.enable_tracking = enable_tracking
        self.progress_callbacks: List[Callable] = []
        self._lock = threading.Lock()
    
    def update(self, successful: bool = True, skipped: bool = False):
        """Update progress"""
        if not self.enable_tracking:
            return
        
        with self._lock:
            self.processed_items += 1
            
            if skipped:
                self.skipped_items += 1
            elif successful:
                self.successful_items += 1
            else:
                self.failed_items += 1
            
            self.last_update_time = datetime.now()
            
            # Call progress callbacks
            for callback in self.progress_callbacks:
                try:
                    callback(self.get_progress())
                except Exception as e:
                    logger.warning(f"Progress callback error: {e}")
    
    def get_progress(self) -> Dict[str, Any]:
        """Get current progress information"""
        elapsed_time = (datetime.now() - self.start_time).total_seconds()
        
        if self.processed_items > 0:
            avg_time_per_item = elapsed_time / self.processed_items
            estimated_remaining = avg_time_per_item * (self.total_items - self.processed_items)
        else:
            avg_time_per_item = 0
            estimated_remaining = 0
        
        return {
            "total_items": self.total_items,
            "processed_items": self.processed_items,
            "successful_items": self.successful_items,
            "failed_items": self.failed_items,
            "skipped_items": self.skipped_items,
            "progress_percent": (self.processed_items / self.total_items) * 100,
            "elapsed_time": elapsed_time,
            "estimated_remaining": estimated_remaining,
            "avg_time_per_item": avg_time_per_item,
            "success_rate": self.successful_items / self.processed_items if self.processed_items > 0 else 0
        }
    
    def add_callback(self, callback: Callable):
        """Add progress callback"""
        self.progress_callbacks.append(callback)
    
    def log_progress(self, interval: int = 100):
        """Log progress at intervals"""
        if self.processed_items % interval == 0:
            progress = self.get_progress()
            logger.info(f"Progress: {progress['progress_percent']:.1f}% "
                       f"({progress['processed_items']}/{progress['total_items']}) "
                       f"Success: {progress['success_rate']:.1%}")


class ChunkProcessor:
    """Intelligent chunking and processing of large datasets"""
    
    def __init__(self, config: ChunkConfig):
        
    """__init__ function."""
self.config = config
        self.memory_monitor = MemoryMonitor(config.max_memory_mb)
        self.stats = ProcessingStats()
        self._chunk_cache: Dict[str, List] = {}
        self._result_cache: Dict[str, Any] = {}
        self._error_cache: Dict[str, List[str]] = defaultdict(list)
    
    def create_chunks(self, targets: List[Any], strategy: ChunkStrategy = ChunkStrategy.FIXED_SIZE) -> Iterator[List[Any]]:
        """Create chunks from target list based on strategy"""
        if not targets:
            return
        
        if strategy == ChunkStrategy.FIXED_SIZE:
            yield from self._fixed_size_chunks(targets)
        elif strategy == ChunkStrategy.MEMORY_BASED:
            yield from self._memory_based_chunks(targets)
        elif strategy == ChunkStrategy.TIME_BASED:
            yield from self._time_based_chunks(targets)
        elif strategy == ChunkStrategy.ADAPTIVE:
            yield from self._adaptive_chunks(targets)
        elif strategy == ChunkStrategy.WEIGHTED:
            yield from self._weighted_chunks(targets)
        else:
            raise ValueError(f"Unknown chunk strategy: {strategy}")
    
    def _fixed_size_chunks(self, targets: List[Any]) -> Iterator[List[Any]]:
        """Create fixed-size chunks"""
        for i in range(0, len(targets), self.config.chunk_size):
            chunk = targets[i:i + self.config.chunk_size]
            yield chunk
    
    def _memory_based_chunks(self, targets: List[Any]) -> Iterator[List[Any]]:
        """Create chunks based on memory usage"""
        current_chunk = []
        current_memory = 0
        
        for target in targets:
            # Estimate memory usage (simplified)
            target_memory = self._estimate_target_memory(target)
            
            if current_memory + target_memory > self.config.max_memory_mb * 1024 * 1024:
                if current_chunk:
                    yield current_chunk
                    current_chunk = []
                    current_memory = 0
                    self.memory_monitor.force_garbage_collection()
            
            current_chunk.append(target)
            current_memory += target_memory
        
        if current_chunk:
            yield current_chunk
    
    def _time_based_chunks(self, targets: List[Any]) -> Iterator[List[Any]]:
        """Create chunks based on estimated processing time"""
        current_chunk = []
        estimated_time = 0
        
        for target in targets:
            target_time = self._estimate_target_time(target)
            
            if estimated_time + target_time > self.config.max_processing_time:
                if current_chunk:
                    yield current_chunk
                    current_chunk = []
                    estimated_time = 0
            
            current_chunk.append(target)
            estimated_time += target_time
        
        if current_chunk:
            yield current_chunk
    
    def _adaptive_chunks(self, targets: List[Any]) -> Iterator[List[Any]]:
        """Create adaptive chunks based on multiple factors"""
        current_chunk = []
        current_memory = 0
        estimated_time = 0
        
        for target in targets:
            target_memory = self._estimate_target_memory(target)
            target_time = self._estimate_target_time(target)
            
            # Check multiple constraints
            memory_exceeded = current_memory + target_memory > self.config.max_memory_mb * 1024 * 1024
            time_exceeded = estimated_time + target_time > self.config.max_processing_time
            size_exceeded = len(current_chunk) >= self.config.chunk_size
            
            if (memory_exceeded or time_exceeded or size_exceeded) and current_chunk:
                yield current_chunk
                current_chunk = []
                current_memory = 0
                estimated_time = 0
                self.memory_monitor.force_garbage_collection()
            
            current_chunk.append(target)
            current_memory += target_memory
            estimated_time += target_time
        
        if current_chunk:
            yield current_chunk
    
    def _weighted_chunks(self, targets: List[Any]) -> Iterator[List[Any]]:
        """Create weighted chunks based on target complexity"""
        current_chunk = []
        current_weight = 0
        max_weight = self.config.chunk_size * 10  # Weight threshold
        
        for target in targets:
            target_weight = self._calculate_target_weight(target)
            
            if current_weight + target_weight > max_weight and current_chunk:
                yield current_chunk
                current_chunk = []
                current_weight = 0
            
            current_chunk.append(target)
            current_weight += target_weight
        
        if current_chunk:
            yield current_chunk
    
    def _estimate_target_memory(self, target: Any) -> int:
        """Estimate memory usage for a target"""
        try:
            # Simple estimation based on target type and size
            if isinstance(target, str):
                return len(target.encode('utf-8'))
            elif isinstance(target, dict):
                return sum(len(str(v).encode('utf-8')) for v in target.values())
            elif isinstance(target, list):
                return sum(self._estimate_target_memory(item) for item in target)
            else:
                return len(str(target).encode('utf-8'))
        except Exception:
            return 1024  # Default estimate
    
    def _estimate_target_time(self, target: Any) -> float:
        """Estimate processing time for a target"""
        try:
            # Simple estimation based on target complexity
            if isinstance(target, dict) and 'complexity' in target:
                return target['complexity'] * 0.1
            elif isinstance(target, str):
                return len(target) * 0.001
            else:
                return 0.1  # Default estimate
        except Exception:
            return 0.1
    
    def _calculate_target_weight(self, target: Any) -> int:
        """Calculate weight for a target based on complexity"""
        try:
            weight = 1
            
            if isinstance(target, dict):
                weight += len(target)
                if 'priority' in target:
                    weight += target['priority']
            
            elif isinstance(target, str):
                weight += len(target) // 100
            
            elif isinstance(target, list):
                weight += len(target)
            
            return weight
        except Exception:
            return 1


class BatchProcessor:
    """High-level batch processing with resource management"""
    
    def __init__(self, config: ChunkConfig):
        
    """__init__ function."""
self.config = config
        self.chunk_processor = ChunkProcessor(config)
        self.memory_monitor = self.chunk_processor.memory_monitor
        self.stats = self.chunk_processor.stats
        self.progress_tracker: Optional[ProgressTracker] = None
        self.result_storage = ResultStorage(config)
        self._processing_semaphore = asyncio.Semaphore(config.max_concurrent_chunks)
        self._stop_processing = False
    
    async def process_targets(self, targets: List[Any], 
                            processor_func: Callable,
                            mode: ProcessingMode = ProcessingMode.PARALLEL,
                            strategy: ChunkStrategy = ChunkStrategy.ADAPTIVE) -> List[Any]:
        """Process targets in batches"""
        if not targets:
            return []
        
        # Initialize tracking
        self.stats.total_targets = len(targets)
        self.stats.start_time = datetime.now()
        self.progress_tracker = ProgressTracker(len(targets), self.config.enable_progress_tracking)
        
        # Add progress logging
        self.progress_tracker.add_callback(lambda p: self._log_progress(p))
        
        logger.info(f"Starting batch processing of {len(targets)} targets")
        logger.info(f"Mode: {mode.value}, Strategy: {strategy.value}")
        
        try:
            if mode == ProcessingMode.SEQUENTIAL:
                results = await self._process_sequential(targets, processor_func, strategy)
            elif mode == ProcessingMode.PARALLEL:
                results = await self._process_parallel(targets, processor_func, strategy)
            elif mode == ProcessingMode.STREAMING:
                results = await self._process_streaming(targets, processor_func, strategy)
            elif mode == ProcessingMode.HYBRID:
                results = await self._process_hybrid(targets, processor_func, strategy)
            else:
                raise ValueError(f"Unknown processing mode: {mode}")
            
            return results
            
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            self.stats.errors.append(str(e))
            raise
        finally:
            self._finalize_processing()
    
    async def _process_sequential(self, targets: List[Any], 
                                processor_func: Callable, 
                                strategy: ChunkStrategy) -> List[Any]:
        """Process targets sequentially"""
        results = []
        chunks = list(self.chunk_processor.create_chunks(targets, strategy))
        
        for i, chunk in enumerate(chunks):
            if self._stop_processing:
                break
            
            logger.info(f"Processing chunk {i+1}/{len(chunks)} ({len(chunk)} targets)")
            
            try:
                chunk_results = await self._process_chunk(chunk, processor_func)
                results.extend(chunk_results)
                self.stats.chunks_processed += 1
                
                # Update progress
                self.progress_tracker.update(successful=True)
                
                # Persist results
                if self.config.enable_result_persistence:
                    await self.result_storage.save_chunk_results(chunk_results, i)
                
            except Exception as e:
                logger.error(f"Chunk {i+1} processing failed: {e}")
                self.stats.chunks_failed += 1
                self.stats.errors.append(f"Chunk {i+1}: {str(e)}")
                
                if self.config.enable_error_recovery:
                    # Continue with next chunk
                    continue
                else:
                    raise
        
        return results
    
    async def _process_parallel(self, targets: List[Any], 
                              processor_func: Callable, 
                              strategy: ChunkStrategy) -> List[Any]:
        """Process targets in parallel"""
        chunks = list(self.chunk_processor.create_chunks(targets, strategy))
        tasks = []
        
        for i, chunk in enumerate(chunks):
            task = asyncio.create_task(
                self._process_chunk_with_semaphore(chunk, processor_func, i)
            )
            tasks.append(task)
        
        # Wait for all tasks to complete
        chunk_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Combine results
        results = []
        for i, chunk_result in enumerate(chunk_results):
            if isinstance(chunk_result, Exception):
                logger.error(f"Chunk {i+1} failed: {chunk_result}")
                self.stats.chunks_failed += 1
                self.stats.errors.append(f"Chunk {i+1}: {str(chunk_result)}")
            else:
                results.extend(chunk_result)
                self.stats.chunks_processed += 1
        
        return results
    
    async def _process_streaming(self, targets: List[Any], 
                               processor_func: Callable, 
                               strategy: ChunkStrategy) -> List[Any]:
        """Process targets in streaming mode"""
        results = []
        chunk_generator = self.chunk_processor.create_chunks(targets, strategy)
        
        async for chunk in self._async_chunk_generator(chunk_generator):
            if self._stop_processing:
                break
            
            try:
                chunk_results = await self._process_chunk(chunk, processor_func)
                results.extend(chunk_results)
                self.stats.chunks_processed += 1
                
                # Update progress
                self.progress_tracker.update(successful=True)
                
                # Persist results immediately
                if self.config.enable_result_persistence:
                    await self.result_storage.save_chunk_results(chunk_results)
                
            except Exception as e:
                logger.error(f"Streaming chunk processing failed: {e}")
                self.stats.chunks_failed += 1
                self.stats.errors.append(str(e))
                
                if not self.config.enable_error_recovery:
                    raise
        
        return results
    
    async def _process_hybrid(self, targets: List[Any], 
                            processor_func: Callable, 
                            strategy: ChunkStrategy) -> List[Any]:
        """Process targets using hybrid approach"""
        # Use parallel processing for large chunks, sequential for small ones
        chunks = list(self.chunk_processor.create_chunks(targets, strategy))
        results = []
        
        for i, chunk in enumerate(chunks):
            if self._stop_processing:
                break
            
            # Choose processing method based on chunk size
            if len(chunk) > self.config.chunk_size // 2:
                # Large chunk - process in parallel
                chunk_results = await self._process_chunk_parallel(chunk, processor_func)
            else:
                # Small chunk - process sequentially
                chunk_results = await self._process_chunk(chunk, processor_func)
            
            results.extend(chunk_results)
            self.stats.chunks_processed += 1
            
            # Update progress
            self.progress_tracker.update(successful=True)
        
        return results
    
    async def _process_chunk_with_semaphore(self, chunk: List[Any], 
                                          processor_func: Callable, 
                                          chunk_index: int) -> List[Any]:
        """Process chunk with semaphore control"""
        async with self._processing_semaphore:
            return await self._process_chunk(chunk, processor_func)
    
    async def _process_chunk(self, chunk: List[Any], processor_func: Callable) -> List[Any]:
        """Process a single chunk"""
        results = []
        
        for target in chunk:
            if self._stop_processing:
                break
            
            try:
                # Check memory limits
                if self.config.enable_memory_monitoring and self.memory_monitor.is_memory_limit_exceeded():
                    logger.warning("Memory limit exceeded, forcing garbage collection")
                    self.memory_monitor.force_garbage_collection()
                
                # Process target
                if asyncio.iscoroutinefunction(processor_func):
                    result = await processor_func(target)
                else:
                    result = processor_func(target)
                
                results.append(result)
                self.stats.successful_targets += 1
                self.progress_tracker.update(successful=True)
                
            except Exception as e:
                logger.error(f"Target processing failed: {e}")
                self.stats.failed_targets += 1
                self.progress_tracker.update(successful=False)
                
                if not self.config.enable_error_recovery:
                    raise
        
        return results
    
    async def _process_chunk_parallel(self, chunk: List[Any], processor_func: Callable) -> List[Any]:
        """Process chunk in parallel"""
        tasks = []
        
        for target in chunk:
            if asyncio.iscoroutinefunction(processor_func):
                task = asyncio.create_task(processor_func(target))
            else:
                # For sync functions, run in thread pool
                loop = asyncio.get_event_loop()
                task = loop.run_in_executor(None, processor_func, target)
            
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                self.stats.failed_targets += 1
                self.progress_tracker.update(successful=False)
            else:
                processed_results.append(result)
                self.stats.successful_targets += 1
                self.progress_tracker.update(successful=True)
        
        return processed_results
    
    async def _async_chunk_generator(self, chunk_generator: Iterator) -> AsyncIterator[List[Any]]:
        """Convert sync generator to async generator"""
        for chunk in chunk_generator:
            yield chunk
            await asyncio.sleep(0)  # Allow other tasks to run
    
    def _log_progress(self, progress: Dict[str, Any]):
        """Log progress information"""
        logger.info(f"Progress: {progress['progress_percent']:.1f}% "
                   f"({progress['processed_items']}/{progress['total_items']}) "
                   f"Success: {progress['success_rate']:.1%} "
                   f"Memory: {self.memory_monitor.get_memory_usage():.1f}MB")
    
    def _finalize_processing(self) -> Any:
        """Finalize processing and update statistics"""
        self.stats.end_time = datetime.now()
        
        if self.stats.start_time and self.stats.end_time:
            self.stats.total_processing_time = (
                self.stats.end_time - self.stats.start_time
            ).total_seconds()
            
            if self.stats.processed_targets > 0:
                self.stats.avg_processing_time = (
                    self.stats.total_processing_time / self.stats.processed_targets
                )
        
        # Update memory statistics
        memory_stats = self.memory_monitor.get_memory_stats()
        self.stats.memory_peak_mb = memory_stats['peak_mb']
        self.stats.memory_current_mb = memory_stats['current_mb']
        
        logger.info(f"Batch processing completed: {self.stats.to_dict()}")
    
    def stop_processing(self) -> Any:
        """Stop processing gracefully"""
        self._stop_processing = True
        logger.info("Processing stop requested")


class ResultStorage:
    """Result storage and persistence"""
    
    def __init__(self, config: ChunkConfig):
        
    """__init__ function."""
self.config = config
        self.storage_path = Path("batch_results")
        self.storage_path.mkdir(exist_ok=True)
        self.db_path = self.storage_path / "results.db"
        self._init_storage()
    
    def _init_storage(self) -> Any:
        """Initialize storage system"""
        if self.config.result_format == "sqlite":
            self._init_sqlite()
    
    def _init_sqlite(self) -> Any:
        """Initialize SQLite database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS results (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        chunk_id INTEGER,
                        target_hash TEXT,
                        result_data TEXT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to initialize SQLite storage: {e}")
    
    async def save_chunk_results(self, results: List[Any], chunk_id: Optional[int] = None):
        """Save chunk results to storage"""
        if not self.config.enable_result_persistence:
            return
        
        try:
            if self.config.result_format == "json":
                await self._save_json_results(results, chunk_id)
            elif self.config.result_format == "pickle":
                await self._save_pickle_results(results, chunk_id)
            elif self.config.result_format == "sqlite":
                await self._save_sqlite_results(results, chunk_id)
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
    
    async def _save_json_results(self, results: List[Any], chunk_id: Optional[int] = None):
        """Save results as JSON"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        chunk_suffix = f"_chunk_{chunk_id}" if chunk_id is not None else ""
        filename = f"results_{timestamp}{chunk_suffix}.json"
        
        if self.config.compression_enabled:
            filename += ".gz"
        
        filepath = self.storage_path / filename
        
        # Convert results to serializable format
        serializable_results = []
        for result in results:
            if hasattr(result, 'to_dict'):
                serializable_results.append(result.to_dict())
            elif isinstance(result, dict):
                serializable_results.append(result)
            else:
                serializable_results.append(str(result))
        
        # Save to file
        if self.config.compression_enabled:
            with gzip.open(filepath, 'wt', encoding='utf-8') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                json.dump(serializable_results, f, indent=2)
        else:
            with open(filepath, 'w', encoding='utf-8') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                json.dump(serializable_results, f, indent=2)
    
    async def _save_pickle_results(self, results: List[Any], chunk_id: Optional[int] = None):
        """Save results as pickle"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        chunk_suffix = f"_chunk_{chunk_id}" if chunk_id is not None else ""
        filename = f"results_{timestamp}{chunk_suffix}.pkl"
        
        if self.config.compression_enabled:
            filename += ".gz"
        
        filepath = self.storage_path / filename
        
        # Save to file
        if self.config.compression_enabled:
            with gzip.open(filepath, 'wb') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                pickle.dump(results, f)
        else:
            with open(filepath, 'wb') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                pickle.dump(results, f)
    
    async def _save_sqlite_results(self, results: List[Any], chunk_id: Optional[int] = None):
        """Save results to SQLite"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                for result in results:
                    # Create hash for target identification
                    target_hash = hashlib.md5(str(result).encode()).hexdigest()
                    
                    # Convert result to JSON string
                    if hasattr(result, 'to_dict'):
                        result_data = json.dumps(result.to_dict())
                    elif isinstance(result, dict):
                        result_data = json.dumps(result)
                    else:
                        result_data = str(result)
                    
                    conn.execute("""
                        INSERT INTO results (chunk_id, target_hash, result_data)
                        VALUES (?, ?, ?)
                    """, (chunk_id, target_hash, result_data))
                
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to save to SQLite: {e}")
    
    async def load_results(self, chunk_id: Optional[int] = None) -> List[Any]:
        """Load results from storage"""
        try:
            if self.config.result_format == "sqlite":
                return await self._load_sqlite_results(chunk_id)
            else:
                return await self._load_file_results(chunk_id)
        except Exception as e:
            logger.error(f"Failed to load results: {e}")
            return []
    
    async def _load_sqlite_results(self, chunk_id: Optional[int] = None) -> List[Any]:
        """Load results from SQLite"""
        results = []
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = "SELECT result_data FROM results"
                if chunk_id is not None:
                    query += f" WHERE chunk_id = {chunk_id}"
                
                cursor = conn.execute(query)
                for row in cursor:
                    try:
                        result_data = json.loads(row[0])
                        results.append(result_data)
                    except json.JSONDecodeError:
                        results.append(row[0])
        except Exception as e:
            logger.error(f"Failed to load from SQLite: {e}")
        
        return results
    
    async def _load_file_results(self, chunk_id: Optional[int] = None) -> List[Any]:
        """Load results from files"""
        results = []
        
        try:
            # Find result files
            pattern = "results_*.json*" if self.config.result_format == "json" else "results_*.pkl*"
            files = list(self.storage_path.glob(pattern))
            
            for filepath in files:
                if chunk_id is not None and f"_chunk_{chunk_id}" not in filepath.name:
                    continue
                
                try:
                    if self.config.compression_enabled:
                        with gzip.open(filepath, 'rt' if self.config.result_format == "json" else 'rb') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                            if self.config.result_format == "json":
                                file_results = json.load(f)
                            else:
                                file_results = pickle.load(f)
                    else:
                        with open(filepath, 'r' if self.config.result_format == "json" else 'rb') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                            if self.config.result_format == "json":
                                file_results = json.load(f)
                            else:
                                file_results = pickle.load(f)
                    
                    results.extend(file_results)
                except Exception as e:
                    logger.warning(f"Failed to load file {filepath}: {e}")
        except Exception as e:
            logger.error(f"Failed to load file results: {e}")
        
        return results


# Example usage and demonstration functions

async def demonstrate_batch_processing():
    """Demonstrate batch processing capabilities"""
    logger.info("Starting batch processing demonstration")
    
    # Configuration
    config = ChunkConfig(
        chunk_size=100,
        max_memory_mb=256,
        max_processing_time=60.0,
        max_concurrent_chunks=2,
        enable_memory_monitoring=True,
        enable_progress_tracking=True,
        enable_error_recovery=True,
        enable_result_persistence=True,
        result_format="json",
        compression_enabled=False
    )
    
    # Create batch processor
    processor = BatchProcessor(config)
    
    # Create sample targets
    targets = []
    for i in range(1000):
        target = {
            "id": i,
            "host": f"host{i}.example.com",
            "port": 80 + (i % 100),
            "complexity": 1 + (i % 5),
            "priority": 1 + (i % 3)
        }
        targets.append(target)
    
    # Define processing function
    async def process_target(target) -> Optional[Dict[str, Any]]:
        """Sample processing function"""
        await asyncio.sleep(0.01)  # Simulate processing time
        
        # Simulate some failures
        if target["id"] % 50 == 0:
            raise Exception(f"Simulated failure for target {target['id']}")
        
        return {
            "target_id": target["id"],
            "result": f"Processed {target['host']}:{target['port']}",
            "timestamp": datetime.now().isoformat()
        }
    
    try:
        # Process targets
        results = await processor.process_targets(
            targets=targets,
            processor_func=process_target,
            mode=ProcessingMode.PARALLEL,
            strategy=ChunkStrategy.ADAPTIVE
        )
        
        logger.info(f"Processing completed. Results: {len(results)}")
        
        # Print statistics
        stats = processor.stats.to_dict()
        logger.info(f"Final statistics: {json.dumps(stats, indent=2)}")
        
        # Load and verify results
        loaded_results = await processor.result_storage.load_results()
        logger.info(f"Loaded results: {len(loaded_results)}")
        
    except Exception as e:
        logger.error(f"Batch processing failed: {e}")


async def demonstrate_memory_management():
    """Demonstrate memory management capabilities"""
    logger.info("Starting memory management demonstration")
    
    # Configuration with strict memory limits
    config = ChunkConfig(
        chunk_size=50,
        max_memory_mb=128,  # Strict memory limit
        max_processing_time=30.0,
        max_concurrent_chunks=1,
        enable_memory_monitoring=True,
        enable_progress_tracking=True
    )
    
    processor = BatchProcessor(config)
    
    # Create memory-intensive targets
    targets = []
    for i in range(500):
        target = {
            "id": i,
            "data": "x" * 1024,  # 1KB of data per target
            "complexity": 5
        }
        targets.append(target)
    
    async def memory_intensive_processor(target) -> Any:
        """Memory-intensive processing function"""
        # Simulate memory usage
        large_data = "x" * (1024 * 10)  # 10KB per processing
        await asyncio.sleep(0.01)
        
        return {
            "target_id": target["id"],
            "processed_data": len(large_data),
            "memory_usage": processor.memory_monitor.get_memory_usage()
        }
    
    try:
        results = await processor.process_targets(
            targets=targets,
            processor_func=memory_intensive_processor,
            mode=ProcessingMode.SEQUENTIAL,
            strategy=ChunkStrategy.MEMORY_BASED
        )
        
        logger.info(f"Memory management completed. Results: {len(results)}")
        
        # Print memory statistics
        memory_stats = processor.memory_monitor.get_memory_stats()
        logger.info(f"Memory statistics: {json.dumps(memory_stats, indent=2)}")
        
    except Exception as e:
        logger.error(f"Memory management failed: {e}")


async def demonstrate_chunk_strategies():
    """Demonstrate different chunking strategies"""
    logger.info("Starting chunking strategies demonstration")
    
    config = ChunkConfig(
        chunk_size=100,
        max_memory_mb=256,
        max_processing_time=30.0,
        enable_memory_monitoring=True
    )
    
    processor = BatchProcessor(config)
    
    # Create diverse targets
    targets = []
    for i in range(1000):
        target = {
            "id": i,
            "size": 100 + (i % 900),  # Variable size
            "complexity": 1 + (i % 10),  # Variable complexity
            "priority": 1 + (i % 5)  # Variable priority
        }
        targets.append(target)
    
    async def simple_processor(target) -> Any:
        """Simple processing function"""
        await asyncio.sleep(0.001)
        return {"processed": target["id"]}
    
    strategies = [
        ChunkStrategy.FIXED_SIZE,
        ChunkStrategy.MEMORY_BASED,
        ChunkStrategy.TIME_BASED,
        ChunkStrategy.ADAPTIVE,
        ChunkStrategy.WEIGHTED
    ]
    
    for strategy in strategies:
        logger.info(f"Testing strategy: {strategy.value}")
        
        try:
            start_time = time.time()
            results = await processor.process_targets(
                targets=targets[:100],  # Use subset for testing
                processor_func=simple_processor,
                mode=ProcessingMode.SEQUENTIAL,
                strategy=strategy
            )
            end_time = time.time()
            
            logger.info(f"Strategy {strategy.value}: {len(results)} results in {end_time - start_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Strategy {strategy.value} failed: {e}")


if __name__ == "__main__":
    # Run demonstrations
    async def main():
        
    """main function."""
try:
            await demonstrate_batch_processing()
            await demonstrate_memory_management()
            await demonstrate_chunk_strategies()
        except KeyboardInterrupt:
            logger.info("Demonstration interrupted by user")
        except Exception as e:
            logger.error(f"Demonstration error: {e}")
    
    asyncio.run(main()) 