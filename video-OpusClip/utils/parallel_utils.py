"""
Advanced Parallel Processing Utilities

Ultra-fast parallel processing using the best libraries available:
- asyncio: Async I/O operations
- multiprocessing: CPU-intensive tasks
- concurrent.futures: Thread/Process pools
- joblib: Parallel computing for scientific Python
- ray: Distributed computing
- dask: Parallel computing with pandas/numpy
- numba: JIT compilation for numerical operations
- uvloop: Ultra-fast event loop (Linux/macOS)
"""

import asyncio
import multiprocessing as mp
import concurrent.futures
import time
import functools
from typing import List, Callable, Any, Dict, Optional, Union, TypeVar
from dataclasses import dataclass
from enum import Enum

# Optional high-performance libraries
try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    joblib = None
    JOBLIB_AVAILABLE = False

try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    ray = None
    RAY_AVAILABLE = False

try:
    import dask.dataframe as dd
    import dask.array as da
    DASK_AVAILABLE = True
except ImportError:
    dd = None
    da = None
    DASK_AVAILABLE = False

try:
    import numba
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    numba = None
    NUMBA_AVAILABLE = False

try:
    import uvloop
    UVLOOP_AVAILABLE = True
except ImportError:
    uvloop = None
    UVLOOP_AVAILABLE = False

import numpy as np
import structlog
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

logger = structlog.get_logger()

# Type variables
T = TypeVar('T')
R = TypeVar('R')

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class ParallelConfig:
    """Configuration for parallel processing."""
    max_workers: int = mp.cpu_count()
    chunk_size: int = 1000
    timeout: Optional[float] = None
    backend: str = "auto"  # auto, thread, process, joblib, ray, dask
    use_uvloop: bool = UVLOOP_AVAILABLE
    use_numba: bool = NUMBA_AVAILABLE

class BackendType(Enum):
    """Available parallel processing backends."""
    AUTO = "auto"
    THREAD = "thread"
    PROCESS = "process"
    JOBLIB = "joblib"
    RAY = "ray"
    DASK = "dask"
    ASYNC = "async"

# =============================================================================
# ASYNC PROCESSING
# =============================================================================

async def async_batch_process(
    items: List[T],
    func: Callable[[T], R],
    max_concurrent: int = 100,
    chunk_size: int = 100
) -> List[R]:
    """
    Process items asynchronously with controlled concurrency.
    
    Args:
        items: List of items to process
        func: Async function to apply to each item
        max_concurrent: Maximum concurrent tasks
        chunk_size: Size of chunks for processing
        
    Returns:
        List of results
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_item(item: T) -> R:
        async with semaphore:
            return await func(item)
    
    # Process in chunks to avoid memory issues
    results = []
    for i in range(0, len(items), chunk_size):
        chunk = items[i:i + chunk_size]
        chunk_results = await asyncio.gather(
            *[process_item(item) for item in chunk]
        )
        results.extend(chunk_results)
    
    return results

def setup_async_loop() -> None:
    """Setup ultra-fast event loop with uvloop if available."""
    if UVLOOP_AVAILABLE: uvloop.install(); logger.info("Using uvloop for ultra-fast async processing")

# =============================================================================
# MULTIPROCESSING WITH JOBLIB
# =============================================================================

def joblib_parallel_process(
    items: List[T],
    func: Callable[[T], R],
    n_jobs: int = -1,
    backend: str = "multiprocessing",
    batch_size: int = 1000,
    verbose: int = 0
) -> List[R]:
    """
    Parallel processing using joblib (best for scientific computing).
    
    Args:
        items: List of items to process
        func: Function to apply to each item
        n_jobs: Number of jobs (-1 for all CPUs)
        backend: Backend to use (multiprocessing, threading, loky)
        batch_size: Size of batches
        verbose: Verbosity level
        
    Returns:
        List of results
    """
    if not JOBLIB_AVAILABLE: raise ImportError("joblib is not installed")
    
    return joblib.Parallel(
        n_jobs=n_jobs,
        backend=backend,
        batch_size=batch_size,
        verbose=verbose
    )(joblib.delayed(func)(item) for item in items)

# =============================================================================
# RAY DISTRIBUTED COMPUTING
# =============================================================================

def ray_parallel_process(
    items: List[T],
    func: Callable[[T], R],
    num_cpus: Optional[int] = None,
    num_gpus: Optional[int] = None,
    resources: Optional[Dict] = None
) -> List[R]:
    """
    Distributed processing using Ray (best for distributed computing).
    
    Args:
        items: List of items to process
        func: Function to apply to each item
        num_cpus: Number of CPUs to use
        num_gpus: Number of GPUs to use
        resources: Additional resources
        
    Returns:
        List of results
    """
    if not RAY_AVAILABLE: raise ImportError("ray is not installed")
    
    # Initialize Ray if not already done
    if not ray.is_initialized(): ray.init()
    
    # Remote function
    @ray.remote
    def remote_func(item):
        return func(item)
    
    # Submit tasks
    futures = [
        remote_func.remote(item) for item in items
    ]
    
    # Get results
    return ray.get(futures)

# =============================================================================
# DASK PARALLEL COMPUTING
# =============================================================================

def dask_parallel_process(
    items: List[T],
    func: Callable[[T], R],
    npartitions: Optional[int] = None,
    scheduler: str = "threads"
) -> List[R]:
    """
    Parallel processing using Dask (best for pandas/numpy operations).
    
    Args:
        items: List of items to process
        func: Function to apply to each item
        npartitions: Number of partitions
        scheduler: Scheduler to use (threads, processes, distributed)
        
    Returns:
        List of results
    """
    if not DASK_AVAILABLE: raise ImportError("dask is not installed")
    
    # Convert to dask array
    dask_array = da.from_array(items, chunks=len(items) // (npartitions or mp.cpu_count()))
    
    # Apply function
    result_array = dask_array.map_blocks(func)
    
    # Compute results
    return result_array.compute(scheduler=scheduler)

# =============================================================================
# NUMBA JIT COMPILATION
# =============================================================================

if NUMBA_AVAILABLE:
    @jit(nopython=True, parallel=True)
    def numba_parallel_array_operation(arr: np.ndarray, func) -> np.ndarray:
        """
        Parallel array operations using Numba JIT compilation.
        
        Args:
            arr: Input array
            func: Function to apply (must be numba-compatible)
            
        Returns:
            Transformed array
        """
        result = np.empty_like(arr)
        for i in prange(len(arr)):
            result[i] = func(arr[i])
        return result

def numba_parallel_process(
    items: List[T],
    func: Callable[[T], R]
) -> List[R]:
    """
    Parallel processing using Numba JIT compilation.
    
    Args:
        items: List of items to process
        func: Function to apply (must be numba-compatible)
        
    Returns:
        List of results
    """
    if not NUMBA_AVAILABLE: raise ImportError("numba is not installed")
    
    # Convert to numpy array for numba
    arr = np.array(items)
    
    # Apply function with JIT compilation
    result_arr = numba_parallel_array_operation(arr, func)
    
    return result_arr.tolist()

# =============================================================================
# HYBRID PROCESSING
# =============================================================================

class HybridParallelProcessor:
    """Hybrid parallel processor that chooses the best backend automatically."""
    
    def __init__(self, config: ParallelConfig):
        self.config = config
        self.backend_stats = {}
    
    def choose_backend(self, items: List[T], func: Callable) -> BackendType:
        """Choose the best backend based on data and function characteristics."""
        n_items = len(items)
        
        # Small datasets: use threading
        if n_items < 100: return BackendType.THREAD
        
        # CPU-intensive tasks: use multiprocessing
        if self._is_cpu_intensive(func):
            if JOBLIB_AVAILABLE: return BackendType.JOBLIB
            return BackendType.PROCESS
        
        # I/O-intensive tasks: use async
        if self._is_io_intensive(func): return BackendType.ASYNC
        
        # Large datasets: use dask
        if n_items > 10000 and DASK_AVAILABLE: return BackendType.DASK
        
        # Distributed computing: use ray
        if n_items > 50000 and RAY_AVAILABLE: return BackendType.RAY
        
        # Default: use joblib
        if JOBLIB_AVAILABLE: return BackendType.JOBLIB
        
        return BackendType.PROCESS
    
    def _is_cpu_intensive(self, func: Callable) -> bool:
        """Detect if function is CPU-intensive."""
        # Simple heuristic: check function name and docstring
        func_name = func.__name__.lower()
        cpu_keywords = ['compute', 'calculate', 'process', 'transform', 'encode', 'decode']
        return any(keyword in func_name for keyword in cpu_keywords)
    
    def _is_io_intensive(self, func: Callable) -> bool:
        """Detect if function is I/O-intensive."""
        func_name = func.__name__.lower()
        io_keywords = ['fetch', 'download', 'upload', 'read', 'write', 'api', 'http']
        return any(keyword in func_name for keyword in io_keywords)
    
    def process(
        self,
        items: List[T],
        func: Callable[[T], R],
        backend: Optional[BackendType] = None
    ) -> List[R]:
        """
        Process items using the best available backend.
        
        Args:
            items: List of items to process
            func: Function to apply
            backend: Force specific backend
            
        Returns:
            List of results
        """
        if backend is None: backend = self.choose_backend(items, func)
        
        start_time = time.perf_counter()
        
        try:
            if backend == BackendType.THREAD:
                result = self._thread_process(items, func)
            elif backend == BackendType.PROCESS:
                result = self._process_process(items, func)
            elif backend == BackendType.JOBLIB:
                result = joblib_parallel_process(items, func)
            elif backend == BackendType.RAY:
                result = ray_parallel_process(items, func)
            elif backend == BackendType.DASK:
                result = dask_parallel_process(items, func)
            elif backend == BackendType.ASYNC:
                result = asyncio.run(async_batch_process(items, func))
            else:
                result = self._process_process(items, func)
            
            duration = time.perf_counter() - start_time
            self.backend_stats[backend.value] = {
                'duration': duration,
                'items_processed': len(items),
                'items_per_second': len(items) / duration
            }
            
            logger.info(
                f"Parallel processing completed",
                backend=backend.value,
                duration=f"{duration:.4f}s",
                items_per_second=f"{len(items) / duration:.2f}"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Parallel processing failed", backend=backend.value, error=str(e))
            raise
    
    def _thread_process(self, items: List[T], func: Callable[[T], R]) -> List[R]:
        """Thread-based processing."""
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            return list(executor.map(func, items))
    
    def _process_process(self, items: List[T], func: Callable[[T], R]) -> List[R]:
        """Process-based processing."""
        with ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
            return list(executor.map(func, items))

# =============================================================================
# SPECIALIZED PROCESSORS
# =============================================================================

class VideoParallelProcessor:
    """Specialized parallel processor for video operations."""
    
    def __init__(self):
        self.hybrid_processor = HybridParallelProcessor(ParallelConfig())
    
    def process_video_clips(
        self,
        video_requests: List[T],
        processor_func: Callable[[T], R]
    ) -> List[R]:
        """Process video clips in parallel."""
        return self.hybrid_processor.process(video_requests, processor_func)
    
    def process_viral_variants(
        self,
        viral_requests: List[T],
        processor_func: Callable[[T], R]
    ) -> List[R]:
        """Process viral variants in parallel."""
        return self.hybrid_processor.process(viral_requests, processor_func)
    
    def batch_encode_videos(
        self,
        videos: List[T],
        encode_func: Callable[[T], bytes]
    ) -> List[bytes]:
        """Batch encode videos in parallel."""
        return self.hybrid_processor.process(videos, encode_func)
    
    def batch_decode_videos(
        self,
        encoded_data: List[bytes],
        decode_func: Callable[[bytes], T]
    ) -> List[T]:
        """Batch decode videos in parallel."""
        return self.hybrid_processor.process(encoded_data, decode_func)

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def parallel_map(
    items: List[T],
    func: Callable[[T], R],
    backend: BackendType = BackendType.AUTO,
    **kwargs
) -> List[R]:
    """
    Simple parallel map function with automatic backend selection.
    
    Args:
        items: List of items to process
        func: Function to apply
        backend: Backend to use
        **kwargs: Additional arguments for the processor
        
    Returns:
        List of results
    """
    processor = HybridParallelProcessor(ParallelConfig(**kwargs))
    return processor.process(items, func, backend)

def get_optimal_chunk_size(total_items: int, num_workers: int) -> int:
    """Calculate optimal chunk size for parallel processing."""
    return max(1, total_items // (num_workers * 4))

def estimate_processing_time(
    items: List[T],
    sample_func: Callable[[T], R],
    sample_size: int = 10
) -> float:
    """Estimate total processing time based on a sample."""
    if len(items) <= sample_size: return 0.0
    
    sample_items = items[:sample_size]
    start_time = time.perf_counter()
    
    for item in sample_items:
        sample_func(item)
    
    sample_duration = time.perf_counter() - start_time
    avg_time_per_item = sample_duration / sample_size
    
    return avg_time_per_item * len(items) 