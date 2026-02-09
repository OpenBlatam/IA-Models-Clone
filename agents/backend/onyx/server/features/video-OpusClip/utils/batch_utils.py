"""
Video Processing Batch Utilities

Optimized batch operations, metrics collection, and mixins for video processing.
"""

import time
import numpy as np
import orjson
import msgspec
from functools import wraps
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, TypeVar, Callable, Union
from collections import defaultdict

import structlog
from prometheus_client import Counter, Histogram

from .constants import (
    SLOW_OPERATION_THRESHOLD,
    DEFAULT_MAX_WORKERS,
    PANDAS_AVAILABLE,
    SENTRY_AVAILABLE
)

logger = structlog.get_logger()

# Type variable for generic operations
T = TypeVar('T')

# =============================================================================
# METRICS COLLECTION
# =============================================================================

class BatchMetrics:
    """Optimized metrics collection with lazy evaluation and caching."""
    
    def __init__(self):
        self.counters: Dict[str, Counter] = {}
        self.histograms: Dict[str, Histogram] = {}
    
    def get_counter(self, name: str) -> Counter:
        """Get or create a counter metric."""
        if name not in self.counters: self.counters[name] = Counter(f"video_model_{name}_total", f"Total {name} calls")
        return self.counters[name]
    
    def get_histogram(self, name: str) -> Histogram:
        """Get or create a histogram metric."""
        if name not in self.histograms: self.histograms[name] = Histogram(f"video_model_{name}_duration", f"Duration of {name} operations")
        return self.histograms[name]

# Global metrics instance
_metrics = BatchMetrics()

# =============================================================================
# TIMING DECORATORS
# =============================================================================

def _optimized_batch_timeit(fn: Callable) -> Callable:
    """Optimized timing decorator with lazy logging and performance thresholds."""
    
    @wraps(fn)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            result = fn(*args, **kwargs)
            duration = time.perf_counter() - start
            
            # Only log slow operations to reduce overhead
            if duration > SLOW_OPERATION_THRESHOLD:
                _metrics.get_counter(fn.__name__).inc()
                _metrics.get_histogram(fn.__name__).observe(duration)
                logger.info(
                    f"batch_{fn.__name__}", 
                    count=len(args[0]) if args else 0, 
                    duration=f"{duration:.4f}"
                )
            
            return result
        except Exception as e:
            if SENTRY_AVAILABLE:
                import sentry_sdk
                sentry_sdk.capture_exception(e)
            logger.error(f"batch_{fn.__name__}_error", error=str(e))
            raise
    
    return wrapper

# =============================================================================
# VECTORIZED OPERATIONS
# =============================================================================

def _vectorized_batch_operation(items: List[T], operation: Callable, **kwargs) -> List[T]:
    """Vectorized batch operations using numpy for better performance."""
    if not items: return []
    
    # Convert to numpy array for vectorized operations
    arr = np.array(items, dtype=object)
    
    # Apply vectorized operation
    if hasattr(operation, '__vectorize__'):
        result = np.vectorize(operation)(arr, **kwargs)
    else:
        result = np.array([operation(item, **kwargs) for item in arr])
    
    return result.tolist()

def _parallel_batch_operation(items: List[T], operation: Callable, 
                            max_workers: int = DEFAULT_MAX_WORKERS, **kwargs) -> List[T]:
    """Parallel batch operations for I/O intensive tasks."""
    if not items: return []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(operation, item, **kwargs) for item in items]
        results = [future.result() for future in as_completed(futures)]
    
    return results

# =============================================================================
# OPTIMIZED BATCH OPERATIONS
# =============================================================================

def _optimized_batch_groupby(items: List[T], key: Union[str, Callable]) -> Dict[Any, List[T]]:
    """Optimized groupby using defaultdict with key function caching."""
    d = defaultdict(list)
    key_fn = key if callable(key) else lambda x: getattr(x, key)
    
    for item in items:
        d[key_fn(item)].append(item)
    
    return dict(d)

def _optimized_batch_filter(items: List[T], fn: Callable[[T], bool]) -> List[T]:
    """Optimized filter using list comprehension with early exit."""
    return [x for x in items if fn(x)]

def _optimized_batch_map(items: List[T], fn: Callable[[T], Any]) -> List[Any]:
    """Optimized map using list comprehension."""
    return [fn(x) for x in items]

def _optimized_batch_sort(items: List[T], key: Union[str, Callable], 
                         reverse: bool = False) -> List[T]:
    """Optimized sort with key function caching."""
    key_fn = key if callable(key) else lambda x: getattr(x, key)
    return sorted(items, key=key_fn, reverse=reverse)

def _optimized_batch_deduplicate(items: List[T], key: Union[str, Callable] = None) -> List[T]:
    """Optimized deduplication with custom key function."""
    if not items: return []
    
    seen = set()
    result = []
    key_fn = key if callable(key) else (lambda x: getattr(x, key) if key else x.as_tuple())
    
    for item in items:
        k = key_fn(item)
        if k not in seen:
            seen.add(k)
            result.append(item)
    
    return result

# =============================================================================
# BATCH MIXIN
# =============================================================================

class OptimizedBatchMixin:
    """Mixin providing optimized batch operations for all video models."""
    
    @classmethod
    def batch_encode(cls, items: List[T]) -> bytes:
        """Ultra-fast batch serialization using msgspec."""
        return msgspec.json.encode(items)
    
    @classmethod
    def batch_decode(cls, data: bytes) -> List[T]:
        """Ultra-fast batch deserialization using msgspec."""
        return msgspec.json.decode(data, type=List[cls])
    
    @classmethod
    def batch_to_numpy(cls, items: List[T]):
        """Vectorized conversion to numpy array."""
        if not items: return np.array([], dtype=object)
        return np.array([item.as_tuple() for item in items], dtype=object)
    
    @classmethod
    def batch_to_pandas(cls, items: List[T]):
        """Optimized conversion to pandas DataFrame."""
        if not PANDAS_AVAILABLE: raise ImportError("pandas is not installed")
        if not items: return pd.DataFrame()
        return pd.DataFrame([item.__dict__ for item in items])
    
    @classmethod
    def batch_to_parquet(cls, items: List[T], path: str):
        """Optimized conversion to Parquet with snappy compression."""
        if not PANDAS_AVAILABLE: raise ImportError("pandas is not installed")
        cls.batch_to_pandas(items).to_parquet(path, compression='snappy')
    
    @classmethod
    def batch_from_parquet(cls, path: str) -> List[T]:
        """Optimized conversion from Parquet."""
        if not PANDAS_AVAILABLE: raise ImportError("pandas is not installed")
        df = pd.read_parquet(path)
        return [cls(**d) for d in df.to_dict(orient="records")]
    
    @classmethod
    def batch_validate_unique(cls, items: List[T], key: Callable = None):
        """Optimized unique validation with early exit."""
        if not items: return
        
        seen = set()
        key_fn = key or (lambda x: x.as_tuple())
        
        for item in items:
            k = key_fn(item)
            if k in seen:
                if SENTRY_AVAILABLE:
                    import sentry_sdk
                    sentry_sdk.capture_message(f"Duplicate key found: {k}")
                raise ValueError(f"Duplicate key found: {k}")
            seen.add(k)
    
    @classmethod
    def batch_to_dicts(cls, items: List[T]) -> List[dict]:
        """Optimized conversion to dictionaries."""
        return [orjson.loads(orjson.dumps(item.__dict__)) for item in items]
    
    @classmethod
    def batch_from_dicts(cls, dicts: List[dict]) -> List[T]:
        """Optimized conversion from dictionaries."""
        return [cls(**d) for d in dicts]
    
    @classmethod
    def batch_deduplicate(cls, items: List[T], key: str = None) -> List[T]:
        """Optimized deduplication."""
        return _optimized_batch_deduplicate(items, key)
    
    @classmethod
    def batch_groupby(cls, items: List[T], key: Union[str, Callable]) -> Dict[Any, List[T]]:
        """Optimized groupby operation."""
        return _optimized_batch_groupby(items, key)
    
    @classmethod
    def batch_filter(cls, items: List[T], fn: Callable[[T], bool]) -> List[T]:
        """Optimized filter operation."""
        return _optimized_batch_filter(items, fn)
    
    @classmethod
    def batch_map(cls, items: List[T], fn: Callable[[T], Any]) -> List[Any]:
        """Optimized map operation."""
        return _optimized_batch_map(items, fn)
    
    @classmethod
    def batch_sort(cls, items: List[T], key: Union[str, Callable], 
                   reverse: bool = False) -> List[T]:
        """Optimized sort operation."""
        return _optimized_batch_sort(items, key, reverse)
    
    @classmethod
    def to_training_example(cls, obj: T) -> dict:
        """Optimized training example conversion."""
        return orjson.loads(orjson.dumps(obj.__dict__))
    
    @classmethod
    def from_training_example(cls, data: dict) -> T:
        """Optimized training example conversion."""
        return cls(**data)

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def batch_chunk(items: List[T], chunk_size: int) -> List[List[T]]:
    """Split items into chunks of specified size."""
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]

def batch_flatten(items: List[List[T]]) -> List[T]:
    """Flatten a list of lists."""
    return [item for sublist in items for item in sublist]

def batch_zip(*iterables) -> List[tuple]:
    """Zip multiple iterables together."""
    return list(zip(*iterables))

def batch_enumerate(items: List[T], start: int = 0) -> List[tuple]:
    """Enumerate items with optional start index."""
    return list(enumerate(items, start=start)) 