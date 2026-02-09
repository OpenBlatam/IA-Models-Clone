"""
Shared Utilities - Comprehensive utility functions.

Provides:
- Time measurement and formatting
- Size formatting
- Result saving/loading
- Retry mechanisms
- Memory monitoring
- Statistical calculations
- File operations
- Progress tracking
"""

import time
import logging
import json
import csv
from typing import Dict, Any, Optional, List, Callable, Union
from functools import wraps
from pathlib import Path
from datetime import datetime, timedelta
from contextlib import contextmanager
import hashlib

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════════
# TIME UTILITIES
# ════════════════════════════════════════════════════════════════════════════════

def measure_time(func: Callable) -> Callable:
    """
    Decorator to measure function execution time.
    
    Returns:
        Tuple of (result, elapsed_ms) instead of just result
    
    Example:
        >>> @measure_time
        >>> def my_function():
        >>>     return "result"
        >>> result, elapsed = my_function()
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.debug(f"{func.__name__} took {elapsed_ms:.2f}ms")
        return result, elapsed_ms
    return wrapper


@contextmanager
def timer(description: str = "Operation"):
    """
    Context manager for timing operations.
    
    Args:
        description: Description of the operation
    
    Example:
        >>> with timer("Processing data"):
        >>>     process_data()
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        logger.info(f"{description} took {elapsed:.3f}s")


def format_size(size_bytes: int) -> str:
    """
    Format bytes to human-readable size.
    
    Args:
        size_bytes: Size in bytes
    
    Returns:
        Human-readable size string
    
    Example:
        >>> format_size(1024)
        '1.00 KB'
        >>> format_size(1048576)
        '1.00 MB'
    """
    if size_bytes == 0:
        return "0 B"
    
    units = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']
    unit_index = 0
    size = float(size_bytes)
    
    while size >= 1024.0 and unit_index < len(units) - 1:
        size /= 1024.0
        unit_index += 1
    
    return f"{size:.2f} {units[unit_index]}"


def format_duration(seconds: float) -> str:
    """
    Format seconds to human-readable duration.
    
    Args:
        seconds: Duration in seconds
    
    Returns:
        Human-readable duration string
    
    Example:
        >>> format_duration(90)
        '1.5m'
        >>> format_duration(3661)
        '1.0h'
    """
    if seconds < 1.0:
        return f"{seconds * 1000:.0f}ms"
    elif seconds < 60.0:
        return f"{seconds:.2f}s"
    elif seconds < 3600.0:
        minutes = seconds / 60.0
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600.0
        if hours < 24.0:
            return f"{hours:.1f}h"
        else:
            days = hours / 24.0
            return f"{days:.1f}d"


def format_timestamp(timestamp: Optional[float] = None, format: str = "iso") -> str:
    """
    Format timestamp to string.
    
    Args:
        timestamp: Unix timestamp (None = now)
        format: Format type (iso, readable, compact)
    
    Returns:
        Formatted timestamp string
    """
    if timestamp is None:
        dt = datetime.now()
    else:
        dt = datetime.fromtimestamp(timestamp)
    
    if format == "iso":
        return dt.isoformat()
    elif format == "readable":
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    elif format == "compact":
        return dt.strftime("%Y%m%d_%H%M%S")
    else:
        return dt.isoformat()


# ════════════════════════════════════════════════════════════════════════════════
# FILE OPERATIONS
# ════════════════════════════════════════════════════════════════════════════════

def save_results(
    results: Union[Dict[str, Any], List[Dict[str, Any]]],
    output_path: Path,
    format: str = "json",
    indent: int = 2,
) -> None:
    """
    Save benchmark results to file.
    
    Supports multiple formats:
    - json: JSON format (default)
    - csv: CSV format (for tabular data)
    
    Args:
        results: Results to save (dict or list of dicts)
        output_path: Path to save file
        format: File format (json, csv)
        indent: JSON indentation (for json format)
    
    Example:
        >>> save_results({"accuracy": 0.95}, Path("results.json"))
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == "json":
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=indent, ensure_ascii=False)
    elif format == "csv":
        if isinstance(results, list) and results:
            # Write CSV
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=results[0].keys())
                writer.writeheader()
                writer.writerows(results)
        else:
            raise ValueError("CSV format requires list of dictionaries")
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    logger.info(f"Results saved to {output_path}")


def load_results(path: Path, format: str = "json") -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Load benchmark results from file.
    
    Args:
        path: Path to results file
        format: File format (json, csv)
    
    Returns:
        Loaded results (dict or list of dicts)
    
    Example:
        >>> results = load_results(Path("results.json"))
    """
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {path}")
    
    if format == "json":
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    elif format == "csv":
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            return list(reader)
    else:
        raise ValueError(f"Unsupported format: {format}")


def ensure_directory(path: Union[str, Path]) -> Path:
    """
    Ensure directory exists, creating if necessary.
    
    Args:
        path: Directory path
    
    Returns:
        Path object
    """
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def get_file_hash(path: Path, algorithm: str = "sha256") -> str:
    """
    Calculate file hash.
    
    Args:
        path: File path
        algorithm: Hash algorithm (md5, sha1, sha256)
    
    Returns:
        Hexadecimal hash string
    """
    hash_obj = hashlib.new(algorithm)
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_obj.update(chunk)
    return hash_obj.hexdigest()


# ════════════════════════════════════════════════════════════════════════════════
# RETRY MECHANISMS (DEPRECATED - Use decorators.retry_on_failure or retry_utils.retry)
# ════════════════════════════════════════════════════════════════════════════════

# Re-export from decorators for backward compatibility
from .decorators import retry_on_failure

# For better retry functionality, use:
# from core.retry_utils import retry
# from core.decorators import retry_on_failure
        def wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        if on_retry:
                            on_retry(e, attempt + 1)
                        else:
                            logger.warning(
                                f"Attempt {attempt + 1}/{max_attempts} failed: {e}. "
                                f"Retrying in {current_delay:.1f}s..."
                            )
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(f"All {max_attempts} attempts failed")
            
            raise last_exception
        return wrapper
    return decorator


# ════════════════════════════════════════════════════════════════════════════════
# MEMORY MONITORING
# ════════════════════════════════════════════════════════════════════════════════

def get_memory_usage() -> Dict[str, float]:
    """
    Get current memory usage in MB.
    
    Returns:
        Dictionary with memory metrics:
        - rss_mb: Resident Set Size (physical memory)
        - vms_mb: Virtual Memory Size
        - percent: Memory usage percentage
    
    Example:
        >>> mem = get_memory_usage()
        >>> print(f"Using {mem['rss_mb']:.1f} MB")
    """
    try:
        import psutil
        process = psutil.Process()
        mem_info = process.memory_info()
        mem_percent = process.memory_percent()
        
        return {
            "rss_mb": mem_info.rss / (1024 * 1024),
            "vms_mb": mem_info.vms / (1024 * 1024),
            "percent": mem_percent,
        }
    except ImportError:
        logger.warning("psutil not available for memory stats")
        return {"rss_mb": 0.0, "vms_mb": 0.0, "percent": 0.0}


def get_gpu_memory_usage() -> Dict[str, float]:
    """
    Get GPU memory usage if available.
    
    Returns:
        Dictionary with GPU memory metrics:
        - allocated_mb: Allocated memory
        - reserved_mb: Reserved memory
        - free_mb: Free memory
        - total_mb: Total memory
    """
    try:
        import torch
        if torch.cuda.is_available():
            return {
                "allocated_mb": torch.cuda.memory_allocated() / (1024 * 1024),
                "reserved_mb": torch.cuda.memory_reserved() / (1024 * 1024),
                "free_mb": (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved()) / (1024 * 1024),
                "total_mb": torch.cuda.get_device_properties(0).total_memory / (1024 * 1024),
            }
    except ImportError:
        pass
    
    return {"allocated_mb": 0.0, "reserved_mb": 0.0, "free_mb": 0.0, "total_mb": 0.0}


@contextmanager
def memory_monitor(description: str = "Operation"):
    """
    Context manager for monitoring memory usage.
    
    Args:
        description: Description of the operation
    
    Example:
        >>> with memory_monitor("Processing"):
        >>>     process_data()
    """
    mem_before = get_memory_usage()
    try:
        yield
    finally:
        mem_after = get_memory_usage()
        delta = mem_after["rss_mb"] - mem_before["rss_mb"]
        logger.info(
            f"{description} memory: {mem_before['rss_mb']:.1f} MB -> "
            f"{mem_after['rss_mb']:.1f} MB (Δ {delta:+.1f} MB)"
        )


# ════════════════════════════════════════════════════════════════════════════════
# STATISTICAL CALCULATIONS
# ════════════════════════════════════════════════════════════════════════════════

def calculate_throughput(
    num_tokens: int,
    elapsed_seconds: float
) -> float:
    """
    Calculate tokens per second.
    
    Args:
        num_tokens: Number of tokens processed
        elapsed_seconds: Time elapsed in seconds
    
    Returns:
        Tokens per second
    """
    if elapsed_seconds > 0:
        return num_tokens / elapsed_seconds
    return 0.0


def calculate_percentiles(
    values: List[float],
    percentiles: List[float] = [50, 75, 90, 95, 99]
) -> Dict[str, float]:
    """
    Calculate percentiles from list of values.
    
    Args:
        values: List of numeric values
        percentiles: List of percentiles to calculate (0-100)
    
    Returns:
        Dictionary mapping percentile names to values
    
    Example:
        >>> values = [1.0, 2.0, 3.0, 4.0, 5.0]
        >>> stats = calculate_percentiles(values)
        >>> print(stats['p50'])  # Median
    """
    if not values:
        return {}
    
    sorted_values = sorted(values)
    n = len(sorted_values)
    
    results = {}
    for p in percentiles:
        if p < 0 or p > 100:
            continue
        
        index = int(n * (p / 100.0))
        index = min(index, n - 1)
        results[f"p{p}"] = sorted_values[index]
    
    # Add min, max, mean
    results["min"] = sorted_values[0]
    results["max"] = sorted_values[n - 1]
    results["mean"] = sum(values) / n
    results["median"] = results.get("p50", sorted_values[n // 2])
    
    # Calculate standard deviation
    mean = results["mean"]
    variance = sum((x - mean) ** 2 for x in values) / n
    results["std_dev"] = variance ** 0.5
    
    return results


def calculate_statistics(values: List[float]) -> Dict[str, float]:
    """
    Calculate comprehensive statistics.
    
    Args:
        values: List of numeric values
    
    Returns:
        Dictionary with statistical measures
    """
    return calculate_percentiles(values)


# ════════════════════════════════════════════════════════════════════════════════
# PROGRESS TRACKING
# ════════════════════════════════════════════════════════════════════════════════

class ProgressTracker:
    """
    Progress tracker for long-running operations.
    
    Example:
        >>> tracker = ProgressTracker(total=100)
        >>> for i in range(100):
        >>>     tracker.update(1)
        >>>     if tracker.should_log():
        >>>         print(f"Progress: {tracker.percent:.1f}%")
    """
    
    def __init__(
        self,
        total: int,
        log_interval: int = 10,
        description: str = "Progress"
    ):
        """
        Initialize progress tracker.
        
        Args:
            total: Total number of items
            log_interval: Log every N items
            description: Description for logging
        """
        self.total = total
        self.current = 0
        self.log_interval = log_interval
        self.description = description
        self.start_time = time.time()
    
    def update(self, increment: int = 1) -> None:
        """Update progress."""
        self.current += increment
    
    @property
    def percent(self) -> float:
        """Get completion percentage."""
        if self.total == 0:
            return 0.0
        return (self.current / self.total) * 100.0
    
    @property
    def elapsed(self) -> float:
        """Get elapsed time in seconds."""
        return time.time() - self.start_time
    
    @property
    def eta(self) -> float:
        """Get estimated time to completion in seconds."""
        if self.current == 0:
            return 0.0
        rate = self.current / self.elapsed
        if rate == 0:
            return 0.0
        return (self.total - self.current) / rate
    
    def should_log(self) -> bool:
        """Check if should log progress."""
        return self.current % self.log_interval == 0 or self.current == self.total
    
    def get_status(self) -> str:
        """Get status string."""
        return (
            f"{self.description}: {self.current}/{self.total} "
            f"({self.percent:.1f}%) "
            f"ETA: {format_duration(self.eta)}"
        )


# ════════════════════════════════════════════════════════════════════════════════
# EXPORTS
# ════════════════════════════════════════════════════════════════════════════════

__all__ = [
    # Time utilities
    "measure_time",
    "timer",
    "format_size",
    "format_duration",
    "format_timestamp",
    # File operations
    "save_results",
    "load_results",
    "ensure_directory",
    "get_file_hash",
    # Retry mechanisms
    "retry_on_failure",
    # Memory monitoring
    "get_memory_usage",
    "get_gpu_memory_usage",
    "memory_monitor",
    # Statistical calculations
    "calculate_throughput",
    "calculate_percentiles",
    "calculate_statistics",
    # Progress tracking
    "ProgressTracker",
]
