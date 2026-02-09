"""Memory utilities."""

import sys
from typing import Any


def get_size(obj: Any) -> int:
    """
    Get size of object in bytes.
    
    Args:
        obj: Object to measure
        
    Returns:
        Size in bytes
    """
    return sys.getsizeof(obj)


def get_deep_size(obj: Any) -> int:
    """
    Get deep size of object (including nested objects).
    
    Args:
        obj: Object to measure
        
    Returns:
        Total size in bytes
    """
    size = sys.getsizeof(obj)
    
    if isinstance(obj, dict):
        size += sum(get_deep_size(k) + get_deep_size(v) for k, v in obj.items())
    elif isinstance(obj, (list, tuple, set)):
        size += sum(get_deep_size(item) for item in obj)
    elif hasattr(obj, '__dict__'):
        size += get_deep_size(obj.__dict__)
    
    return size


def format_size(bytes_value: int) -> str:
    """
    Format bytes to human-readable size.
    
    Args:
        bytes_value: Bytes to format
        
    Returns:
        Formatted string
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.2f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.2f} PB"


def get_memory_usage() -> dict:
    """
    Get current memory usage.
    
    Returns:
        Dictionary with memory information
    """
    try:
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        
        return {
            "rss": mem_info.rss,  # Resident Set Size
            "vms": mem_info.vms,  # Virtual Memory Size
            "percent": process.memory_percent(),
        }
    except ImportError:
        # psutil not available
        return {
            "rss": 0,
            "vms": 0,
            "percent": 0.0,
        }


def clear_cache() -> None:
    """
    Clear Python cache (gc.collect).
    """
    import gc
    gc.collect()

