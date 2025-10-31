"""Enhanced functional utilities with performance optimizations."""

from typing import Dict, Any, List, Optional, Callable, TypeVar, Union
from functools import wraps, lru_cache, partial
import asyncio
import hashlib
import json
import re
from datetime import datetime, timedelta
from pathlib import Path

T = TypeVar('T')


def memoize_async(maxsize: int = 128):
    """Async memoization decorator."""
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        cache = {}
        
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Create cache key
            key = str(args) + str(sorted(kwargs.items()))
            key_hash = hashlib.md5(key.encode()).hexdigest()
            
            if key_hash in cache:
                return cache[key_hash]
            
            result = await func(*args, **kwargs)
            cache[key_hash] = result
            
            # Limit cache size
            if len(cache) > maxsize:
                # Remove oldest entry
                oldest_key = next(iter(cache))
                del cache[oldest_key]
            
            return result
        
        return wrapper
    return decorator


def pipe(*functions: Callable) -> Callable:
    """Create a function pipeline."""
    def pipeline(value: T) -> T:
        for func in functions:
            value = func(value)
        return value
    return pipeline


def async_pipe(*functions: Callable) -> Callable:
    """Create an async function pipeline."""
    async def pipeline(value: T) -> T:
        for func in functions:
            value = await func(value)
        return value
    return pipeline


def safe_get(data: Dict[str, Any], key_path: str, default: Any = None) -> Any:
    """Safely get nested dictionary value using dot notation."""
    keys = key_path.split('.')
    current = data
    
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    
    return current


def safe_set(data: Dict[str, Any], key_path: str, value: Any) -> Dict[str, Any]:
    """Safely set nested dictionary value using dot notation."""
    keys = key_path.split('.')
    current = data
    
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    
    current[keys[-1]] = value
    return data


def chunk_list(items: List[T], chunk_size: int) -> List[List[T]]:
    """Split list into chunks of specified size."""
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


def flatten_dict(data: Dict[str, Any], separator: str = '.') -> Dict[str, Any]:
    """Flatten nested dictionary."""
    def _flatten(obj: Any, parent_key: str = '') -> Dict[str, Any]:
        items = []
        
        if isinstance(obj, dict):
            for key, value in obj.items():
                new_key = f"{parent_key}{separator}{key}" if parent_key else key
                items.extend(_flatten(value, new_key).items())
        else:
            items.append((parent_key, obj))
        
        return dict(items)
    
    return _flatten(data)


def unflatten_dict(data: Dict[str, Any], separator: str = '.') -> Dict[str, Any]:
    """Unflatten dictionary with dot notation keys."""
    result = {}
    
    for key, value in data.items():
        keys = key.split(separator)
        current = result
        
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        
        current[keys[-1]] = value
    
    return result


def sanitize_text(text: str, max_length: Optional[int] = None) -> str:
    """Sanitize text input."""
    if not text:
        return ""
    
    # Remove control characters
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Strip and limit length
    text = text.strip()
    if max_length and len(text) > max_length:
        text = text[:max_length]
    
    return text


def generate_id(prefix: str = "", length: int = 8) -> str:
    """Generate a unique ID."""
    import secrets
    random_part = secrets.token_hex(length)
    return f"{prefix}{random_part}" if prefix else random_part


def is_valid_email(email: str) -> bool:
    """Validate email format."""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))


def is_valid_url(url: str) -> bool:
    """Validate URL format."""
    pattern = r'^https?://[^\s/$.?#].[^\s]*$'
    return bool(re.match(pattern, url))


def extract_domain(email: str) -> Optional[str]:
    """Extract domain from email."""
    if not is_valid_email(email):
        return None
    return email.split('@')[1]


def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"


def format_duration(seconds: float) -> str:
    """Format duration in human readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def create_progress_tracker(total: int) -> Callable[[int], Dict[str, Any]]:
    """Create a progress tracking function."""
    def track(current: int) -> Dict[str, Any]:
        percentage = (current / total) * 100 if total > 0 else 0
        return {
            "current": current,
            "total": total,
            "percentage": round(percentage, 2),
            "remaining": total - current,
            "is_complete": current >= total
        }
    return track


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0
):
    """Retry decorator with exponential backoff."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    if attempt < max_retries - 1:
                        delay = min(
                            base_delay * (exponential_base ** attempt),
                            max_delay
                        )
                        await asyncio.sleep(delay)
            
            raise last_exception
        return wrapper
    return decorator


def create_circuit_breaker(
    failure_threshold: int = 5,
    recovery_timeout: float = 60.0
):
    """Create a circuit breaker decorator."""
    def decorator(func: Callable) -> Callable:
        state = {
            "failures": 0,
            "last_failure": None,
            "state": "closed"  # closed, open, half_open
        }
        
        @wraps(func)
        async def wrapper(*args, **kwargs):
            current_time = time.time()
            
            # Check if circuit is open
            if state["state"] == "open":
                if current_time - state["last_failure"] > recovery_timeout:
                    state["state"] = "half_open"
                else:
                    raise Exception("Circuit breaker is open")
            
            try:
                result = await func(*args, **kwargs)
                
                # Reset on success
                if state["state"] == "half_open":
                    state["state"] = "closed"
                    state["failures"] = 0
                
                return result
                
            except Exception as e:
                state["failures"] += 1
                state["last_failure"] = current_time
                
                if state["failures"] >= failure_threshold:
                    state["state"] = "open"
                
                raise e
        
        return wrapper
    return decorator


def create_batch_processor(
    batch_size: int = 10,
    max_concurrent: int = 5
):
    """Create a batch processor decorator."""
    def decorator(processor_func: Callable) -> Callable:
        @wraps(processor_func)
        async def wrapper(items: List[Any]) -> List[Any]:
            results = []
            
            for batch in chunk_list(items, batch_size):
                semaphore = asyncio.Semaphore(max_concurrent)
                
                async def process_item(item):
                    async with semaphore:
                        return await processor_func(item)
                
                batch_results = await asyncio.gather(
                    *[process_item(item) for item in batch],
                    return_exceptions=True
                )
                
                results.extend(batch_results)
            
            return results
        return wrapper
    return decorator


def create_metrics_collector():
    """Create a metrics collection function."""
    metrics = {
        "counters": {},
        "timers": {},
        "gauges": {}
    }
    
    def increment_counter(name: str, value: int = 1):
        metrics["counters"][name] = metrics["counters"].get(name, 0) + value
    
    def record_timer(name: str, duration: float):
        if name not in metrics["timers"]:
            metrics["timers"][name] = []
        metrics["timers"][name].append(duration)
    
    def set_gauge(name: str, value: float):
        metrics["gauges"][name] = value
    
    def get_metrics() -> Dict[str, Any]:
        return {
            "counters": metrics["counters"].copy(),
            "timers": {
                name: {
                    "count": len(times),
                    "avg": sum(times) / len(times) if times else 0,
                    "min": min(times) if times else 0,
                    "max": max(times) if times else 0
                }
                for name, times in metrics["timers"].items()
            },
            "gauges": metrics["gauges"].copy()
        }
    
    return {
        "increment_counter": increment_counter,
        "record_timer": record_timer,
        "set_gauge": set_gauge,
        "get_metrics": get_metrics
    }
