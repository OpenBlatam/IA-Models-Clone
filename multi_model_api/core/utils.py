"""
Utility functions for Multi-Model API
Common utilities and helpers
"""

import time
import logging
import hashlib
import json
from typing import Any, Dict, Optional, Callable, Awaitable
from contextlib import asynccontextmanager
from functools import wraps

logger = logging.getLogger(__name__)


def generate_request_id() -> str:
    """
    Generate a unique request ID
    
    Returns:
        Unique request identifier
    """
    import uuid
    return str(uuid.uuid4())


def hash_string(text: str, length: int = 16) -> str:
    """
    Generate hash from string
    
    Args:
        text: Text to hash
        length: Length of hash to return
        
    Returns:
        Hash string
    """
    return hashlib.sha256(text.encode()).hexdigest()[:length]


def safe_json_dumps(obj: Any, default: Optional[Callable] = None) -> str:
    """
    Safely serialize object to JSON
    
    Args:
        obj: Object to serialize
        default: Optional default serializer
        
    Returns:
        JSON string
    """
    try:
        import orjson
        return orjson.dumps(obj, default=default).decode()
    except (ImportError, TypeError):
        return json.dumps(obj, default=default or str)


def safe_json_loads(text: str) -> Any:
    """
    Safely deserialize JSON string
    
    Args:
        text: JSON string
        
    Returns:
        Deserialized object
    """
    try:
        import orjson
        return orjson.loads(text)
    except (ImportError, TypeError):
        return json.loads(text)


def format_latency_ms(latency_ms: Optional[float]) -> str:
    """
    Format latency in milliseconds to human-readable string
    
    Args:
        latency_ms: Latency in milliseconds
        
    Returns:
        Formatted string
    """
    if latency_ms is None:
        return "N/A"
    
    if latency_ms < 1:
        return f"{latency_ms * 1000:.2f}μs"
    elif latency_ms < 1000:
        return f"{latency_ms:.2f}ms"
    else:
        return f"{latency_ms / 1000:.2f}s"


def format_bytes(bytes_count: int) -> str:
    """
    Format bytes to human-readable string
    
    Args:
        bytes_count: Number of bytes
        
    Returns:
        Formatted string
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_count < 1024.0:
            return f"{bytes_count:.2f}{unit}"
        bytes_count /= 1024.0
    return f"{bytes_count:.2f}PB"


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate text to maximum length
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


@asynccontextmanager
async def timer(operation_name: str = "operation", logger_instance: Optional[logging.Logger] = None):
    """
    Context manager for timing async operations
    
    Args:
        operation_name: Name of operation
        logger_instance: Optional logger instance
        
    Yields:
        Timer context
        
    Example:
        async with timer("model_execution") as t:
            result = await execute_model()
        print(f"Took {t.elapsed_ms}ms")
    """
    log = logger_instance or logger
    start_time = time.time()
    
    class TimerContext:
        def __init__(self):
            self.start_time = start_time
            self.elapsed_ms = 0.0
        
        def update(self):
            self.elapsed_ms = (time.time() - self.start_time) * 1000
    
    timer_ctx = TimerContext()
    
    try:
        yield timer_ctx
    finally:
        timer_ctx.update()
        log.debug(
            f"{operation_name} completed in {timer_ctx.elapsed_ms:.2f}ms",
            extra={"operation": operation_name, "latency_ms": timer_ctx.elapsed_ms}
        )


def retry_on_exception(
    max_attempts: int = 3,
    delay: float = 1.0,
    exceptions: tuple = (Exception,)
):
    """
    Decorator for retrying functions on exception
    
    Args:
        max_attempts: Maximum number of attempts
        delay: Delay between attempts in seconds
        exceptions: Tuple of exceptions to catch
        
    Returns:
        Decorated function
    """
    import asyncio
    import inspect
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        logger.warning(
                            f"{func.__name__} failed (attempt {attempt + 1}/{max_attempts}): {e}"
                        )
                        await asyncio.sleep(delay * (2 ** attempt))
                    else:
                        logger.error(f"{func.__name__} failed after {max_attempts} attempts")
            if last_exception:
                raise last_exception
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        logger.warning(
                            f"{func.__name__} failed (attempt {attempt + 1}/{max_attempts}): {e}"
                        )
                        time.sleep(delay * (2 ** attempt))
                    else:
                        logger.error(f"{func.__name__} failed after {max_attempts} attempts")
            if last_exception:
                raise last_exception
        
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


def calculate_percentage(part: float, total: float) -> float:
    """
    Calculate percentage
    
    Args:
        part: Part value
        total: Total value
        
    Returns:
        Percentage (0-100)
    """
    if total == 0:
        return 0.0
    return (part / total) * 100.0


def calculate_success_rate(success_count: int, total_count: int) -> float:
    """
    Calculate success rate percentage
    
    Args:
        success_count: Number of successes
        total_count: Total number of attempts
        
    Returns:
        Success rate (0-100)
    """
    return calculate_percentage(success_count, total_count)


def merge_dicts(*dicts: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge multiple dictionaries
    
    Args:
        *dicts: Dictionaries to merge
        
    Returns:
        Merged dictionary
    """
    result = {}
    for d in dicts:
        if d:
            result.update(d)
    return result


def get_nested_value(data: Dict[str, Any], path: str, default: Any = None) -> Any:
    """
    Get nested value from dictionary using dot notation
    
    Args:
        data: Dictionary to search
        path: Dot-separated path (e.g., "cache.l1.size")
        default: Default value if not found
        
    Returns:
        Value or default
    """
    keys = path.split('.')
    value = data
    for key in keys:
        if isinstance(value, dict):
            value = value.get(key, default)
            if value is default:
                return default
        else:
            return default
    return value


def sanitize_for_logging(text: str, max_length: int = 200) -> str:
    """
    Sanitize text for safe logging
    
    Args:
        text: Text to sanitize
        max_length: Maximum length
        
    Returns:
        Sanitized text
    """
    # Remove newlines and control characters
    sanitized = ''.join(char for char in text if ord(char) >= 32 or char == '\n')
    sanitized = sanitized.replace('\n', ' ')
    
    # Truncate if too long
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length] + "..."
    
    return sanitized

