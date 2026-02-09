"""
Decorator for automatic metrics collection on API endpoints.
"""

import time
import functools
from typing import Callable, Any, Awaitable
from ..utils.metrics import metrics_collector


def track_metrics(endpoint_name: str):
    """
    Decorator to automatically track metrics for an API endpoint.
    
    Args:
        endpoint_name: Name of the endpoint for metrics tracking
    """
    def decorator(func: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.time()
            success = True
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                raise
            finally:
                duration = time.time() - start_time
                metrics_collector.record_request(endpoint_name, duration, success=success)
        
        return wrapper
    return decorator






