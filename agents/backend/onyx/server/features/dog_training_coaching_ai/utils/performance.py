"""
Performance Utilities
=====================
"""

import asyncio
from functools import wraps
from typing import Callable, Any
import time

from .logger import get_logger

logger = get_logger(__name__)


def async_timing(func: Callable) -> Callable:
    """Decorator para medir tiempo de ejecución."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start = time.time()
        try:
            result = await func(*args, **kwargs)
            duration = time.time() - start
            logger.info(
                "function_timing",
                function=func.__name__,
                duration=round(duration, 3),
                status="success"
            )
            return result
        except Exception as e:
            duration = time.time() - start
            logger.error(
                "function_timing",
                function=func.__name__,
                duration=round(duration, 3),
                status="error",
                error=str(e)
            )
            raise
    return wrapper


async def batch_process(items: list, func: Callable, batch_size: int = 10) -> list:
    """Procesar items en batches."""
    results = []
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        batch_results = await asyncio.gather(*[func(item) for item in batch])
        results.extend(batch_results)
    return results

