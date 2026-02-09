"""
Timeout Utilities
=================

Timeout decorators and utilities for async operations.
"""

import asyncio
import functools
from typing import Callable, Any, Optional
import logging

logger = logging.getLogger(__name__)


def timeout(seconds: float):
    """
    Timeout decorator for async functions.
    
    Args:
        seconds: Timeout in seconds
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            try:
                return await asyncio.wait_for(func(*args, **kwargs), timeout=seconds)
            except asyncio.TimeoutError:
                logger.error(
                    f"Function {func.__name__} timed out after {seconds}s",
                    extra={
                        "function": func.__name__,
                        "timeout": seconds,
                    },
                )
                raise TimeoutError(f"Operation timed out after {seconds} seconds")

        return async_wrapper

    return decorator


class TimeoutContext:
    """Context manager for timeouts."""

    def __init__(self, timeout: float):
        self.timeout = timeout
        self.task: Optional[asyncio.Task] = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass

    async def run(self, coro):
        """Run coroutine with timeout."""
        self.task = asyncio.create_task(coro)
        try:
            return await asyncio.wait_for(self.task, timeout=self.timeout)
        except asyncio.TimeoutError:
            self.task.cancel()
            raise TimeoutError(f"Operation timed out after {self.timeout} seconds")




