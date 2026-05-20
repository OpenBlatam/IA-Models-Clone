"""
Retry helpers for resilient API calls.
"""

import asyncio
import logging
from typing import Callable, Type, Tuple, Any

logger = logging.getLogger(__name__)

DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 1.0


async def retry_with_exponential_backoff(
    func: Callable,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_delay: float = DEFAULT_RETRY_DELAY,
    retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,),
    operation_name: str = "operation",
) -> Any:
    """
    Retry a function with exponential backoff.
    
    Args:
        func: Async function to retry
        max_retries: Maximum number of retries
        retry_delay: Initial delay between retries (seconds)
        retryable_exceptions: Tuple of exception types to retry on
        operation_name: Name of operation for logging
        
    Returns:
        Function result
        
    Raises:
        Last exception if all retries fail
    """
    last_exception = None
    
    for attempt in range(max_retries + 1):
        try:
            return await func()
        except retryable_exceptions as e:
            last_exception = e
            
            if attempt < max_retries:
                delay = retry_delay * (2 ** attempt)
                logger.warning(
                    f"{operation_name} failed (attempt {attempt + 1}/{max_retries + 1}): {e}. "
                    f"Retrying in {delay}s..."
                )
                await asyncio.sleep(delay)
            else:
                logger.error(f"{operation_name} failed after {max_retries + 1} attempts")
    
    raise last_exception


