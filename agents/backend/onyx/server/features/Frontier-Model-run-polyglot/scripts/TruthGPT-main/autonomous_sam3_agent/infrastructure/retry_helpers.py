"""
Retry helper functions for resilient API calls.

This module consolidates retry logic with exponential backoff to eliminate
duplication across clients.
"""

import asyncio
import logging
from typing import Callable, TypeVar, Optional
from functools import wraps

logger = logging.getLogger(__name__)

T = TypeVar('T')


async def retry_with_exponential_backoff(
    func: Callable[[], T],
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: tuple = (Exception,),
    error_handler: Optional[Callable[[Exception, int], None]] = None,
) -> T:
    """
    Retry a function with exponential backoff.
    
    Args:
        func: Async function to retry
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds
        backoff_factor: Multiplier for delay on each retry
        exceptions: Tuple of exceptions to catch and retry
        error_handler: Optional function to handle errors (receives exception and attempt number)
    
    Returns:
        Function result
    
    Raises:
        Last exception if all retries fail
    """
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            return await func()
        except exceptions as e:
            last_exception = e
            
            if error_handler:
                error_handler(e, attempt)
            
            if attempt < max_retries - 1:
                delay = initial_delay * (backoff_factor ** attempt)
                logger.warning(f"Retry attempt {attempt + 1}/{max_retries} after {delay:.1f}s: {e}")
                await asyncio.sleep(delay)
                continue
            
            # Last attempt failed
            logger.error(f"All {max_retries} retry attempts failed")
            raise
    
    # Should never reach here, but for type safety
    if last_exception:
        raise last_exception
    raise RuntimeError("Retry logic error")
