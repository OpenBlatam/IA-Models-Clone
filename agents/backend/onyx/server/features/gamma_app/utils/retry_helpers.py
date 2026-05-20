"""
Retry Helpers
Utility functions for retry logic using tenacity
"""

from typing import Callable, TypeVar, Any
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    RetryCallState
)
import logging

logger = logging.getLogger(__name__)

T = TypeVar('T')

def retry_redis_operation(
    max_attempts: int = 3,
    min_wait: float = 1.0,
    max_wait: float = 10.0
):
    """Decorator for retrying Redis operations"""
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=1, min=min_wait, max=max_wait),
        retry=retry_if_exception_type((ConnectionError, TimeoutError, OSError)),
        reraise=True
    )

def retry_api_call(
    max_attempts: int = 3,
    min_wait: float = 2.0,
    max_wait: float = 30.0
):
    """Decorator for retrying API calls"""
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=2, min=min_wait, max=max_wait),
        retry=retry_if_exception_type((ConnectionError, TimeoutError)),
        reraise=True
    )







