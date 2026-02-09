"""
Retry Utilities
===============

Utilities for retrying operations on failure.
"""

import logging
import time
from functools import wraps

logger = logging.getLogger(__name__)


def retry_on_failure(max_attempts: int = 3, delay: float = 0.5):
    """Decorator to retry operations on failure."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        logger.warning(
                            f"Attempt {attempt + 1} failed for {func.__name__}: {e}. Retrying..."
                        )
                        time.sleep(delay * (attempt + 1))
                    else:
                        logger.error(f"All {max_attempts} attempts failed for {func.__name__}")
            raise last_exception
        return wrapper
    return decorator
