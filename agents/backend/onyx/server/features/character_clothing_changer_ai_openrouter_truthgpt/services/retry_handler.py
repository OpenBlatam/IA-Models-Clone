"""
Retry Handler
=============

Handles retry logic with exponential backoff for API requests.
"""

import logging
import random
import asyncio
from typing import Callable, Any, Optional
import httpx

logger = logging.getLogger(__name__)

DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 1.0


class RetryHandler:
    """Handles retry logic for API requests."""
    
    def __init__(
        self,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_delay: float = DEFAULT_RETRY_DELAY
    ):
        """
        Initialize Retry Handler.
        
        Args:
            max_retries: Maximum number of retry attempts
            retry_delay: Base delay between retries
        """
        self.max_retries = max_retries
        self.retry_delay = retry_delay
    
    def calculate_retry_delay(self, attempt: int) -> float:
        """
        Calculate retry delay with exponential backoff and jitter.
        
        Args:
            attempt: Current attempt number (0-indexed)
            
        Returns:
            Delay in seconds
        """
        base_delay = self.retry_delay * (2 ** attempt)
        jitter = random.uniform(0, base_delay * 0.1)
        return base_delay + jitter
    
    def extract_error_message(self, error: httpx.HTTPStatusError) -> str:
        """
        Extract error message from HTTP error.
        
        Args:
            error: HTTP status error
            
        Returns:
            Error message string
        """
        try:
            error_data = error.response.json()
            if isinstance(error_data, dict):
                return error_data.get("error", {}).get("message", str(error))
            return str(error_data)
        except Exception:
            return str(error)
    
    async def execute_with_retry(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Execute function with retry logic.
        
        Args:
            func: Async function to execute
            *args: Positional arguments for function
            **kwargs: Keyword arguments for function
            
        Returns:
            Function result
            
        Raises:
            httpx.HTTPStatusError: If request fails after retries
            httpx.TimeoutException: If request times out after retries
            Exception: For other unexpected errors
        """
        last_error = None
        
        for attempt in range(self.max_retries + 1):
            try:
                return await func(*args, **kwargs)
            except httpx.HTTPStatusError as e:
                last_error = e
                if attempt < self.max_retries:
                    delay = self.calculate_retry_delay(attempt)
                    error_msg = self.extract_error_message(e)
                    logger.warning(
                        f"Request failed (attempt {attempt + 1}/{self.max_retries + 1}): "
                        f"{error_msg}. Retrying in {delay:.2f}s..."
                    )
                    await asyncio.sleep(delay)
                else:
                    error_msg = self.extract_error_message(e)
                    logger.error(f"Request failed after {self.max_retries + 1} attempts: {error_msg}")
                    raise
            except httpx.TimeoutException as e:
                last_error = e
                if attempt < self.max_retries:
                    delay = self.calculate_retry_delay(attempt)
                    logger.warning(
                        f"Request timed out (attempt {attempt + 1}/{self.max_retries + 1}). "
                        f"Retrying in {delay:.2f}s..."
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"Request timed out after {self.max_retries + 1} attempts")
                    raise
        
        # Should never reach here, but just in case
        if last_error:
            raise last_error

