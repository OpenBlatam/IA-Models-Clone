"""
Retry Handler
=============
Handles retry logic with exponential backoff
"""

import asyncio
import logging
import random
from typing import Callable, TypeVar, Optional
import httpx

logger = logging.getLogger(__name__)

T = TypeVar('T')


class RetryHandler:
    """
    Handles retry logic with exponential backoff.
    """
    
    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True
    ):
        """
        Initialize retry handler.
        
        Args:
            max_retries: Maximum number of retry attempts
            base_delay: Base delay in seconds
            max_delay: Maximum delay in seconds
            exponential_base: Base for exponential backoff
            jitter: Whether to add random jitter to delays
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
    
    def calculate_delay(self, attempt: int) -> float:
        """
        Calculate retry delay for attempt.
        
        Args:
            attempt: Current attempt number (0-indexed)
            
        Returns:
            Delay in seconds
        """
        delay = self.base_delay * (self.exponential_base ** attempt)
        delay = min(delay, self.max_delay)
        
        if self.jitter:
            # Add random jitter (±25%)
            jitter_amount = delay * 0.25
            delay += random.uniform(-jitter_amount, jitter_amount)
            delay = max(0.1, delay)  # Ensure minimum delay
        
        return delay
    
    def extract_error_message(self, error: httpx.HTTPStatusError) -> str:
        """
        Extract error message from HTTP error.
        
        Args:
            error: HTTP status error
            
        Returns:
            Error message string
        """
        try:
            response_data = error.response.json()
            if isinstance(response_data, dict):
                return response_data.get('error', response_data.get('message', str(error)))
            return str(response_data)
        except Exception:
            return f"HTTP {error.response.status_code}: {error.response.text[:200]}"
    
    async def execute_with_retry(
        self,
        func: Callable[[], T],
        retry_on: Optional[Callable[[Exception], bool]] = None
    ) -> T:
        """
        Execute function with retry logic.
        
        Args:
            func: Async function to execute
            retry_on: Optional function to determine if error should be retried
            
        Returns:
            Function result
            
        Raises:
            Exception: If all retries fail
        """
        last_error: Optional[Exception] = None
        
        for attempt in range(self.max_retries):
            try:
                return await func()
            except Exception as e:
                last_error = e
                
                # Check if we should retry this error
                if retry_on and not retry_on(e):
                    raise
                
                # Check if we should retry based on error type
                if not self._should_retry(e):
                    raise
                
                if attempt < self.max_retries - 1:
                    wait_time = self.calculate_delay(attempt)
                    error_msg = self._get_error_message(e)
                    logger.warning(
                        f"Attempt {attempt + 1} failed: {error_msg}. "
                        f"Retrying in {wait_time:.1f}s..."
                    )
                    await asyncio.sleep(wait_time)
                else:
                    error_msg = self._get_error_message(e)
                    logger.error(f"All {self.max_retries} attempts failed: {error_msg}")
                    raise Exception(
                        f"Operation failed after {self.max_retries} attempts: {error_msg}"
                    ) from e
        
        # Should never reach here, but for type safety
        if last_error:
            raise last_error
        raise Exception("Unexpected error in retry handler")
    
    def _should_retry(self, error: Exception) -> bool:
        """
        Determine if error should be retried.
        
        Args:
            error: Exception to check
            
        Returns:
            True if error should be retried
        """
        # Retry on network errors and 5xx errors
        if isinstance(error, (httpx.TimeoutException, httpx.NetworkError)):
            return True
        
        if isinstance(error, httpx.HTTPStatusError):
            # Retry on server errors (5xx) and rate limiting (429)
            status = error.response.status_code
            return 500 <= status < 600 or status == 429
        
        return False
    
    def _get_error_message(self, error: Exception) -> str:
        """
        Get error message from exception.
        
        Args:
            error: Exception to extract message from
            
        Returns:
            Error message string
        """
        if isinstance(error, httpx.HTTPStatusError):
            return self.extract_error_message(error)
        return str(error)

