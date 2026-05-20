"""
ReAct Retry Executor

Handles retry logic for tool execution with exponential backoff.
Provides reusable retry mechanism for any callable function.
"""

import time
import logging
from typing import Callable, Dict, Any, Optional

from .react_constants import Defaults, ErrorMessages

logger = logging.getLogger(__name__)


class RetryExecutor:
    """
    Executor with retry logic and exponential backoff.
    
    Provides a reusable mechanism for executing functions with
    automatic retry on failure.
    """
    
    def __init__(
        self,
        max_retries: int = Defaults.MAX_RETRIES,
        retry_on_error: bool = True,
        timeout: Optional[float] = None,
        base_delay: float = Defaults.RETRY_DELAY_BASE
    ):
        """
        Initialize retry executor.
        
        Args:
            max_retries: Maximum number of retry attempts
            retry_on_error: Whether to retry on errors
            timeout: Optional timeout per attempt (not fully implemented, placeholder)
            base_delay: Base delay for exponential backoff
        """
        self.max_retries = max_retries
        self.retry_on_error = retry_on_error
        self.timeout = timeout
        self.base_delay = base_delay
    
    def execute_with_retry(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute function with retry logic.
        
        Args:
            func: Function to execute
            *args: Positional arguments for function
            **kwargs: Keyword arguments for function
            
        Returns:
            Result dictionary with 'success' and 'result'/'error' keys
        """
        last_error = None
        
        for attempt in range(self.max_retries if self.retry_on_error else 1):
            try:
                # Execute function
                result = func(*args, **kwargs)
                
                # If result is a dict with success key, check it
                if isinstance(result, dict) and result.get("success"):
                    return result
                elif isinstance(result, dict):
                    # Result dict but not successful
                    last_error = result.get("error", "Unknown error")
                    if not self.retry_on_error:
                        break
                else:
                    # Non-dict result, assume success
                    return {
                        "success": True,
                        "result": result
                    }
                
            except Exception as e:
                last_error = str(e)
                logger.warning(
                    f"Execution attempt {attempt + 1} failed: {e}"
                )
                
                # Don't retry on last attempt
                if attempt < self.max_retries - 1:
                    # Exponential backoff
                    delay = self.base_delay * (attempt + 1)
                    time.sleep(delay)
        
        # All retries exhausted
        return {
            "success": False,
            "error": ErrorMessages.EXECUTION_FAILED.format(
                attempts=self.max_retries,
                error=last_error or "Unknown error"
            ),
            "result": None
        }



