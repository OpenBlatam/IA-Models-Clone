"""
Async helper utilities.
"""

from typing import List, Callable, Any, Coroutine
import asyncio
import logging

logger = logging.getLogger(__name__)


async def run_in_parallel(
    coroutines: List[Coroutine],
    max_concurrent: int = 10
) -> List[Any]:
    """
    Run multiple coroutines in parallel with concurrency limit.
    
    Args:
        coroutines: List of coroutines to run
        max_concurrent: Maximum concurrent executions
        
    Returns:
        List of results
    """
    results = []
    
    for i in range(0, len(coroutines), max_concurrent):
        batch = coroutines[i:i + max_concurrent]
        batch_results = await asyncio.gather(*batch, return_exceptions=True)
        
        for result in batch_results:
            if isinstance(result, Exception):
                logger.error(f"Error in parallel execution: {result}")
                results.append(None)
            else:
                results.append(result)
    
    return results


async def retry_async(
    func: Callable,
    max_retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0
) -> Any:
    """
    Retry an async function with exponential backoff.
    
    Args:
        func: Async function to retry
        max_retries: Maximum number of retries
        delay: Initial delay in seconds
        backoff: Backoff multiplier
        
    Returns:
        Function result
        
    Raises:
        Exception: If all retries fail
    """
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            return await func()
        except Exception as e:
            last_exception = e
            if attempt < max_retries - 1:
                await asyncio.sleep(delay * (backoff ** attempt))
                logger.warning(f"Retry {attempt + 1}/{max_retries} after error: {e}")
            else:
                logger.error(f"All retries failed: {e}")
    
    raise last_exception


async def timeout_async(
    coro: Coroutine,
    timeout: float
) -> Any:
    """
    Execute a coroutine with timeout.
    
    Args:
        coro: Coroutine to execute
        timeout: Timeout in seconds
        
    Returns:
        Coroutine result
        
    Raises:
        asyncio.TimeoutError: If timeout is exceeded
    """
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        logger.error(f"Operation timed out after {timeout}s")
        raise


# Alias for backward compatibility
run_parallel = run_in_parallel






