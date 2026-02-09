"""
Async Optimizer
===============

Optimizations for async operations.
"""

import logging
import asyncio
from typing import Any, Callable, List, Optional
from functools import wraps
import time

logger = logging.getLogger(__name__)


class AsyncOptimizer:
    """Optimizer for async operations."""
    
    @staticmethod
    def batch_operations(
        operations: List[Callable],
        batch_size: int = 10,
        max_concurrent: int = 5
    ):
        """Batch async operations for better performance."""
        async def run_batches():
            results = []
            for i in range(0, len(operations), batch_size):
                batch = operations[i:i + batch_size]
                batch_results = await asyncio.gather(
                    *[op() if asyncio.iscoroutinefunction(op) else asyncio.to_thread(op) 
                      for op in batch],
                    return_exceptions=True
                )
                results.extend(batch_results)
            return results
        return run_batches()
    
    @staticmethod
    def with_timeout(timeout: float):
        """Add timeout to async function."""
        def decorator(func: Callable):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                try:
                    return await asyncio.wait_for(
                        func(*args, **kwargs),
                        timeout=timeout
                    )
                except asyncio.TimeoutError:
                    logger.error(f"Operation {func.__name__} timed out after {timeout}s")
                    raise
            return wrapper
        return decorator
    
    @staticmethod
    def with_retry(max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
        """Add retry logic to async function."""
        def decorator(func: Callable):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                last_exception = None
                current_delay = delay
                
                for attempt in range(max_retries):
                    try:
                        return await func(*args, **kwargs)
                    except Exception as e:
                        last_exception = e
                        if attempt < max_retries - 1:
                            logger.warning(
                                f"Attempt {attempt + 1} failed for {func.__name__}: {e}. "
                                f"Retrying in {current_delay}s..."
                            )
                            await asyncio.sleep(current_delay)
                            current_delay *= backoff
                        else:
                            logger.error(f"All {max_retries} attempts failed for {func.__name__}")
                
                raise last_exception
            return wrapper
        return decorator
    
    @staticmethod
    def with_circuit_breaker(
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: type = Exception
    ):
        """Add circuit breaker pattern."""
        def decorator(func: Callable):
            failures = [0]  # Use list to allow modification in nested function
            last_failure_time = [None]
            state = ["closed"]  # closed, open, half_open
            
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # Check circuit state
                if state[0] == "open":
                    if last_failure_time[0] and time.time() - last_failure_time[0] > recovery_timeout:
                        state[0] = "half_open"
                        logger.info(f"Circuit breaker for {func.__name__} entering half-open state")
                    else:
                        raise Exception(f"Circuit breaker is OPEN for {func.__name__}")
                
                try:
                    result = await func(*args, **kwargs)
                    if state[0] == "half_open":
                        state[0] = "closed"
                        failures[0] = 0
                        logger.info(f"Circuit breaker for {func.__name__} closed")
                    return result
                except expected_exception as e:
                    failures[0] += 1
                    last_failure_time[0] = time.time()
                    
                    if failures[0] >= failure_threshold:
                        state[0] = "open"
                        logger.error(f"Circuit breaker for {func.__name__} opened after {failures[0]} failures")
                    
                    raise
            return wrapper
        return decorator
    
    @staticmethod
    def parallel_execute(tasks: List[Callable], max_concurrent: Optional[int] = None):
        """Execute tasks in parallel with concurrency limit."""
        async def run():
            semaphore = asyncio.Semaphore(max_concurrent) if max_concurrent else None
            
            async def run_with_semaphore(task):
                if semaphore:
                    async with semaphore:
                        return await task() if asyncio.iscoroutinefunction(task) else await asyncio.to_thread(task)
                else:
                    return await task() if asyncio.iscoroutinefunction(task) else await asyncio.to_thread(task)
            
            return await asyncio.gather(
                *[run_with_semaphore(task) for task in tasks],
                return_exceptions=True
            )
        
        return run()

