"""
Async Coordinator Utilities for Piel Mejorador AI SAM3
======================================================

Unified async coordination and task management utilities.
"""

import asyncio
import logging
from typing import Callable, Any, List, Optional, TypeVar, Dict, Awaitable
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class TaskResult:
    """Task execution result."""
    task_id: str
    success: bool
    result: Any = None
    error: Optional[Exception] = None
    duration: float = 0.0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class AsyncCoordinator:
    """Unified async coordination utilities."""
    
    @staticmethod
    async def execute_with_timeout(
        coro: Awaitable[T],
        timeout: float,
        default: Optional[T] = None,
        timeout_handler: Optional[Callable[[], T]] = None
    ) -> Optional[T]:
        """
        Execute coroutine with timeout.
        
        Args:
            coro: Coroutine to execute
            timeout: Timeout in seconds
            default: Default value on timeout
            timeout_handler: Optional handler function for timeout
            
        Returns:
            Result or default
        """
        try:
            return await asyncio.wait_for(coro, timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning(f"Operation timed out after {timeout}s")
            if timeout_handler:
                if asyncio.iscoroutinefunction(timeout_handler):
                    return await timeout_handler()
                else:
                    return timeout_handler()
            return default
    
    @staticmethod
    async def gather_with_results(
        *coros: Awaitable[Any],
        return_exceptions: bool = True
    ) -> List[Any]:
        """
        Gather coroutines and return results with error handling.
        
        Args:
            *coros: Coroutines to gather
            return_exceptions: Whether to return exceptions
            
        Returns:
            List of results
        """
        results = await asyncio.gather(*coros, return_exceptions=return_exceptions)
        return results
    
    @staticmethod
    async def execute_parallel(
        tasks: List[Callable[[], Awaitable[T]]],
        max_concurrent: int = 5,
        timeout: Optional[float] = None
    ) -> List[TaskResult]:
        """
        Execute tasks in parallel with concurrency limit.
        
        Args:
            tasks: List of async task functions
            max_concurrent: Maximum concurrent executions
            timeout: Optional timeout per task
            
        Returns:
            List of TaskResult
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        results: List[TaskResult] = []
        
        async def execute_task(task_id: str, task: Callable[[], Awaitable[T]]) -> TaskResult:
            async with semaphore:
                started_at = datetime.now()
                try:
                    if timeout:
                        result = await AsyncCoordinator.execute_with_timeout(
                            task(),
                            timeout=timeout
                        )
                    else:
                        result = await task()
                    
                    completed_at = datetime.now()
                    duration = (completed_at - started_at).total_seconds()
                    
                    return TaskResult(
                        task_id=task_id,
                        success=True,
                        result=result,
                        duration=duration,
                        started_at=started_at,
                        completed_at=completed_at
                    )
                except Exception as e:
                    completed_at = datetime.now()
                    duration = (completed_at - started_at).total_seconds()
                    
                    logger.error(f"Task {task_id} failed: {e}")
                    return TaskResult(
                        task_id=task_id,
                        success=False,
                        error=e,
                        duration=duration,
                        started_at=started_at,
                        completed_at=completed_at
                    )
        
        task_coros = [
            execute_task(f"task_{i}", task)
            for i, task in enumerate(tasks)
        ]
        
        results = await asyncio.gather(*task_coros)
        return results
    
    @staticmethod
    async def wait_for_condition(
        condition: Callable[[], bool],
        timeout: float = 30.0,
        check_interval: float = 0.5,
        async_condition: Optional[Callable[[], Awaitable[bool]]] = None
    ) -> bool:
        """
        Wait for condition to become true.
        
        Args:
            condition: Sync condition function
            timeout: Maximum time to wait
            check_interval: Interval between checks
            async_condition: Optional async condition function
            
        Returns:
            True if condition met, False if timeout
        """
        start_time = asyncio.get_event_loop().time()
        
        while True:
            if async_condition:
                result = await async_condition()
            else:
                result = condition()
            
            if result:
                return True
            
            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed >= timeout:
                return False
            
            await asyncio.sleep(check_interval)
    
    @staticmethod
    async def race(
        *coros: Awaitable[T],
        return_first: bool = True
    ) -> T:
        """
        Race multiple coroutines (return first to complete).
        
        Args:
            *coros: Coroutines to race
            return_first: Whether to return first result (or wait for all)
            
        Returns:
            First result
        """
        if return_first:
            done, pending = await asyncio.wait(
                [asyncio.create_task(coro) for coro in coros],
                return_when=asyncio.FIRST_COMPLETED
            )
            
            # Cancel pending tasks
            for task in pending:
                task.cancel()
            
            result = await done.pop()
            return result.result()
        else:
            results = await asyncio.gather(*coros)
            return results[0]
    
    @staticmethod
    async def batch_execute(
        items: List[Any],
        processor: Callable[[Any], Awaitable[T]],
        batch_size: int = 10,
        max_concurrent: int = 5
    ) -> List[T]:
        """
        Execute processor on items in batches.
        
        Args:
            items: Items to process
            processor: Async processor function
            batch_size: Batch size
            max_concurrent: Max concurrent per batch
            
        Returns:
            List of results
        """
        results: List[T] = []
        
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            batch_results = await AsyncCoordinator.execute_parallel(
                [lambda item=item: processor(item) for item in batch],
                max_concurrent=max_concurrent
            )
            
            # Extract results
            for task_result in batch_results:
                if task_result.success:
                    results.append(task_result.result)
                else:
                    logger.error(f"Batch item failed: {task_result.error}")
        
        return results


# Convenience functions
async def execute_with_timeout(coro: Awaitable[T], timeout: float, **kwargs) -> Optional[T]:
    """Execute coroutine with timeout."""
    return await AsyncCoordinator.execute_with_timeout(coro, timeout, **kwargs)


async def gather_with_results(*coros: Awaitable[Any], **kwargs) -> List[Any]:
    """Gather coroutines with results."""
    return await AsyncCoordinator.gather_with_results(*coros, **kwargs)


async def execute_parallel(tasks: List[Callable[[], Awaitable[T]]], **kwargs) -> List[TaskResult]:
    """Execute tasks in parallel."""
    return await AsyncCoordinator.execute_parallel(tasks, **kwargs)


async def wait_for_condition(condition: Callable[[], bool], **kwargs) -> bool:
    """Wait for condition."""
    return await AsyncCoordinator.wait_for_condition(condition, **kwargs)




