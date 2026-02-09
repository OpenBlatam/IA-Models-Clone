"""
Background task helper functions.

This module provides utilities for running background tasks safely,
with error handling and logging.
"""

import asyncio
from typing import Callable, Any, Optional, Dict
import logging

logger = logging.getLogger(__name__)


async def run_background_task(
    coro: Callable,
    *args,
    task_name: Optional[str] = None,
    log_errors: bool = True,
    **kwargs
) -> None:
    """
    Run a coroutine as a background task with error handling.
    
    Creates a task that runs independently and logs errors without
    affecting the main request flow.
    
    Args:
        coro: Coroutine function to run
        *args: Positional arguments for coroutine
        task_name: Optional name for logging
        log_errors: Whether to log errors
        **kwargs: Keyword arguments for coroutine
    
    Example:
        await run_background_task(
            webhook_service.trigger_webhook,
            event_type,
            data,
            task_name="webhook_trigger"
        )
    """
    name = task_name or coro.__name__
    
    async def wrapped_task():
        try:
            await coro(*args, **kwargs)
            logger.debug(f"Background task {name} completed successfully")
        except Exception as e:
            if log_errors:
                logger.error(
                    f"Background task {name} failed: {e}",
                    exc_info=True
                )
            # Don't re-raise - background tasks shouldn't affect main flow
    
    asyncio.create_task(wrapped_task())


async def run_background_task_safe(
    coro: Callable,
    *args,
    task_name: Optional[str] = None,
    on_error: Optional[Callable] = None,
    **kwargs
) -> None:
    """
    Run a background task with custom error handling.
    
    Args:
        coro: Coroutine function to run
        *args: Positional arguments for coroutine
        task_name: Optional name for logging
        on_error: Optional callback for error handling
        **kwargs: Keyword arguments for coroutine
    
    Example:
        async def handle_error(error):
            await error_notification_service.notify(error)
        
        await run_background_task_safe(
            webhook_service.trigger_webhook,
            event_type,
            data,
            on_error=handle_error
        )
    """
    name = task_name or coro.__name__
    
    async def wrapped_task():
        try:
            await coro(*args, **kwargs)
        except Exception as e:
            logger.error(f"Background task {name} failed: {e}", exc_info=True)
            if on_error:
                try:
                    await on_error(e)
                except Exception as callback_error:
                    logger.error(
                        f"Error handler for {name} failed: {callback_error}",
                        exc_info=True
                    )
    
    asyncio.create_task(wrapped_task())


def create_background_task(
    coro: Callable,
    *args,
    task_name: Optional[str] = None,
    **kwargs
) -> asyncio.Task:
    """
    Create a background task without awaiting it.
    
    Returns the task object for potential cancellation or tracking.
    
    Args:
        coro: Coroutine function to run
        *args: Positional arguments for coroutine
        task_name: Optional name for logging
        **kwargs: Keyword arguments for coroutine
    
    Returns:
        asyncio.Task object
    
    Example:
        task = create_background_task(
            webhook_service.trigger_webhook,
            event_type,
            data
        )
        # Can cancel later if needed: task.cancel()
    """
    name = task_name or coro.__name__
    
    async def wrapped_task():
        try:
            await coro(*args, **kwargs)
        except Exception as e:
            logger.error(f"Background task {name} failed: {e}", exc_info=True)
    
    return asyncio.create_task(wrapped_task())


async def run_multiple_background_tasks(
    tasks: list[tuple[Callable, tuple, dict]],
    wait_for_completion: bool = False,
    log_errors: bool = True
) -> list[asyncio.Task]:
    """
    Run multiple background tasks concurrently.
    
    Args:
        tasks: List of tuples (coro, args, kwargs)
        wait_for_completion: Whether to wait for all tasks to complete
        log_errors: Whether to log errors
    
    Returns:
        List of task objects
    
    Example:
        tasks = [
            (webhook_service.trigger_webhook, (event, data), {}),
            (analytics_service.track, (event,), {"user_id": user_id}),
        ]
        await run_multiple_background_tasks(tasks, wait_for_completion=False)
    """
    created_tasks = []
    
    for coro, args, kwargs in tasks:
        task = create_background_task(
            coro,
            *args,
            **kwargs
        )
        created_tasks.append(task)
    
    if wait_for_completion:
        try:
            await asyncio.gather(*created_tasks, return_exceptions=True)
        except Exception as e:
            if log_errors:
                logger.error(f"Error in background tasks: {e}", exc_info=True)
    
    return created_tasks








