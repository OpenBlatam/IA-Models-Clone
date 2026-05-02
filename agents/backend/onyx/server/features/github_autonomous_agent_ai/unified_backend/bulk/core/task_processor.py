"""
Task Processor - Common task processing patterns
=================================================

Provides common utilities for processing document generation tasks.
"""

import logging
from typing import Dict, Any, Optional, Callable
from datetime import datetime

from .task_error_handler import TaskErrorHandler

logger = logging.getLogger(__name__)


class TaskProcessor:
    """Common utilities for processing tasks."""
    
    @staticmethod
    async def process_task_with_error_handling(
        task: Any,
        active_tasks: Dict[str, Any],
        task_queue: list,
        process_func: Callable,
        on_error_callback: Optional[Callable] = None,
        max_retries: int = 3,
        retry_delay_base: float = 1.0,
        completed_tasks: Optional[Dict[str, Any]] = None,
        processing_stats: Optional[Dict[str, Any]] = None,
        error_analysis: Optional[Dict[str, int]] = None
    ) -> None:
        """
        Process a task with standard error handling and retry logic.
        
        Args:
            task: The task to process
            active_tasks: Dictionary of active tasks
            task_queue: Queue to add task back to if retrying
            process_func: Async function to process the task
            on_error_callback: Optional callback for errors
            max_retries: Maximum number of retries
            retry_delay_base: Base delay for exponential backoff
            completed_tasks: Optional dictionary to store completed/failed tasks
            processing_stats: Optional statistics dictionary
            error_analysis: Optional error analysis dictionary
        """
        try:
            task.status = "processing"
            active_tasks[task.id] = task
            
            await process_func(task)
            
        except Exception as e:
            logger.error(f"Task failed: {task.id} - {e}")
            
            async def error_callback(t, err):
                if on_error_callback:
                    if hasattr(on_error_callback, '__call__'):
                        try:
                            if hasattr(on_error_callback, '__code__') and on_error_callback.__code__.co_argcount == 2:
                                await on_error_callback(t, err)
                            else:
                                await on_error_callback(t)
                        except Exception as callback_error:
                            logger.error(f"Error in error callback: {callback_error}")
            
            should_retry = await TaskErrorHandler.handle_task_error(
                task=task,
                error=e,
                task_queue=task_queue,
                max_retries=max_retries,
                retry_delay_base=retry_delay_base,
                on_error_callback=error_callback
            )
            
            if not should_retry:
                TaskErrorHandler.mark_task_failed(
                    task=task,
                    completed_tasks=completed_tasks or {},
                    processing_stats=processing_stats,
                    error_analysis=error_analysis,
                    error=e
                )
        
        finally:
            if task.id in active_tasks:
                del active_tasks[task.id]
    
    @staticmethod
    def mark_task_processing(task: Any, active_tasks: Dict[str, Any]) -> None:
        """Mark a task as processing."""
        task.status = "processing"
        active_tasks[task.id] = task
    
    @staticmethod
    def cleanup_task(task: Any, active_tasks: Dict[str, Any]) -> None:
        """Remove task from active tasks."""
        if task.id in active_tasks:
            del active_tasks[task.id]






