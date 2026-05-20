"""
Task Error Handler - Common error handling and retry logic
==========================================================

Handles common error handling and retry logic for document generation tasks.
"""

import asyncio
import logging
from typing import Any, Optional, Callable
from datetime import datetime

from .constants import DEFAULT_MAX_RETRIES, DEFAULT_RETRY_DELAY_BASE, MAX_RETRY_DELAY

logger = logging.getLogger(__name__)


class TaskErrorHandler:
    """Handles error processing and retry logic for tasks."""
    
    @staticmethod
    async def handle_task_error(
        task: Any,
        error: Exception,
        task_queue: list,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_delay_base: float = DEFAULT_RETRY_DELAY_BASE,
        on_error_callback: Optional[Callable] = None
    ) -> bool:
        """
        Handle task error and determine if should retry.
        
        Args:
            task: The task that failed
            error: The exception that occurred
            task_queue: Queue to add task back to if retrying
            max_retries: Maximum number of retries
            retry_delay_base: Base delay for exponential backoff
            on_error_callback: Optional callback to execute on error
            
        Returns:
            True if task should be retried, False otherwise
        """
        task.status = "failed"
        task.error = str(error)
        
        if not hasattr(task, 'retry_count'):
            task.retry_count = 0
        task.retry_count += 1
        
        should_retry = task.retry_count < max_retries
        
        if should_retry:
            delay = min(MAX_RETRY_DELAY, retry_delay_base * (2 ** (task.retry_count - 1)))
            await asyncio.sleep(delay)
            task.status = "pending"
            task_queue.append(task)
            logger.info(f"Retrying task: {task.id} (attempt {task.retry_count + 1}/{max_retries})")
        else:
            logger.error(f"Task failed permanently: {task.id} after {task.retry_count} attempts")
        
        if on_error_callback:
            try:
                if asyncio.iscoroutinefunction(on_error_callback):
                    await on_error_callback(task, error)
                else:
                    on_error_callback(task, error)
            except Exception as callback_error:
                logger.error(f"Error in error callback: {callback_error}")
        
        return should_retry
    
    @staticmethod
    def mark_task_completed(
        task: Any,
        completed_tasks: dict,
        processing_stats: Optional[dict] = None
    ) -> None:
        """
        Mark a task as completed and update statistics.
        
        Args:
            task: The completed task
            completed_tasks: Dictionary to store completed tasks
            processing_stats: Optional statistics dictionary to update
        """
        task.status = "completed"
        if not hasattr(task, 'completed_at') or task.completed_at is None:
            task.completed_at = datetime.now()
        
        completed_tasks[task.id] = task
        
        if processing_stats:
            processing_stats["total_documents_generated"] = processing_stats.get(
                "total_documents_generated", 0
            ) + 1
    
    @staticmethod
    def mark_task_failed(
        task: Any,
        completed_tasks: dict,
        processing_stats: Optional[dict] = None,
        error_analysis: Optional[dict] = None,
        error: Optional[Exception] = None
    ) -> None:
        """
        Mark a task as permanently failed and update statistics.
        
        Args:
            task: The failed task
            completed_tasks: Dictionary to store failed tasks
            processing_stats: Optional statistics dictionary to update
            error_analysis: Optional error analysis dictionary to update
            error: Optional exception that caused the failure
        """
        task.status = "failed"
        
        # Store error information
        if error:
            task.error = str(error)
        elif not hasattr(task, 'error') or not task.error:
            task.error = "Unknown error"
        
        completed_tasks[task.id] = task
        
        if processing_stats:
            processing_stats["total_documents_failed"] = processing_stats.get(
                "total_documents_failed", 0
            ) + 1
        
        if error_analysis is not None:
            error_type = "Unknown"
            if error:
                error_type = type(error).__name__
            elif hasattr(task, 'error') and task.error:
                if isinstance(task.error, Exception):
                    error_type = type(task.error).__name__
                elif isinstance(task.error, str):
                    error_type = task.error.split(':')[0] if ':' in task.error else "Error"
            error_analysis[error_type] = error_analysis.get(error_type, 0) + 1

