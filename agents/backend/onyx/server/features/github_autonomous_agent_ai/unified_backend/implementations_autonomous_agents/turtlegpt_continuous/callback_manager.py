"""
Callback Manager

Manages callbacks for tasks and errors.
"""

from typing import Callable, List
import logging
import asyncio

from .models import AgentTask

logger = logging.getLogger(__name__)


class CallbackManager:
    """Manages callbacks for tasks and errors."""
    
    def __init__(self):
        """Initialize callback manager."""
        self.task_callbacks: List[Callable[[AgentTask], None]] = []
        self.error_callbacks: List[Callable[[Exception], None]] = []
    
    def add_task_callback(self, callback: Callable[[AgentTask], None]) -> None:
        """
        Add callback for task completion.
        
        Args:
            callback: Callback function
        """
        self.task_callbacks.append(callback)
    
    def add_error_callback(self, callback: Callable[[Exception], None]) -> None:
        """
        Add callback for errors.
        
        Args:
            callback: Callback function
        """
        self.error_callbacks.append(callback)
    
    async def execute_task_callbacks(self, task: AgentTask) -> None:
        """
        Execute task callbacks.
        
        Args:
            task: Completed task
        """
        for callback in self.task_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(task)
                else:
                    callback(task)
            except Exception as e:
                logger.error(f"Error in task callback: {e}", exc_info=True)
    
    async def execute_error_callbacks(self, error: Exception) -> None:
        """
        Execute error callbacks.
        
        Args:
            error: Exception that occurred
        """
        for callback in self.error_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(error)
                else:
                    callback(error)
            except Exception as e:
                logger.error(f"Error in error callback: {e}", exc_info=True)



