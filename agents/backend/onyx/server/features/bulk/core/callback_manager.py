"""
Callback Manager - Manages callbacks for processor events
=========================================================

Handles callback registration and safe execution for processor events.
"""

import asyncio
import logging
from typing import Callable, Any, Optional

logger = logging.getLogger(__name__)


class CallbackManager:
    """Manages callbacks for processor events."""
    
    def __init__(self):
        self.on_document_generated: Optional[Callable[[Any, Any], None]] = None
        self.on_task_completed: Optional[Callable[[Any], None]] = None
        self.on_error: Optional[Callable[[Exception], None]] = None
    
    async def execute_document_callback(self, task: Any, processed_doc: Any) -> None:
        """Execute document generated callback safely."""
        if self.on_document_generated:
            await self._safe_callback(self.on_document_generated, task, processed_doc)
    
    async def execute_error_callback(self, *args: Any) -> None:
        """Execute error callback safely."""
        if self.on_error:
            await self._safe_callback(self.on_error, *args)
    
    async def execute_task_callback(self, task: Any) -> None:
        """Execute task completed callback safely."""
        if self.on_task_completed:
            await self._safe_callback(self.on_task_completed, task)
    
    async def execute_callback(self, callback: Callable, *args: Any, **kwargs: Any) -> None:
        """Execute an arbitrary callback safely."""
        await self._safe_callback(callback, *args, **kwargs)
    
    async def _safe_callback(self, callback: Callable, *args: Any, **kwargs: Any) -> None:
        """Safely execute a callback function."""
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(*args, **kwargs)
            else:
                callback(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in callback: {e}", exc_info=True)
    
    def set_document_callback(self, callback: Callable[[Any, Any], None]) -> None:
        """Set callback for when documents are generated."""
        self.on_document_generated = callback
    
    def set_task_callback(self, callback: Callable[[Any], None]) -> None:
        """Set callback for when tasks are completed."""
        self.on_task_completed = callback
    
    def set_error_callback(self, callback: Callable[[Exception], None]) -> None:
        """Set callback for when errors occur."""
        self.on_error = callback
