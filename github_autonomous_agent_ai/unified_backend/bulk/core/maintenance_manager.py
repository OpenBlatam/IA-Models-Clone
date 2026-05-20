"""
Maintenance Manager - Handles maintenance tasks during idle time
================================================================

Manages maintenance operations like cleanup and cache management.
"""

import logging
from typing import Any, Optional
from datetime import timedelta

from .processor_config import ProcessorConfig

logger = logging.getLogger(__name__)


class MaintenanceManager:
    """Manages maintenance tasks during idle time."""
    
    def __init__(self, config: ProcessorConfig, processor: Any):
        self.config = config
        self.processor = processor
    
    async def perform_maintenance(self) -> None:
        """Perform maintenance tasks during idle time."""
        try:
            await self._cleanup_old_tasks()
            await self._cleanup_cache()
            logger.debug("Maintenance tasks completed")
        except Exception as e:
            logger.error(f"Error during maintenance: {e}", exc_info=True)
    
    async def _cleanup_old_tasks(self) -> None:
        """Clean up old completed tasks to prevent memory buildup."""
        if not hasattr(self.processor, 'completed_tasks'):
            return
        
        completed_tasks = self.processor.completed_tasks
        max_tasks = self.config.max_completed_tasks_to_keep
        
        if len(completed_tasks) > max_tasks:
            sorted_tasks = sorted(
                completed_tasks.items(),
                key=lambda x: self._get_task_timestamp(x[1]),
                reverse=True
            )
            
            tasks_to_remove = sorted_tasks[max_tasks:]
            for task_id, _ in tasks_to_remove:
                del completed_tasks[task_id]
            
            logger.info(f"Cleaned up {len(tasks_to_remove)} old tasks")
    
    def _get_task_timestamp(self, task: Any) -> Any:
        """Get timestamp from task for sorting."""
        if hasattr(task, 'completed_at') and task.completed_at:
            return task.completed_at
        if hasattr(task, 'created_at'):
            return task.created_at
        return None
    
    async def _cleanup_cache(self) -> None:
        """Clean up processor cache if available."""
        if hasattr(self.processor, 'document_processor'):
            if hasattr(self.processor.document_processor, 'cleanup_cache'):
                await self.processor.document_processor.cleanup_cache()
