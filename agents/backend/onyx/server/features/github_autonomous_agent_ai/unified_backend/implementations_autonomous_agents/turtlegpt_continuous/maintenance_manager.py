"""
Maintenance Manager

Handles periodic maintenance operations.
"""

from typing import Dict, Any, Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class MaintenanceManager:
    """Manages periodic maintenance operations."""
    
    def __init__(
        self,
        task_manager: Any,
        reflection_planner: Any,
        maintenance_interval_seconds: float = 300.0
    ):
        """
        Initialize maintenance manager.
        
        Args:
            task_manager: Task manager instance
            reflection_planner: Reflection planner instance
            maintenance_interval_seconds: Maintenance interval in seconds
        """
        self.task_manager = task_manager
        self.reflection_planner = reflection_planner
        self.maintenance_interval_seconds = maintenance_interval_seconds
        self.last_maintenance: Optional[datetime] = None
    
    def should_perform_maintenance(self) -> bool:
        """
        Check if maintenance should be performed.
        
        Returns:
            True if maintenance should be performed
        """
        if self.last_maintenance is None:
            return True
        
        elapsed = (datetime.now() - self.last_maintenance).total_seconds()
        return elapsed >= self.maintenance_interval_seconds
    
    async def perform_maintenance(self) -> None:
        """Perform periodic maintenance."""
        logger.debug("Performing maintenance...")
        
        # Cleanup old tasks
        self.task_manager.cleanup_completed_tasks(max_keep=100)
        
        # Cleanup old insights
        self.reflection_planner.cleanup_insights(max_keep=20)
        
        # Log status
        task_stats = self.task_manager.get_stats()
        reflection_status = self.reflection_planner.get_status()
        
        logger.info(
            f"Agent status: {task_stats['queue_size']} queued, "
            f"{task_stats['active_tasks']} active, "
            f"{reflection_status['insights_count']} insights, "
            f"{reflection_status['current_plan_size']} planned actions"
        )
        
        self.last_maintenance = datetime.now()



