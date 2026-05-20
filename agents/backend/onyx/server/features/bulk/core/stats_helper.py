"""
Stats Helper - Common statistics calculation utilities
=======================================================

Provides common utilities for calculating and formatting statistics.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime


class StatsHelper:
    """Helper class for calculating statistics."""
    
    @staticmethod
    def calculate_progress(completed: int, total: int) -> float:
        """
        Calculate progress percentage.
        
        Args:
            completed: Number of completed items
            total: Total number of items
            
        Returns:
            Progress percentage (0-100)
        """
        if total == 0:
            return 0.0
        return min(100.0, (completed / total) * 100.0)
    
    @staticmethod
    def get_base_stats(
        active_requests: dict,
        active_tasks: dict,
        task_queue: list,
        completed_tasks: dict,
        is_running: bool
    ) -> Dict[str, Any]:
        """
        Get base statistics common to all processors.
        
        Args:
            active_requests: Dictionary of active requests
            active_tasks: Dictionary of active tasks
            task_queue: List of queued tasks
            completed_tasks: Dictionary of completed tasks
            is_running: Whether processor is running
            
        Returns:
            Dictionary with base statistics
        """
        return {
            "active_requests": len(active_requests),
            "active_tasks": len(active_tasks),
            "queued_tasks": len(task_queue),
            "completed_tasks": len(completed_tasks),
            "is_running": is_running
        }
    
    @staticmethod
    def get_request_status_base(
        request_id: str,
        request: Any,
        completed_tasks: dict,
        active_tasks: dict,
        task_queue: list,
        failed_tasks: Optional[List[Any]] = None
    ) -> Dict[str, Any]:
        """
        Get base request status information.
        
        Args:
            request_id: ID of the request
            request: Request object
            completed_tasks: Dictionary of completed tasks
            active_tasks: Dictionary of active tasks
            task_queue: List of queued tasks
            failed_tasks: Optional list of failed tasks
            
        Returns:
            Dictionary with base request status
        """
        if failed_tasks is None:
            failed_tasks = [
                task for task in completed_tasks.values()
                if task.request_id == request_id and task.status == "failed"
            ]
        
        completed = [
            task for task in completed_tasks.values()
            if task.request_id == request_id and task.status == "completed"
        ]
        
        active = [
            task for task in active_tasks.values()
            if task.request_id == request_id
        ]
        
        queued = [
            task for task in task_queue
            if task.request_id == request_id
        ]
        
        return {
            "request_id": request_id,
            "status": "active" if request_id else "completed",
            "query": request.query,
            "max_documents": request.max_documents,
            "documents_generated": len(completed),
            "documents_failed": len(failed_tasks),
            "active_tasks": len(active),
            "queued_tasks": len(queued),
            "progress_percentage": StatsHelper.calculate_progress(
                len(completed),
                request.max_documents
            ),
            "created_at": request.created_at.isoformat() if hasattr(request, 'created_at') and request.created_at else None
        }






