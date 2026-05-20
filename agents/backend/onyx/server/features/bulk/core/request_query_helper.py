"""
Request Query Helper - Common query utilities for requests
==========================================================

Provides common utilities for querying request status and documents.
"""

import logging
from typing import Dict, List, Optional, Any

from .task_attributes import TaskAttributes

logger = logging.getLogger(__name__)


class RequestQueryHelper:
    """Common utilities for querying request information."""
    
    @staticmethod
    def get_request_documents(
        request_id: str,
        completed_tasks: Dict[str, Any],
        include_metadata: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Get all generated documents for a request.
        
        Args:
            request_id: ID of the request
            completed_tasks: Dictionary of completed tasks
            include_metadata: Whether to include additional metadata
            
        Returns:
            List of document dictionaries with task information
        """
        documents = []
        
        for task in completed_tasks.values():
            if (task.request_id == request_id and 
                task.status == "completed" and 
                hasattr(task, 'content') and task.content):
                
                doc = {
                    "task_id": task.id,
                    "document_type": task.document_type,
                    "business_area": task.business_area,
                    "content": task.content,
                }
                
                if hasattr(task, 'created_at') and task.created_at:
                    doc["created_at"] = task.created_at.isoformat()
                
                if hasattr(task, 'completed_at') and task.completed_at:
                    doc["completed_at"] = task.completed_at.isoformat()
                
                if include_metadata:
                    if hasattr(task, 'quality_score'):
                        doc["quality_score"] = task.quality_score
                    if hasattr(task, 'processing_time'):
                        doc["processing_time"] = task.processing_time
                    if hasattr(task, 'model_used'):
                        doc["model_used"] = task.model_used
                    if hasattr(task, 'cache_hit'):
                        doc["cache_hit"] = task.cache_hit
                
                documents.append(doc)
        
        return documents
    
    @staticmethod
    def find_request(
        request_id: str,
        active_requests: Dict[str, Any]
    ) -> Optional[Any]:
        """
        Find a request by ID.
        
        Args:
            request_id: ID of the request to find
            active_requests: Dictionary of active requests
            
        Returns:
            Request object if found, None otherwise
        """
        return active_requests.get(request_id)
    
    @staticmethod
    def get_completed_tasks_for_request(
        request_id: str,
        completed_tasks: Dict[str, Any],
        status_filter: Optional[str] = "completed"
    ) -> List[Any]:
        """
        Get all completed tasks for a specific request.
        
        Args:
            request_id: ID of the request
            completed_tasks: Dictionary of completed tasks
            status_filter: Optional status filter (e.g., "completed", "failed")
            
        Returns:
            List of tasks matching the criteria
        """
        tasks = [
            task for task in completed_tasks.values()
            if task.request_id == request_id
        ]
        
        if status_filter:
            tasks = [task for task in tasks if task.status == status_filter]
        
        return tasks
    
    @staticmethod
    def calculate_request_metrics(
        request_id: str,
        completed_tasks: Dict[str, Any],
        active_tasks: Dict[str, Any],
        task_queue: List[Any]
    ) -> Dict[str, Any]:
        """
        Calculate metrics for a request.
        
        Args:
            request_id: ID of the request
            completed_tasks: Dictionary of completed tasks
            active_tasks: Dictionary of active tasks
            task_queue: List of queued tasks
            
        Returns:
            Dictionary with request metrics
        """
        completed = RequestQueryHelper.get_completed_tasks_for_request(
            request_id, completed_tasks, "completed"
        )
        failed = RequestQueryHelper.get_completed_tasks_for_request(
            request_id, completed_tasks, "failed"
        )
        active = [
            task for task in active_tasks.values()
            if task.request_id == request_id
        ]
        queued = [
            task for task in task_queue
            if task.request_id == request_id
        ]
        
        metrics = {
            "completed_count": len(completed),
            "failed_count": len(failed),
            "active_count": len(active),
            "queued_count": len(queued),
            "total_processed": len(completed) + len(failed)
        }
        
        if completed and hasattr(completed[0], 'quality_score'):
            quality_scores = [task.quality_score for task in completed if hasattr(task, 'quality_score')]
            if quality_scores:
                metrics["average_quality_score"] = sum(quality_scores) / len(quality_scores)
                metrics["min_quality_score"] = min(quality_scores)
                metrics["max_quality_score"] = max(quality_scores)
        
        if completed and hasattr(completed[0], 'cache_hit'):
            cache_hits = sum(1 for task in completed if hasattr(task, 'cache_hit') and task.cache_hit)
            metrics["cache_hit_count"] = cache_hits
            metrics["cache_hit_rate"] = cache_hits / len(completed) if completed else 0.0
        
        return metrics

