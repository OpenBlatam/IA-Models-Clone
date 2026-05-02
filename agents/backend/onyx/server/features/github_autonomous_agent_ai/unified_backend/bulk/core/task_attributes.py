"""
Task Attributes - Safe task attribute access utilities
======================================================

Provides safe utilities for accessing task attributes with defaults.
"""

import logging
from typing import Any, Optional, List
from datetime import datetime

logger = logging.getLogger(__name__)


class TaskAttributes:
    """Safe utilities for accessing task attributes."""
    
    @staticmethod
    def get_quality_score(task: Any, default: float = 0.0) -> float:
        """
        Safely get quality score from a task.
        
        Args:
            task: Task object
            default: Default value if not available
            
        Returns:
            Quality score or default
        """
        return getattr(task, 'quality_score', default)
    
    @staticmethod
    def get_processing_time(task: Any, default: float = 0.0) -> float:
        """
        Safely get processing time from a task.
        
        Args:
            task: Task object
            default: Default value if not available
            
        Returns:
            Processing time or default
        """
        return getattr(task, 'processing_time', default)
    
    @staticmethod
    def get_cache_hit(task: Any, default: bool = False) -> bool:
        """
        Safely get cache hit status from a task.
        
        Args:
            task: Task object
            default: Default value if not available
            
        Returns:
            Cache hit status or default
        """
        return getattr(task, 'cache_hit', default)
    
    @staticmethod
    def get_model_used(task: Any, default: Optional[str] = None) -> Optional[str]:
        """
        Safely get model used from a task.
        
        Args:
            task: Task object
            default: Default value if not available
            
        Returns:
            Model name or default
        """
        return getattr(task, 'model_used', default)
    
    @staticmethod
    def get_content(task: Any, default: Optional[str] = None) -> Optional[str]:
        """
        Safely get content from a task.
        
        Args:
            task: Task object
            default: Default value if not available
            
        Returns:
            Content or default
        """
        return getattr(task, 'content', default)
    
    @staticmethod
    def get_created_at(task: Any) -> Optional[datetime]:
        """
        Safely get created_at timestamp from a task.
        
        Args:
            task: Task object
            
        Returns:
            Created timestamp or None
        """
        return getattr(task, 'created_at', None)
    
    @staticmethod
    def get_completed_at(task: Any) -> Optional[datetime]:
        """
        Safely get completed_at timestamp from a task.
        
        Args:
            task: Task object
            
        Returns:
            Completed timestamp or None
        """
        return getattr(task, 'completed_at', None)
    
    @staticmethod
    def has_quality_score(task: Any) -> bool:
        """Check if task has quality_score attribute with valid value."""
        return hasattr(task, 'quality_score') and task.quality_score > 0
    
    @staticmethod
    def has_processing_time(task: Any) -> bool:
        """Check if task has processing_time attribute with valid value."""
        return hasattr(task, 'processing_time') and task.processing_time > 0
    
    @staticmethod
    def has_content(task: Any) -> bool:
        """Check if task has content attribute with non-empty value."""
        return hasattr(task, 'content') and bool(task.content)
    
    @staticmethod
    def filter_tasks_with_quality_scores(tasks: List[Any]) -> List[Any]:
        """Filter tasks that have valid quality scores."""
        return [task for task in tasks if TaskAttributes.has_quality_score(task)]
    
    @staticmethod
    def filter_tasks_with_processing_times(tasks: List[Any]) -> List[Any]:
        """Filter tasks that have valid processing times."""
        return [task for task in tasks if TaskAttributes.has_processing_time(task)]






