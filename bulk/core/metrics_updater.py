"""
Metrics Updater - Common metrics update utilities
=================================================

Provides common utilities for updating processing metrics and statistics.
"""

import logging
import statistics
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


class MetricsUpdater:
    """Common utilities for updating metrics."""
    
    @staticmethod
    def calculate_average_quality_score(completed_tasks: Dict[str, Any]) -> float:
        """
        Calculate average quality score from completed tasks.
        
        Args:
            completed_tasks: Dictionary of completed tasks
            
        Returns:
            Average quality score, or 0.0 if no valid scores found
        """
        quality_scores = [
            task.quality_score for task in completed_tasks.values()
            if hasattr(task, 'quality_score') and task.quality_score > 0
        ]
        
        if quality_scores:
            return statistics.mean(quality_scores)
        return 0.0
    
    @staticmethod
    def calculate_cache_hit_rate(completed_tasks: Dict[str, Any]) -> float:
        """
        Calculate cache hit rate from completed tasks.
        
        Args:
            completed_tasks: Dictionary of completed tasks
            
        Returns:
            Cache hit rate (0.0 to 1.0), or 0.0 if no tasks
        """
        if not completed_tasks:
            return 0.0
        
        cache_hits = sum(
            1 for task in completed_tasks.values()
            if hasattr(task, 'cache_hit') and task.cache_hit
        )
        
        return cache_hits / len(completed_tasks) if completed_tasks else 0.0
    
    @staticmethod
    def calculate_average_processing_time(completed_tasks: Dict[str, Any]) -> float:
        """
        Calculate average processing time from completed tasks.
        
        Args:
            completed_tasks: Dictionary of completed tasks
            
        Returns:
            Average processing time in seconds, or 0.0 if no valid times found
        """
        processing_times = [
            task.processing_time for task in completed_tasks.values()
            if hasattr(task, 'processing_time') and task.processing_time > 0
        ]
        
        if processing_times:
            return statistics.mean(processing_times)
        return 0.0
    
    @staticmethod
    def update_enhanced_metrics(
        metrics: Any,
        completed_tasks: Dict[str, Any]
    ) -> None:
        """
        Update enhanced processing metrics from completed tasks.
        
        Args:
            metrics: EnhancedProcessingMetrics instance to update
            completed_tasks: Dictionary of completed tasks
        """
        if not completed_tasks:
            return
        
        metrics.average_quality_score = MetricsUpdater.calculate_average_quality_score(
            completed_tasks
        )
        
        metrics.cache_hit_rate = MetricsUpdater.calculate_cache_hit_rate(
            completed_tasks
        )
        
        metrics.average_processing_time = MetricsUpdater.calculate_average_processing_time(
            completed_tasks
        )
    
    @staticmethod
    def update_processing_stats(
        processing_stats: Dict[str, Any],
        completed_tasks: Dict[str, Any],
        active_requests: Dict[str, Any],
        task_queue: List[Any]
    ) -> None:
        """
        Update standard processing statistics.
        
        Args:
            processing_stats: Dictionary to update
            completed_tasks: Dictionary of completed tasks
            active_requests: Dictionary of active requests
            task_queue: List of queued tasks
        """
        processing_stats["active_requests"] = len(active_requests)
        processing_stats["queued_tasks"] = len(task_queue)
        
        if completed_tasks:
            completed_count = sum(
                1 for task in completed_tasks.values()
                if hasattr(task, 'status') and task.status == "completed"
            )
            failed_count = sum(
                1 for task in completed_tasks.values()
                if hasattr(task, 'status') and task.status == "failed"
            )
            
            processing_stats["total_documents_generated"] = completed_count
            processing_stats["total_documents_failed"] = failed_count
            
            processing_times = [
                task.processing_time for task in completed_tasks.values()
                if hasattr(task, 'processing_time') and task.processing_time > 0
            ]
            if processing_times:
                processing_stats["average_processing_time"] = statistics.mean(processing_times)

