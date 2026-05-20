"""
Batch Statistics
===============
Calculates batch statistics and estimates
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta

from .batch_tracker import BatchOperation, BatchStatus

logger = logging.getLogger(__name__)


class BatchStatistics:
    """
    Calculates batch statistics and estimates.
    """
    
    @staticmethod
    def calculate_batch_statistics(batch_op: BatchOperation) -> Dict[str, Any]:
        """
        Calculate statistics for a batch operation.
        
        Args:
            batch_op: Batch operation
            
        Returns:
            Statistics dictionary
        """
        total = batch_op.total_items
        completed = batch_op.completed_items
        failed = batch_op.failed_items
        pending = total - completed - failed
        
        duration = None
        if batch_op.started_at:
            end_time = batch_op.completed_at or datetime.now()
            duration = (end_time - batch_op.started_at).total_seconds()
        
        avg_time_per_item = None
        if completed > 0 and duration:
            avg_time_per_item = duration / completed
        
        return {
            "total_items": total,
            "completed_items": completed,
            "failed_items": failed,
            "pending_items": pending,
            "completion_rate": (completed / total * 100) if total > 0 else 0,
            "failure_rate": (failed / total * 100) if total > 0 else 0,
            "duration_seconds": duration,
            "avg_time_per_item_seconds": avg_time_per_item,
            "status": batch_op.status.value
        }
    
    @staticmethod
    def estimate_remaining_time(batch_op: BatchOperation) -> Optional[Dict[str, Any]]:
        """
        Estimate remaining time for batch completion.
        
        Args:
            batch_op: Batch operation
            
        Returns:
            Estimation dictionary or None if cannot estimate
        """
        if batch_op.status in (BatchStatus.COMPLETED, BatchStatus.FAILED, BatchStatus.CANCELLED):
            return {
                "remaining_items": 0,
                "estimated_seconds": 0,
                "estimated_completion": None
            }
        
        if not batch_op.started_at:
            return None
        
        completed = batch_op.completed_items
        total = batch_op.total_items
        remaining = total - completed - batch_op.failed_items
        
        if completed == 0 or remaining <= 0:
            return {
                "remaining_items": remaining,
                "estimated_seconds": None,
                "estimated_completion": None
            }
        
        # Calculate average time per item
        elapsed = (datetime.now() - batch_op.started_at).total_seconds()
        avg_time_per_item = elapsed / completed
        
        # Estimate remaining time
        estimated_seconds = avg_time_per_item * remaining
        estimated_completion = datetime.now() + timedelta(seconds=estimated_seconds)
        
        return {
            "remaining_items": remaining,
            "estimated_seconds": estimated_seconds,
            "estimated_completion": estimated_completion.isoformat(),
            "avg_time_per_item_seconds": avg_time_per_item,
            "elapsed_seconds": elapsed
        }
    
    @staticmethod
    def get_global_statistics(batches: List[BatchOperation]) -> Dict[str, Any]:
        """
        Get global statistics across all batches.
        
        Args:
            batches: List of batch operations
            
        Returns:
            Global statistics dictionary
        """
        total_batches = len(batches)
        total_items = sum(b.total_items for b in batches)
        total_completed = sum(b.completed_items for b in batches)
        total_failed = sum(b.failed_items for b in batches)
        
        status_counts = {}
        for status in BatchStatus:
            status_counts[status.value] = sum(
                1 for b in batches if b.status == status
            )
        
        return {
            "total_batches": total_batches,
            "total_items": total_items,
            "total_completed": total_completed,
            "total_failed": total_failed,
            "overall_completion_rate": (total_completed / total_items * 100) if total_items > 0 else 0,
            "overall_failure_rate": (total_failed / total_items * 100) if total_items > 0 else 0,
            "status_distribution": status_counts
        }

