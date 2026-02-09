"""
Batch Result Builder
====================
Builds response dictionaries for batch operations
"""

import logging
from typing import Dict, Any, List
from datetime import datetime

from .batch_tracker import BatchOperation, BatchStatus

logger = logging.getLogger(__name__)


class BatchResultBuilder:
    """
    Builds response dictionaries for batch operations.
    """
    
    @staticmethod
    def build_batch_response(
        batch_op: BatchOperation,
        results: List[Dict[str, Any]],
        duration: float
    ) -> Dict[str, Any]:
        """
        Build response dictionary for batch operation.
        
        Args:
            batch_op: Batch operation
            results: List of item results
            duration: Total duration in seconds
            
        Returns:
            Batch response dictionary
        """
        return {
            "batch_id": batch_op.batch_id,
            "operation_type": batch_op.operation_type,
            "total_items": batch_op.total_items,
            "completed": batch_op.completed_items,
            "failed": batch_op.failed_items,
            "results": results,
            "duration": duration,
            "status": batch_op.status.value,
            "started_at": batch_op.started_at.isoformat() if batch_op.started_at else None,
            "completed_at": batch_op.completed_at.isoformat() if batch_op.completed_at else None
        }
    
    @staticmethod
    def build_batch_details(batch_op: BatchOperation) -> Dict[str, Any]:
        """
        Build detailed batch information dictionary.
        
        Args:
            batch_op: Batch operation
            
        Returns:
            Detailed batch information
        """
        items_details = []
        for item in batch_op.items:
            item_detail = {
                "item_id": item.item_id,
                "status": item.status.value,
                "image_url": item.image_url,
                "started_at": item.started_at.isoformat() if item.started_at else None,
                "completed_at": item.completed_at.isoformat() if item.completed_at else None
            }
            
            if item.result:
                item_detail["result"] = item.result
            if item.error:
                item_detail["error"] = item.error
            if item.prompt_id:
                item_detail["prompt_id"] = item.prompt_id
            
            items_details.append(item_detail)
        
        duration = None
        if batch_op.started_at:
            end_time = batch_op.completed_at or datetime.now()
            duration = (end_time - batch_op.started_at).total_seconds()
        
        return {
            "batch_id": batch_op.batch_id,
            "operation_type": batch_op.operation_type,
            "status": batch_op.status.value,
            "total_items": batch_op.total_items,
            "completed_items": batch_op.completed_items,
            "failed_items": batch_op.failed_items,
            "pending_items": batch_op.total_items - batch_op.completed_items - batch_op.failed_items,
            "duration_seconds": duration,
            "started_at": batch_op.started_at.isoformat() if batch_op.started_at else None,
            "completed_at": batch_op.completed_at.isoformat() if batch_op.completed_at else None,
            "items": items_details,
            "metadata": batch_op.metadata
        }
    
    @staticmethod
    def build_batch_status(batch_op: BatchOperation) -> Dict[str, Any]:
        """
        Build status dictionary for batch operation.
        
        Args:
            batch_op: Batch operation
            
        Returns:
            Status dictionary
        """
        return {
            "batch_id": batch_op.batch_id,
            "status": batch_op.status.value,
            "total_items": batch_op.total_items,
            "completed_items": batch_op.completed_items,
            "failed_items": batch_op.failed_items,
            "pending_items": batch_op.total_items - batch_op.completed_items - batch_op.failed_items,
            "progress_percentage": (
                (batch_op.completed_items + batch_op.failed_items) / batch_op.total_items * 100
                if batch_op.total_items > 0 else 0
            )
        }

