"""
Batch Tracker
=============
Tracks batch operations and their status
"""

import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class BatchStatus(str, Enum):
    """Batch operation status"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class BatchItem:
    """Single item in a batch operation"""
    item_id: str
    image_url: str
    clothing_description: Optional[str] = None
    face_url: Optional[str] = None
    mask_url: Optional[str] = None
    character_name: Optional[str] = None
    status: BatchStatus = BatchStatus.PENDING
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    prompt_id: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


@dataclass
class BatchOperation:
    """Batch operation container"""
    batch_id: str
    operation_type: str
    items: List[BatchItem]
    status: BatchStatus = BatchStatus.PENDING
    total_items: int = 0
    completed_items: int = 0
    failed_items: int = 0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class BatchTracker:
    """
    Tracks batch operations and their status.
    """
    
    def __init__(self):
        """Initialize batch tracker"""
        self.active_batches: Dict[str, BatchOperation] = {}
    
    def create_batch_operation(
        self,
        batch_id: str,
        operation_type: str,
        items: List[Dict[str, Any]]
    ) -> BatchOperation:
        """
        Create a new batch operation.
        
        Args:
            batch_id: Unique batch identifier
            operation_type: Type of operation
            items: List of item dictionaries
            
        Returns:
            Created batch operation
        """
        batch_items = []
        for i, item in enumerate(items):
            batch_item = BatchItem(
                item_id=item.get("item_id") or f"item_{i}",
                image_url=item["image_url"],
                clothing_description=item.get("clothing_description"),
                face_url=item.get("face_url"),
                mask_url=item.get("mask_url"),
                character_name=item.get("character_name")
            )
            batch_items.append(batch_item)
        
        batch_op = BatchOperation(
            batch_id=batch_id,
            operation_type=operation_type,
            items=batch_items,
            total_items=len(batch_items),
            started_at=datetime.now()
        )
        
        self.active_batches[batch_id] = batch_op
        return batch_op
    
    def get_batch(self, batch_id: str) -> Optional[BatchOperation]:
        """
        Get batch operation by ID.
        
        Args:
            batch_id: Batch identifier
            
        Returns:
            Batch operation or None if not found
        """
        return self.active_batches.get(batch_id)
    
    def update_batch_status(
        self,
        batch_id: str,
        status: BatchStatus,
        completed_items: Optional[int] = None,
        failed_items: Optional[int] = None
    ) -> bool:
        """
        Update batch operation status.
        
        Args:
            batch_id: Batch identifier
            status: New status
            completed_items: Number of completed items
            failed_items: Number of failed items
            
        Returns:
            True if update successful, False if batch not found
        """
        batch_op = self.active_batches.get(batch_id)
        if not batch_op:
            return False
        
        batch_op.status = status
        
        if completed_items is not None:
            batch_op.completed_items = completed_items
        if failed_items is not None:
            batch_op.failed_items = failed_items
        
        if status in (BatchStatus.COMPLETED, BatchStatus.FAILED, BatchStatus.CANCELLED):
            batch_op.completed_at = datetime.now()
        
        return True
    
    def update_item_status(
        self,
        batch_id: str,
        item_id: str,
        status: BatchStatus,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        prompt_id: Optional[str] = None
    ) -> bool:
        """
        Update individual item status.
        
        Args:
            batch_id: Batch identifier
            item_id: Item identifier
            status: New status
            result: Processing result
            error: Error message
            prompt_id: Prompt ID
            
        Returns:
            True if update successful, False otherwise
        """
        batch_op = self.active_batches.get(batch_id)
        if not batch_op:
            return False
        
        for item in batch_op.items:
            if item.item_id == item_id:
                item.status = status
                item.result = result
                item.error = error
                item.prompt_id = prompt_id
                
                if status == BatchStatus.PROCESSING and not item.started_at:
                    item.started_at = datetime.now()
                elif status in (BatchStatus.COMPLETED, BatchStatus.FAILED):
                    item.completed_at = datetime.now()
                
                return True
        
        return False
    
    def list_batches(
        self,
        status_filter: Optional[BatchStatus] = None,
        limit: Optional[int] = None
    ) -> List[BatchOperation]:
        """
        List batch operations.
        
        Args:
            status_filter: Optional status filter
            limit: Optional limit on number of results
            
        Returns:
            List of batch operations
        """
        batches = list(self.active_batches.values())
        
        if status_filter:
            batches = [b for b in batches if b.status == status_filter]
        
        # Sort by started_at (most recent first)
        batches.sort(key=lambda b: b.started_at or datetime.min, reverse=True)
        
        if limit:
            batches = batches[:limit]
        
        return batches
    
    def cancel_batch(self, batch_id: str) -> bool:
        """
        Cancel a batch operation.
        
        Args:
            batch_id: Batch identifier
            
        Returns:
            True if cancellation successful, False otherwise
        """
        return self.update_batch_status(batch_id, BatchStatus.CANCELLED)
    
    def cleanup_completed_batches(self, max_age_hours: int = 24) -> int:
        """
        Cleanup completed batches older than specified age.
        
        Args:
            max_age_hours: Maximum age in hours for completed batches
            
        Returns:
            Number of batches cleaned up
        """
        from datetime import timedelta
        
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        batches_to_remove = []
        
        for batch_id, batch_op in self.active_batches.items():
            if batch_op.status in (BatchStatus.COMPLETED, BatchStatus.FAILED, BatchStatus.CANCELLED):
                if batch_op.completed_at and batch_op.completed_at < cutoff_time:
                    batches_to_remove.append(batch_id)
        
        for batch_id in batches_to_remove:
            del self.active_batches[batch_id]
        
        return len(batches_to_remove)

