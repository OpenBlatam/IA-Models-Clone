"""
Batch Operations
================

Advanced batch operation system with progress tracking and error handling.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Callable, Awaitable, TypeVar
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)

T = TypeVar('T')
R = TypeVar('R')


class BatchStatus(Enum):
    """Batch status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class BatchItem:
    """Batch item."""
    id: str
    data: T
    status: BatchStatus = BatchStatus.PENDING
    result: Optional[R] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchOperation:
    """Batch operation definition."""
    id: str
    items: List[BatchItem]
    processor: Callable[[T], Awaitable[R]]
    max_concurrent: int = 5
    stop_on_error: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class BatchResult:
    """Batch operation result."""
    operation_id: str
    status: BatchStatus
    total_items: int
    successful_items: int
    failed_items: int
    results: List[BatchItem] = field(default_factory=list)
    duration: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


class BatchOperationManager:
    """Manager for batch operations."""
    
    def __init__(self):
        """Initialize batch operation manager."""
        self.operations: Dict[str, BatchOperation] = {}
        self.results: Dict[str, BatchResult] = {}
        self.max_results = 1000
    
    async def execute(
        self,
        operation_id: str,
        items: List[T],
        processor: Callable[[T], Awaitable[R]],
        max_concurrent: int = 5,
        stop_on_error: bool = False,
        metadata: Optional[Dict[str, Any]] = None
    ) -> BatchResult:
        """
        Execute batch operation.
        
        Args:
            operation_id: Operation ID
            items: List of items to process
            processor: Processing function
            max_concurrent: Maximum concurrent operations
            stop_on_error: Whether to stop on first error
            metadata: Optional metadata
            
        Returns:
            Batch result
        """
        start = datetime.now()
        
        # Create batch items
        batch_items = [
            BatchItem(id=f"{operation_id}_{i}", data=item)
            for i, item in enumerate(items)
        ]
        
        # Create operation
        operation = BatchOperation(
            id=operation_id,
            items=batch_items,
            processor=processor,
            max_concurrent=max_concurrent,
            stop_on_error=stop_on_error,
            metadata=metadata or {}
        )
        
        self.operations[operation_id] = operation
        operation.items[0].status = BatchStatus.PROCESSING  # Mark as processing
        
        # Process items
        semaphore = asyncio.Semaphore(max_concurrent)
        successful = 0
        failed = 0
        
        async def process_item(item: BatchItem):
            async with semaphore:
                try:
                    item.status = BatchStatus.PROCESSING
                    result = await processor(item.data)
                    item.result = result
                    item.status = BatchStatus.COMPLETED
                    return True
                except Exception as e:
                    item.error = str(e)
                    item.status = BatchStatus.FAILED
                    return False
        
        # Execute with concurrency control
        tasks = [process_item(item) for item in batch_items]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Count results
        for success in results:
            if success is True:
                successful += 1
            else:
                failed += 1
        
        # Determine status
        if failed == 0:
            status = BatchStatus.COMPLETED
        elif successful == 0:
            status = BatchStatus.FAILED
        else:
            status = BatchStatus.PARTIAL
        
        duration = (datetime.now() - start).total_seconds()
        
        result = BatchResult(
            operation_id=operation_id,
            status=status,
            total_items=len(items),
            successful_items=successful,
            failed_items=failed,
            results=batch_items,
            duration=duration
        )
        
        self.results[operation_id] = result
        if len(self.results) > self.max_results:
            # Remove oldest
            oldest_id = min(self.results.keys(), key=lambda k: self.results[k].timestamp)
            del self.results[oldest_id]
        
        return result
    
    def get_result(self, operation_id: str) -> Optional[BatchResult]:
        """Get batch result by operation ID."""
        return self.results.get(operation_id)
    
    def get_operation(self, operation_id: str) -> Optional[BatchOperation]:
        """Get operation by ID."""
        return self.operations.get(operation_id)
    
    def get_recent_results(self, limit: int = 100) -> List[BatchResult]:
        """Get recent batch results."""
        return sorted(
            self.results.values(),
            key=lambda r: r.timestamp,
            reverse=True
        )[:limit]

