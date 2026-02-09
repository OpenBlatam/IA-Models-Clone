"""
Batch Processing Service V2 - Refactored
=========================================
Refactored batch processing service using modular helpers

This version uses helper modules for better separation of concerns:
- BatchValidator: Validation of items and parameters
- BatchItemProcessor: Processing of individual items
- BatchExecutor: Execution with concurrency control
- BatchTracker: Status tracking
- BatchStatistics: Statistics and estimates
- BatchResultBuilder: Result building
- BatchWebhookManager: Webhook notifications
"""

import logging
import time
from typing import Dict, Any, List, Optional, Callable, Awaitable
from datetime import datetime

from services.clothing_service import ClothingChangeService
from services.webhook_service import get_webhook_service
from utils.helpers import generate_prompt_id
from services.batch_helpers import (
    BatchValidator,
    BatchItemProcessor,
    BatchExecutor,
    BatchTracker,
    BatchStatistics,
    BatchResultBuilder,
    BatchWebhookManager,
    BatchStatus,
    OPERATION_TYPE_CLOTHING_CHANGE,
    OPERATION_TYPE_FACE_SWAP,
)

logger = logging.getLogger(__name__)

# Constants
DEFAULT_MAX_CONCURRENT = 5


class BatchProcessingServiceV2:
    """
    Refactored batch processing service using modular helpers.
    
    This service orchestrates the helper modules to provide
    a clean interface for batch operations.
    """
    
    def __init__(self, max_concurrent: int = DEFAULT_MAX_CONCURRENT):
        """
        Initialize batch processing service V2.
        
        Args:
            max_concurrent: Maximum concurrent operations
        """
        self.clothing_service = ClothingChangeService()
        self.max_concurrent = BatchValidator.validate_max_concurrent(max_concurrent)
        
        # Initialize helper modules
        self.item_processor = BatchItemProcessor(self.clothing_service)
        self.executor = BatchExecutor(self.item_processor, self.max_concurrent)
        self.tracker = BatchTracker()
        self.statistics = BatchStatistics()
        self.result_builder = BatchResultBuilder()
        
        # Initialize webhook manager
        webhook_service = get_webhook_service()
        self.webhook_manager = BatchWebhookManager(webhook_service)
        
        logger.info(f"Batch Processing Service V2 initialized (max_concurrent={self.max_concurrent})")
    
    async def batch_clothing_change(
        self,
        items: List[Dict[str, Any]],
        max_concurrent: Optional[int] = None,
        progress_callback: Optional[Callable[[int, int], Awaitable[None]]] = None
    ) -> Dict[str, Any]:
        """
        Process multiple clothing changes in batch.
        
        Args:
            items: List of clothing change requests
            max_concurrent: Maximum concurrent operations
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary with batch results
        """
        # Validate inputs
        BatchValidator.validate_batch_items(items, OPERATION_TYPE_CLOTHING_CHANGE)
        max_concurrent = BatchValidator.validate_max_concurrent(max_concurrent or self.max_concurrent)
        
        # Create batch operation
        batch_id = generate_prompt_id("batch", str(time.time()))
        batch_op = self.tracker.create_batch_operation(
            batch_id,
            OPERATION_TYPE_CLOTHING_CHANGE,
            items
        )
        
        logger.info(f"Starting batch clothing change: {len(items)} items, max_concurrent={max_concurrent}")
        
        start_time = time.time()
        batch_op.status = BatchStatus.PROCESSING
        
        try:
            # Execute batch
            executor = BatchExecutor(self.item_processor, max_concurrent)
            results = await executor.execute_batch(
                items,
                OPERATION_TYPE_CLOTHING_CHANGE,
                progress_callback
            )
            
            # Update batch status
            completed = sum(1 for r in results if r.get("success"))
            failed = len(results) - completed
            
            self.tracker.update_batch_status(
                batch_id,
                BatchStatus.COMPLETED,
                completed,
                failed
            )
            
            # Update item statuses
            for result in results:
                item_id = result.get("item_id")
                if result.get("success"):
                    self.tracker.update_item_status(
                        batch_id,
                        item_id,
                        BatchStatus.COMPLETED,
                        result=result.get("result"),
                        prompt_id=result.get("prompt_id")
                    )
                else:
                    self.tracker.update_item_status(
                        batch_id,
                        item_id,
                        BatchStatus.FAILED,
                        error=result.get("error")
                    )
            
            duration = time.time() - start_time
            
            # Build response
            response = self.result_builder.build_batch_response(
                batch_op,
                results,
                duration
            )
            
            # Send webhook
            await self.webhook_manager.send_batch_completed(
                batch_id,
                OPERATION_TYPE_CLOTHING_CHANGE,
                batch_op.total_items,
                completed,
                failed,
                duration,
                BatchStatus.COMPLETED.value
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error in batch clothing change: {e}", exc_info=True)
            self.tracker.update_batch_status(batch_id, BatchStatus.FAILED)
            raise
    
    async def batch_face_swap(
        self,
        items: List[Dict[str, Any]],
        max_concurrent: Optional[int] = None,
        progress_callback: Optional[Callable[[int, int], Awaitable[None]]] = None
    ) -> Dict[str, Any]:
        """
        Process multiple face swaps in batch.
        
        Args:
            items: List of face swap requests
            max_concurrent: Maximum concurrent operations
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary with batch results
        """
        # Validate inputs
        BatchValidator.validate_batch_items(items, OPERATION_TYPE_FACE_SWAP)
        max_concurrent = BatchValidator.validate_max_concurrent(max_concurrent or self.max_concurrent)
        
        # Create batch operation
        batch_id = generate_prompt_id("batch", str(time.time()))
        batch_op = self.tracker.create_batch_operation(
            batch_id,
            OPERATION_TYPE_FACE_SWAP,
            items
        )
        
        logger.info(f"Starting batch face swap: {len(items)} items, max_concurrent={max_concurrent}")
        
        start_time = time.time()
        batch_op.status = BatchStatus.PROCESSING
        
        try:
            # Execute batch
            executor = BatchExecutor(self.item_processor, max_concurrent)
            results = await executor.execute_batch(
                items,
                OPERATION_TYPE_FACE_SWAP,
                progress_callback
            )
            
            # Update batch status
            completed = sum(1 for r in results if r.get("success"))
            failed = len(results) - completed
            
            self.tracker.update_batch_status(
                batch_id,
                BatchStatus.COMPLETED,
                completed,
                failed
            )
            
            # Update item statuses
            for result in results:
                item_id = result.get("item_id")
                if result.get("success"):
                    self.tracker.update_item_status(
                        batch_id,
                        item_id,
                        BatchStatus.COMPLETED,
                        result=result.get("result"),
                        prompt_id=result.get("prompt_id")
                    )
                else:
                    self.tracker.update_item_status(
                        batch_id,
                        item_id,
                        BatchStatus.FAILED,
                        error=result.get("error")
                    )
            
            duration = time.time() - start_time
            
            # Build response
            response = self.result_builder.build_batch_response(
                batch_op,
                results,
                duration
            )
            
            # Send webhook
            await self.webhook_manager.send_batch_completed(
                batch_id,
                OPERATION_TYPE_FACE_SWAP,
                batch_op.total_items,
                completed,
                failed,
                duration,
                BatchStatus.COMPLETED.value
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error in batch face swap: {e}", exc_info=True)
            self.tracker.update_batch_status(batch_id, BatchStatus.FAILED)
            raise
    
    async def get_batch_status(self, batch_id: str) -> Dict[str, Any]:
        """Get status of a batch operation"""
        batch_op = self.tracker.get_batch(batch_id)
        if not batch_op:
            return {
                "batch_id": batch_id,
                "error": "Batch not found"
            }
        
        return self.result_builder.build_batch_status(batch_op)
    
    async def get_batch_details(self, batch_id: str) -> Dict[str, Any]:
        """Get detailed information about a batch operation"""
        batch_op = self.tracker.get_batch(batch_id)
        if not batch_op:
            return {
                "batch_id": batch_id,
                "error": "Batch not found"
            }
        
        return self.result_builder.build_batch_details(batch_op)
    
    async def cancel_batch(self, batch_id: str) -> Dict[str, Any]:
        """Cancel a batch operation"""
        success = self.tracker.cancel_batch(batch_id)
        
        if success:
            return {
                "batch_id": batch_id,
                "status": "cancelled",
                "message": "Batch cancelled successfully"
            }
        else:
            return {
                "batch_id": batch_id,
                "error": "Batch not found or cannot be cancelled"
            }
    
    async def list_batches(
        self,
        status_filter: Optional[BatchStatus] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """List batch operations"""
        batches = self.tracker.list_batches(status_filter, limit)
        
        return [
            self.result_builder.build_batch_status(batch_op)
            for batch_op in batches
        ]
    
    async def get_batch_statistics(self) -> Dict[str, Any]:
        """Get global batch statistics"""
        all_batches = self.tracker.list_batches()
        return self.statistics.get_global_statistics(all_batches)
    
    async def estimate_remaining_time(self, batch_id: str) -> Dict[str, Any]:
        """Estimate remaining time for batch completion"""
        batch_op = self.tracker.get_batch(batch_id)
        if not batch_op:
            return {
                "batch_id": batch_id,
                "error": "Batch not found"
            }
        
        estimate = self.statistics.estimate_remaining_time(batch_op)
        if estimate is None:
            return {
                "batch_id": batch_id,
                "error": "Cannot estimate remaining time"
            }
        
        return {
            "batch_id": batch_id,
            **estimate
        }
    
    async def retry_failed_items(
        self,
        batch_id: str,
        max_concurrent: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Retry failed items from a batch.
        
        Args:
            batch_id: Batch identifier
            max_concurrent: Maximum concurrent operations
            
        Returns:
            Retry results
        """
        batch_op = self.tracker.get_batch(batch_id)
        if not batch_op:
            return {
                "batch_id": batch_id,
                "error": "Batch not found"
            }
        
        # Get failed items
        failed_items = [
            item for item in batch_op.items
            if item.status == BatchStatus.FAILED
        ]
        
        if not failed_items:
            return {
                "batch_id": batch_id,
                "message": "No failed items to retry",
                "retried_count": 0
            }
        
        # Prepare items for retry
        items_to_retry = []
        for item in failed_items:
            item_dict = {
                "item_id": item.item_id,
                "image_url": item.image_url,
                "clothing_description": item.clothing_description,
                "face_url": item.face_url,
                "mask_url": item.mask_url,
                "character_name": item.character_name
            }
            items_to_retry.append(item_dict)
        
        # Execute retry
        max_concurrent = BatchValidator.validate_max_concurrent(max_concurrent or self.max_concurrent)
        executor = BatchExecutor(self.item_processor, max_concurrent)
        
        results = await executor.execute_batch(
            items_to_retry,
            batch_op.operation_type
        )
        
        # Update item statuses
        retried_count = 0
        for result in results:
            item_id = result.get("item_id")
            if result.get("success"):
                self.tracker.update_item_status(
                    batch_id,
                    item_id,
                    BatchStatus.COMPLETED,
                    result=result.get("result"),
                    prompt_id=result.get("prompt_id")
                )
                retried_count += 1
            else:
                # Keep as failed
                self.tracker.update_item_status(
                    batch_id,
                    item_id,
                    BatchStatus.FAILED,
                    error=result.get("error")
                )
        
        return {
            "batch_id": batch_id,
            "retried_count": retried_count,
            "total_failed": len(failed_items),
            "results": results
        }
    
    async def cleanup_completed_batches(self, max_age_hours: int = 24) -> Dict[str, Any]:
        """Cleanup completed batches older than specified age"""
        removed_count = self.tracker.cleanup_completed_batches(max_age_hours)
        remaining_count = len(self.tracker.active_batches)
        
        return {
            "removed_count": removed_count,
            "remaining_count": remaining_count
        }

