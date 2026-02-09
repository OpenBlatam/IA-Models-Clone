"""
Batch Processing Service
========================

Service for processing multiple clothing changes or face swaps in batch.
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Callable, Awaitable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from services.clothing_service import ClothingChangeService
from services.webhook_service import get_webhook_service, WebhookEvent
from utils.helpers import generate_prompt_id

logger = logging.getLogger(__name__)

# Constants
DEFAULT_MAX_CONCURRENT = 5
DEFAULT_GUIDANCE_SCALE = 50.0
DEFAULT_NUM_STEPS = 12
OPERATION_TYPE_CLOTHING_CHANGE = "clothing_change"
OPERATION_TYPE_FACE_SWAP = "face_swap"
MAX_BATCH_ITEMS = 1000  # Maximum items per batch
MIN_MAX_CONCURRENT = 1  # Minimum concurrent operations
MAX_MAX_CONCURRENT = 50  # Maximum concurrent operations


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
    operation_type: str  # "clothing_change" or "face_swap"
    items: List[BatchItem]
    status: BatchStatus = BatchStatus.PENDING
    total_items: int = 0
    completed_items: int = 0
    failed_items: int = 0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class BatchProcessingService:
    """
    Service for batch processing of clothing changes and face swaps.
    
    Features:
    - Parallel processing with concurrency control
    - Progress tracking
    - Error handling per item
    - Result aggregation
    """
    
    def __init__(self, max_concurrent: int = DEFAULT_MAX_CONCURRENT):
        """
        Initialize batch processing service.
        
        Args:
            max_concurrent: Maximum concurrent operations (default: 5)
                Will be clamped between MIN_MAX_CONCURRENT and MAX_MAX_CONCURRENT
        """
        self.clothing_service = ClothingChangeService()
        self.max_concurrent = max(MIN_MAX_CONCURRENT, min(max_concurrent, MAX_MAX_CONCURRENT))
        self.active_batches: Dict[str, BatchOperation] = {}
        self.webhook_service = get_webhook_service()
    
    def _validate_batch_items(self, items: List[Dict[str, Any]], operation_type: str) -> None:
        """
        Validate batch items.
        
        Args:
            items: List of items to validate
            operation_type: Type of operation ("clothing_change" or "face_swap")
            
        Raises:
            ValueError: If items are invalid
        """
        if not items:
            raise ValueError("Items list cannot be empty")
        
        if not isinstance(items, list):
            raise ValueError("Items must be a list")
        
        if len(items) > MAX_BATCH_ITEMS:
            raise ValueError(
                f"Batch size exceeds maximum of {MAX_BATCH_ITEMS} items. "
                f"Got {len(items)} items."
            )
        
        # Validate operation type
        if operation_type not in (OPERATION_TYPE_CLOTHING_CHANGE, OPERATION_TYPE_FACE_SWAP):
            raise ValueError(
                f"Invalid operation_type: {operation_type}. "
                f"Must be '{OPERATION_TYPE_CLOTHING_CHANGE}' or '{OPERATION_TYPE_FACE_SWAP}'"
            )
        
        # Validate each item
        for i, item in enumerate(items):
            if not isinstance(item, dict):
                raise ValueError(f"Item {i} must be a dictionary")
            
            # Validate image_url (required for all operations)
            image_url = item.get("image_url")
            if not image_url or not isinstance(image_url, str) or not image_url.strip():
                raise ValueError(f"Item {i} missing or invalid required field: image_url")
            
            # Validate face_url for face swap operations
            if operation_type == OPERATION_TYPE_FACE_SWAP:
                face_url = item.get("face_url")
                if not face_url or not isinstance(face_url, str) or not face_url.strip():
                    raise ValueError(f"Item {i} missing or invalid required field: face_url")
    
    def _validate_max_concurrent(self, max_concurrent: Optional[int]) -> int:
        """
        Validate and clamp max_concurrent value.
        
        Args:
            max_concurrent: Maximum concurrent operations to validate
            
        Returns:
            Validated max_concurrent value (clamped to valid range)
        """
        if max_concurrent is None:
            return self.max_concurrent
        
        return max(MIN_MAX_CONCURRENT, min(max_concurrent, MAX_MAX_CONCURRENT))
    
    def _create_batch_items(
        self,
        items: List[Dict[str, Any]],
        batch_id: str,
        operation_type: str
    ) -> List[BatchItem]:
        """
        Create BatchItem objects from request items.
        
        Args:
            items: List of request items
            batch_id: Batch operation ID
            operation_type: Type of operation
            
        Returns:
            List of BatchItem objects
        """
        batch_items = []
        for i, item in enumerate(items):
            batch_item = BatchItem(
                item_id=f"{batch_id}_{i}",
                image_url=item["image_url"],
                clothing_description=item.get("clothing_description"),
                face_url=item.get("face_url"),
                mask_url=item.get("mask_url"),
                character_name=item.get("character_name"),
            )
            batch_items.append(batch_item)
        return batch_items
    
    def _create_batch_operation(
        self,
        batch_id: str,
        operation_type: str,
        batch_items: List[BatchItem]
    ) -> BatchOperation:
        """
        Create and register a batch operation.
        
        Args:
            batch_id: Batch operation ID
            operation_type: Type of operation
            batch_items: List of batch items
            
        Returns:
            BatchOperation object
        """
        batch_op = BatchOperation(
            batch_id=batch_id,
            operation_type=operation_type,
            items=batch_items,
            total_items=len(batch_items),
            started_at=datetime.now()
        )
        self.active_batches[batch_id] = batch_op
        batch_op.status = BatchStatus.PROCESSING
        return batch_op
    
    def _prepare_item_for_processing(self, item: BatchItem) -> None:
        """
        Prepare item for processing by setting status and timestamp.
        
        Args:
            item: BatchItem to prepare
        """
        item.started_at = datetime.now()
        item.status = BatchStatus.PROCESSING
    
    async def _process_clothing_change_item(
        self,
        item: BatchItem,
        request_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process a single clothing change item.
        
        Args:
            item: BatchItem to process
            request_data: Request data dictionary
            
        Returns:
            Result dictionary
        """
        self._prepare_item_for_processing(item)
        
        try:
            result = await self.clothing_service.change_clothing(
                image_url=request_data["image_url"],
                clothing_description=request_data.get("clothing_description", ""),
                mask_url=request_data.get("mask_url"),
                character_name=request_data.get("character_name"),
                negative_prompt=request_data.get("negative_prompt", ""),
                guidance_scale=request_data.get("guidance_scale", DEFAULT_GUIDANCE_SCALE),
                num_steps=request_data.get("num_steps", DEFAULT_NUM_STEPS),
                seed=request_data.get("seed"),
                optimize_prompt=request_data.get("optimize_prompt", True)
            )
            
            return self._build_item_result(item, result)
            
        except Exception as e:
            logger.error(f"Error processing batch item {item.item_id}: {e}", exc_info=True)
            return self._build_item_error(item, str(e))
    
    async def _process_face_swap_item(
        self,
        item: BatchItem,
        request_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process a single face swap item.
        
        Args:
            item: BatchItem to process
            request_data: Request data dictionary
            
        Returns:
            Result dictionary
        """
        self._prepare_item_for_processing(item)
        
        try:
            result = await self.clothing_service.face_swap(
                image_url=request_data["image_url"],
                face_url=request_data["face_url"],
                mask_url=request_data.get("mask_url"),
                character_name=request_data.get("character_name"),
                prompt=request_data.get("prompt"),
                negative_prompt=request_data.get("negative_prompt", ""),
                guidance_scale=request_data.get("guidance_scale", DEFAULT_GUIDANCE_SCALE),
                num_steps=request_data.get("num_steps", DEFAULT_NUM_STEPS),
                seed=request_data.get("seed"),
                optimize_prompt=request_data.get("optimize_prompt", True)
            )
            
            return self._build_item_result(item, result)
            
        except Exception as e:
            logger.error(f"Error processing batch item {item.item_id}: {e}", exc_info=True)
            return self._build_item_error(item, str(e))
    
    def _build_item_result(self, item: BatchItem, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build result dictionary from item and result.
        
        Args:
            item: BatchItem that was processed
            result: Result from service call
            
        Returns:
            Result dictionary
        """
        item.result = result
        item.prompt_id = result.get("prompt_id")
        item.status = BatchStatus.COMPLETED if result.get("success") else BatchStatus.FAILED
        item.error = result.get("error")
        item.completed_at = datetime.now()
        
        return {
            "item_id": item.item_id,
            "success": result.get("success", False),
            "prompt_id": item.prompt_id,
            "result": result,
            "error": item.error
        }
    
    def _build_item_error(self, item: BatchItem, error_message: str) -> Dict[str, Any]:
        """
        Build error result dictionary.
        
        Args:
            item: BatchItem that failed
            error_message: Error message
            
        Returns:
            Error result dictionary
        """
        item.status = BatchStatus.FAILED
        item.error = error_message
        item.completed_at = datetime.now()
        
        return {
            "item_id": item.item_id,
            "success": False,
            "error": error_message
        }
    
    async def _process_batch_items(
        self,
        batch_op: BatchOperation,
        items: List[Dict[str, Any]],
        max_concurrent: int,
        process_item_func: Callable[[BatchItem, Dict[str, Any]], Awaitable[Dict[str, Any]]],
        progress_callback: Optional[Callable[[int, int], Awaitable[None]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Process batch items with concurrency control and progress tracking.
        
        Args:
            batch_op: BatchOperation object
            items: List of request items
            max_concurrent: Maximum concurrent operations
            process_item_func: Function to process a single item
            progress_callback: Optional progress callback
            
        Returns:
            List of result dictionaries
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        results = []
        
        async def process_with_semaphore(item: BatchItem, request_data: Dict[str, Any]):
            """Process item with semaphore control"""
            async with semaphore:
                return await process_item_func(item, request_data)
        
        # Create tasks
        tasks = [
            process_with_semaphore(batch_op.items[i], items[i])
            for i in range(len(items))
        ]
        
        # Process with progress tracking
        completed = 0
        for coro in asyncio.as_completed(tasks):
            result = await coro
            results.append(result)
            completed += 1
            
            if result.get("success"):
                batch_op.completed_items += 1
            else:
                batch_op.failed_items += 1
            
            # Call progress callback if provided
            if progress_callback:
                try:
                    await progress_callback(completed, len(items))
                except Exception as e:
                    logger.warning(f"Progress callback error: {e}")
        
        return results
    
    def _build_batch_response(
        self,
        batch_op: BatchOperation,
        results: List[Dict[str, Any]],
        duration: float
    ) -> Dict[str, Any]:
        """
        Build final batch response.
        
        Args:
            batch_op: BatchOperation object
            results: List of result dictionaries
            duration: Total duration in seconds
            
        Returns:
            Final batch response dictionary
        """
        batch_op.status = BatchStatus.COMPLETED
        batch_op.completed_at = datetime.now()
        
        logger.info(
            f"Batch {batch_op.operation_type} completed: "
            f"{batch_op.completed_items} succeeded, "
            f"{batch_op.failed_items} failed in {duration:.2f}s"
        )
        
        return {
            "batch_id": batch_op.batch_id,
            "operation_type": batch_op.operation_type,
            "total_items": batch_op.total_items,
            "completed": batch_op.completed_items,
            "failed": batch_op.failed_items,
            "results": results,
            "duration": duration,
            "status": batch_op.status.value
        }
    
    async def batch_clothing_change(
        self,
        items: List[Dict[str, Any]],
        max_concurrent: Optional[int] = None,
        progress_callback: Optional[Callable[[int, int], Awaitable[None]]] = None
    ) -> Dict[str, Any]:
        """
        Process multiple clothing changes in batch.
        
        Args:
            items: List of clothing change requests, each with:
                - image_url: str (required)
                - clothing_description: Optional[str]
                - mask_url: Optional[str]
                - character_name: Optional[str]
                - negative_prompt: Optional[str]
                - guidance_scale: Optional[float] (default: 50.0)
                - num_steps: Optional[int] (default: 12)
                - seed: Optional[int]
                - optimize_prompt: Optional[bool] (default: True)
            max_concurrent: Maximum concurrent operations (default: from config)
            progress_callback: Optional callback for progress updates (current, total)
            
        Returns:
            Dictionary with batch results:
            - batch_id: str
            - operation_type: str
            - total_items: int
            - completed: int
            - failed: int
            - results: List[Dict] - Individual results
            - duration: float - Total duration in seconds
            - status: str
            
        Raises:
            ValueError: If items are invalid
        """
        # Validate inputs
        self._validate_batch_items(items, OPERATION_TYPE_CLOTHING_CHANGE)
        max_concurrent = self._validate_max_concurrent(max_concurrent)
        
        batch_id = generate_prompt_id("batch", str(time.time()))
        
        logger.info(f"Starting batch clothing change: {len(items)} items, max_concurrent={max_concurrent}")
        
        start_time = time.time()
        
        # Create batch items and operation
        batch_items = self._create_batch_items(items, batch_id, OPERATION_TYPE_CLOTHING_CHANGE)
        batch_op = self._create_batch_operation(batch_id, OPERATION_TYPE_CLOTHING_CHANGE, batch_items)
        
        # Process items
        results = await self._process_batch_items(
            batch_op,
            items,
            max_concurrent,
            self._process_clothing_change_item,
            progress_callback
        )
        
        duration = time.time() - start_time
        
        # Build and return response
        return self._build_batch_response(batch_op, results, duration)
    
    async def batch_face_swap(
        self,
        items: List[Dict[str, Any]],
        max_concurrent: Optional[int] = None,
        progress_callback: Optional[Callable[[int, int], Awaitable[None]]] = None
    ) -> Dict[str, Any]:
        """
        Process multiple face swaps in batch.
        
        Args:
            items: List of face swap requests, each with:
                - image_url: str (required)
                - face_url: str (required)
                - mask_url: Optional[str]
                - character_name: Optional[str]
                - prompt: Optional[str]
                - negative_prompt: Optional[str]
                - guidance_scale: Optional[float] (default: 50.0)
                - num_steps: Optional[int] (default: 12)
                - seed: Optional[int]
                - optimize_prompt: Optional[bool] (default: True)
            max_concurrent: Maximum concurrent operations (default: from config)
            progress_callback: Optional callback for progress updates (current, total)
            
        Returns:
            Dictionary with batch results:
            - batch_id: str
            - operation_type: str
            - total_items: int
            - completed: int
            - failed: int
            - results: List[Dict] - Individual results
            - duration: float - Total duration in seconds
            - status: str
            
        Raises:
            ValueError: If items are invalid
        """
        # Validate inputs
        self._validate_batch_items(items, OPERATION_TYPE_FACE_SWAP)
        max_concurrent = self._validate_max_concurrent(max_concurrent)
        
        batch_id = generate_prompt_id("batch_face_swap", str(time.time()))
        
        logger.info(f"Starting batch face swap: {len(items)} items, max_concurrent={max_concurrent}")
        
        start_time = time.time()
        
        # Create batch items and operation
        batch_items = self._create_batch_items(items, batch_id, OPERATION_TYPE_FACE_SWAP)
        batch_op = self._create_batch_operation(batch_id, OPERATION_TYPE_FACE_SWAP, batch_items)
        
        # Process items
        results = await self._process_batch_items(
            batch_op,
            items,
            max_concurrent,
            self._process_face_swap_item,
            progress_callback
        )
        
        duration = time.time() - start_time
        
        # Build and return response
        return self._build_batch_response(batch_op, results, duration)
    
    async def get_batch_status(self, batch_id: str) -> Dict[str, Any]:
        """
        Get status of a batch operation.
        
        Args:
            batch_id: Batch operation ID
            
        Returns:
            Dictionary with batch status:
            - batch_id: str
            - operation_type: str
            - status: str
            - total_items: int
            - completed_items: int
            - failed_items: int
            - progress: float - Progress percentage (0-100)
            - started_at: Optional[str] - ISO format timestamp
            - completed_at: Optional[str] - ISO format timestamp
            - error: Optional[str] - Error message if batch not found
        """
        if batch_id not in self.active_batches:
            return {
                "batch_id": batch_id,
                "error": "Batch not found"
            }
        
        batch_op = self.active_batches[batch_id]
        
        # Calculate progress percentage
        total_processed = batch_op.completed_items + batch_op.failed_items
        progress = (total_processed / batch_op.total_items * 100) if batch_op.total_items > 0 else 0.0
        
        return {
            "batch_id": batch_id,
            "operation_type": batch_op.operation_type,
            "status": batch_op.status.value,
            "total_items": batch_op.total_items,
            "completed_items": batch_op.completed_items,
            "failed_items": batch_op.failed_items,
            "progress": round(progress, 2),
            "started_at": batch_op.started_at.isoformat() if batch_op.started_at else None,
            "completed_at": batch_op.completed_at.isoformat() if batch_op.completed_at else None
        }
    
    async def cancel_batch(self, batch_id: str) -> Dict[str, Any]:
        """
        Cancel a batch operation.
        
        Attempts to cancel all active prompts in the batch operation.
        
        Args:
            batch_id: Batch operation ID
            
        Returns:
            Dictionary with cancellation result:
            - batch_id: str
            - success: bool
            - cancelled_prompts: int - Number of prompts cancelled
            - message: str
            - error: Optional[str] - Error message if batch not found
        """
        if batch_id not in self.active_batches:
            return {
                "batch_id": batch_id,
                "success": False,
                "error": "Batch not found"
            }
        
        batch_op = self.active_batches[batch_id]
        
        # Don't cancel if already completed or cancelled
        if batch_op.status in (BatchStatus.COMPLETED, BatchStatus.CANCELLED):
            return {
                "batch_id": batch_id,
                "success": False,
                "error": f"Batch is already {batch_op.status.value}"
            }
        
        batch_op.status = BatchStatus.CANCELLED
        
        # Cancel individual prompts if possible
        cancelled_count = 0
        for item in batch_op.items:
            if item.status == BatchStatus.PROCESSING and item.prompt_id:
                try:
                    cancel_result = await self.clothing_service.comfyui_service.cancel_prompt(item.prompt_id)
                    if cancel_result.get("success"):
                        cancelled_count += 1
                        item.status = BatchStatus.CANCELLED
                    else:
                        logger.warning(f"Failed to cancel prompt {item.prompt_id}: {cancel_result.get('error')}")
                except Exception as e:
                    logger.warning(f"Error cancelling prompt {item.prompt_id}: {e}", exc_info=True)
        
        logger.info(f"Cancelled batch {batch_id}: {cancelled_count} prompts cancelled")
        
        return {
            "batch_id": batch_id,
            "success": True,
            "cancelled_prompts": cancelled_count,
            "message": "Batch operation cancelled"
        }
    
    async def list_batches(
        self,
        status_filter: Optional[BatchStatus] = None,
        operation_type_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        List all active batch operations with optional filtering.
        
        Args:
            status_filter: Optional status to filter by
            operation_type_filter: Optional operation type to filter by
            
        Returns:
            List of batch status dictionaries
        """
        batches = []
        for batch_id, batch_op in self.active_batches.items():
            # Apply filters
            if status_filter and batch_op.status != status_filter:
                continue
            if operation_type_filter and batch_op.operation_type != operation_type_filter:
                continue
            
            # Calculate progress
            total_processed = batch_op.completed_items + batch_op.failed_items
            progress = (total_processed / batch_op.total_items * 100) if batch_op.total_items > 0 else 0.0
            
            batches.append({
                "batch_id": batch_id,
                "operation_type": batch_op.operation_type,
                "status": batch_op.status.value,
                "total_items": batch_op.total_items,
                "completed_items": batch_op.completed_items,
                "failed_items": batch_op.failed_items,
                "progress": round(progress, 2),
                "started_at": batch_op.started_at.isoformat() if batch_op.started_at else None,
                "completed_at": batch_op.completed_at.isoformat() if batch_op.completed_at else None
            })
        
        return batches
    
    async def cleanup_completed_batches(
        self,
        older_than_hours: float = 24.0
    ) -> Dict[str, Any]:
        """
        Clean up completed or cancelled batches older than specified hours.
        
        Args:
            older_than_hours: Remove batches older than this many hours (default: 24.0)
            
        Returns:
            Dictionary with cleanup results:
            - removed_count: int - Number of batches removed
            - remaining_count: int - Number of batches remaining
        """
        if older_than_hours <= 0:
            raise ValueError("older_than_hours must be positive")
        
        cutoff_time = datetime.now().timestamp() - (older_than_hours * 3600)
        removed_count = 0
        batch_ids_to_remove = []
        
        for batch_id, batch_op in self.active_batches.items():
            # Only remove completed or cancelled batches
            if batch_op.status not in (BatchStatus.COMPLETED, BatchStatus.CANCELLED):
                continue
            
            # Check if batch is old enough
            if batch_op.completed_at:
                batch_timestamp = batch_op.completed_at.timestamp()
            elif batch_op.started_at:
                batch_timestamp = batch_op.started_at.timestamp()
            else:
                continue
            
            if batch_timestamp < cutoff_time:
                batch_ids_to_remove.append(batch_id)
        
        # Remove batches
        for batch_id in batch_ids_to_remove:
            del self.active_batches[batch_id]
            removed_count += 1
        
        remaining_count = len(self.active_batches)
        
        if removed_count > 0:
            logger.info(
                f"Cleaned up {removed_count} batch(es) older than {older_than_hours} hours. "
                f"{remaining_count} batch(es) remaining."
            )
        
        return {
            "removed_count": removed_count,
            "remaining_count": remaining_count
        }
    
    async def _send_batch_webhook(
        self,
        batch_op: BatchOperation,
        response: Dict[str, Any]
    ) -> None:
        """
        Send webhook notification for batch completion.
        
        Args:
            batch_op: BatchOperation object
            response: Batch response dictionary
        """
        try:
            event = WebhookEvent(
                event_type="batch_completed",
                batch_id=batch_op.batch_id,
                data={
                    "operation_type": batch_op.operation_type,
                    "total_items": batch_op.total_items,
                    "completed": batch_op.completed_items,
                    "failed": batch_op.failed_items,
                    "duration": response.get("duration"),
                    "status": batch_op.status.value
                }
            )
            await self.webhook_service.send_event(event)
        except Exception as e:
            # Don't fail the main operation if webhook fails
            logger.warning(f"Failed to send batch webhook notification: {e}")
    
    async def get_batch_details(self, batch_id: str) -> Dict[str, Any]:
        """
        Get detailed information about a batch operation including all items.
        
        Args:
            batch_id: Batch operation ID
            
        Returns:
            Dictionary with detailed batch information:
            - batch_id: str
            - operation_type: str
            - status: str
            - total_items: int
            - completed_items: int
            - failed_items: int
            - pending_items: int
            - processing_items: int
            - progress: float
            - started_at: Optional[str]
            - completed_at: Optional[str]
            - items: List[Dict] - Detailed information about each item
            - statistics: Dict - Aggregated statistics
            - error: Optional[str] - Error message if batch not found
        """
        if batch_id not in self.active_batches:
            return {
                "batch_id": batch_id,
                "error": "Batch not found"
            }
        
        batch_op = self.active_batches[batch_id]
        
        # Calculate item status counts
        status_counts = {
            BatchStatus.PENDING: 0,
            BatchStatus.PROCESSING: 0,
            BatchStatus.COMPLETED: 0,
            BatchStatus.FAILED: 0,
            BatchStatus.CANCELLED: 0
        }
        
        for item in batch_op.items:
            status_counts[item.status] = status_counts.get(item.status, 0) + 1
        
        # Calculate progress
        total_processed = batch_op.completed_items + batch_op.failed_items
        progress = (total_processed / batch_op.total_items * 100) if batch_op.total_items > 0 else 0.0
        
        # Build item details
        items_details = []
        for item in batch_op.items:
            item_detail = {
                "item_id": item.item_id,
                "status": item.status.value,
                "image_url": item.image_url,
                "clothing_description": item.clothing_description,
                "face_url": item.face_url,
                "mask_url": item.mask_url,
                "character_name": item.character_name,
                "prompt_id": item.prompt_id,
                "error": item.error,
                "started_at": item.started_at.isoformat() if item.started_at else None,
                "completed_at": item.completed_at.isoformat() if item.completed_at else None
            }
            
            # Calculate item duration if available
            if item.started_at and item.completed_at:
                duration = (item.completed_at - item.started_at).total_seconds()
                item_detail["duration_seconds"] = round(duration, 2)
            
            items_details.append(item_detail)
        
        # Calculate statistics
        statistics = {
            "status_counts": {status.value: count for status, count in status_counts.items()},
            "success_rate": (
                (batch_op.completed_items / batch_op.total_items * 100)
                if batch_op.total_items > 0 else 0.0
            ),
            "failure_rate": (
                (batch_op.failed_items / batch_op.total_items * 100)
                if batch_op.total_items > 0 else 0.0
            )
        }
        
        # Calculate average processing time for completed items
        completed_items_with_duration = [
            item for item in batch_op.items
            if item.status == BatchStatus.COMPLETED
            and item.started_at
            and item.completed_at
        ]
        
        if completed_items_with_duration:
            durations = [
                (item.completed_at - item.started_at).total_seconds()
                for item in completed_items_with_duration
            ]
            statistics["avg_processing_time_seconds"] = round(sum(durations) / len(durations), 2)
            statistics["min_processing_time_seconds"] = round(min(durations), 2)
            statistics["max_processing_time_seconds"] = round(max(durations), 2)
        
        return {
            "batch_id": batch_id,
            "operation_type": batch_op.operation_type,
            "status": batch_op.status.value,
            "total_items": batch_op.total_items,
            "completed_items": batch_op.completed_items,
            "failed_items": batch_op.failed_items,
            "pending_items": status_counts[BatchStatus.PENDING],
            "processing_items": status_counts[BatchStatus.PROCESSING],
            "progress": round(progress, 2),
            "started_at": batch_op.started_at.isoformat() if batch_op.started_at else None,
            "completed_at": batch_op.completed_at.isoformat() if batch_op.completed_at else None,
            "items": items_details,
            "statistics": statistics
        }
    
    async def get_batch_statistics(self) -> Dict[str, Any]:
        """
        Get aggregated statistics across all active batches.
        
        Returns:
            Dictionary with aggregated statistics:
            - total_batches: int
            - batches_by_status: Dict[str, int]
            - batches_by_operation_type: Dict[str, int]
            - total_items_processed: int
            - total_items_completed: int
            - total_items_failed: int
            - overall_success_rate: float
        """
        total_batches = len(self.active_batches)
        batches_by_status: Dict[str, int] = {}
        batches_by_operation_type: Dict[str, int] = {}
        total_items_processed = 0
        total_items_completed = 0
        total_items_failed = 0
        
        for batch_op in self.active_batches.values():
            # Count by status
            status_value = batch_op.status.value
            batches_by_status[status_value] = batches_by_status.get(status_value, 0) + 1
            
            # Count by operation type
            op_type = batch_op.operation_type
            batches_by_operation_type[op_type] = batches_by_operation_type.get(op_type, 0) + 1
            
            # Aggregate item counts
            total_items_processed += batch_op.total_items
            total_items_completed += batch_op.completed_items
            total_items_failed += batch_op.failed_items
        
        overall_success_rate = (
            (total_items_completed / total_items_processed * 100)
            if total_items_processed > 0 else 0.0
        )
        
        return {
            "total_batches": total_batches,
            "batches_by_status": batches_by_status,
            "batches_by_operation_type": batches_by_operation_type,
            "total_items_processed": total_items_processed,
            "total_items_completed": total_items_completed,
            "total_items_failed": total_items_failed,
            "overall_success_rate": round(overall_success_rate, 2)
        }
    
    async def estimate_remaining_time(self, batch_id: str) -> Dict[str, Any]:
        """
        Estimate remaining time for a batch operation.
        
        Args:
            batch_id: Batch operation ID
            
        Returns:
            Dictionary with time estimates:
            - batch_id: str
            - estimated_seconds_remaining: Optional[float]
            - estimated_completion_time: Optional[str]
            - items_remaining: int
            - avg_time_per_item: Optional[float]
            - error: Optional[str]
        """
        if batch_id not in self.active_batches:
            return {
                "batch_id": batch_id,
                "error": "Batch not found"
            }
        
        batch_op = self.active_batches[batch_id]
        
        # Count remaining items
        items_remaining = sum(
            1 for item in batch_op.items
            if item.status in (BatchStatus.PENDING, BatchStatus.PROCESSING)
        )
        
        if items_remaining == 0:
            return {
                "batch_id": batch_id,
                "estimated_seconds_remaining": 0.0,
                "estimated_completion_time": batch_op.completed_at.isoformat() if batch_op.completed_at else None,
                "items_remaining": 0,
                "avg_time_per_item": None
            }
        
        # Calculate average time per item from completed items
        completed_items_with_duration = [
            item for item in batch_op.items
            if item.status == BatchStatus.COMPLETED
            and item.started_at
            and item.completed_at
        ]
        
        if not completed_items_with_duration:
            return {
                "batch_id": batch_id,
                "estimated_seconds_remaining": None,
                "estimated_completion_time": None,
                "items_remaining": items_remaining,
                "avg_time_per_item": None,
                "message": "No completed items yet, cannot estimate"
            }
        
        durations = [
            (item.completed_at - item.started_at).total_seconds()
            for item in completed_items_with_duration
        ]
        avg_time_per_item = sum(durations) / len(durations)
        
        # Estimate remaining time
        estimated_seconds_remaining = avg_time_per_item * items_remaining
        
        # Estimate completion time
        if batch_op.started_at:
            estimated_completion = datetime.now().timestamp() + estimated_seconds_remaining
            estimated_completion_time = datetime.fromtimestamp(estimated_completion).isoformat()
        else:
            estimated_completion_time = None
        
        return {
            "batch_id": batch_id,
            "estimated_seconds_remaining": round(estimated_seconds_remaining, 2),
            "estimated_completion_time": estimated_completion_time,
            "items_remaining": items_remaining,
            "avg_time_per_item": round(avg_time_per_item, 2)
        }
    
    async def retry_failed_items(
        self,
        batch_id: str,
        max_concurrent: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Retry failed items from a batch operation.
        
        Args:
            batch_id: Batch operation ID
            max_concurrent: Maximum concurrent operations for retry
            
        Returns:
            Dictionary with retry results:
            - batch_id: str
            - retry_batch_id: str - ID of new batch for retries
            - items_retried: int
            - error: Optional[str]
        """
        if batch_id not in self.active_batches:
            return {
                "batch_id": batch_id,
                "error": "Batch not found"
            }
        
        batch_op = self.active_batches[batch_id]
        
        # Get failed items
        failed_items = [
            item for item in batch_op.items
            if item.status == BatchStatus.FAILED
        ]
        
        if not failed_items:
            return {
                "batch_id": batch_id,
                "retry_batch_id": None,
                "items_retried": 0,
                "message": "No failed items to retry"
            }
        
        # Reconstruct original request data from failed items
        retry_items = []
        for item in failed_items:
            item_data = {
                "image_url": item.image_url,
                "clothing_description": item.clothing_description,
                "mask_url": item.mask_url,
                "character_name": item.character_name
            }
            
            if batch_op.operation_type == OPERATION_TYPE_FACE_SWAP:
                item_data["face_url"] = item.face_url
            
            retry_items.append(item_data)
        
        # Create new batch for retries
        if batch_op.operation_type == OPERATION_TYPE_CLOTHING_CHANGE:
            retry_result = await self.batch_clothing_change(
                items=retry_items,
                max_concurrent=max_concurrent
            )
        else:
            retry_result = await self.batch_face_swap(
                items=retry_items,
                max_concurrent=max_concurrent
            )
        
        return {
            "batch_id": batch_id,
            "retry_batch_id": retry_result.get("batch_id"),
            "items_retried": len(failed_items),
            "retry_batch_status": retry_result.get("status")
        }

