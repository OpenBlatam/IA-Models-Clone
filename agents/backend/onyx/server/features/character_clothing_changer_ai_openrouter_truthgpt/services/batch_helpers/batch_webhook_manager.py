"""
Batch Webhook Manager
=====================
Manages webhook notifications for batch operations
"""

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class BatchWebhookManager:
    """
    Manages webhook notifications for batch operations.
    """
    
    def __init__(self, webhook_service: Optional[Any] = None):
        """
        Initialize batch webhook manager.
        
        Args:
            webhook_service: Optional webhook service instance
        """
        self.webhook_service = webhook_service
    
    async def send_batch_completed(
        self,
        batch_id: str,
        operation_type: str,
        total_items: int,
        completed_items: int,
        failed_items: int,
        duration: Optional[float] = None,
        status: str = "completed"
    ) -> bool:
        """
        Send webhook notification for batch completion.
        
        Args:
            batch_id: Batch identifier
            operation_type: Type of operation
            total_items: Total number of items
            completed_items: Number of completed items
            failed_items: Number of failed items
            duration: Duration in seconds
            status: Batch status
            
        Returns:
            True if webhook sent successfully, False otherwise
        """
        if not self.webhook_service:
            return False
        
        try:
            from services.webhook_service import WebhookEvent
            
            event = WebhookEvent(
                event_type="batch_completed",
                batch_id=batch_id,
                data={
                    "operation_type": operation_type,
                    "total_items": total_items,
                    "completed": completed_items,
                    "failed": failed_items,
                    "duration": duration,
                    "status": status
                }
            )
            await self.webhook_service.send_event(event)
            return True
        except Exception as e:
            logger.warning(f"Failed to send batch webhook notification: {e}")
            return False
    
    async def send_batch_status_update(
        self,
        batch_id: str,
        status: str,
        progress: Optional[float] = None
    ) -> bool:
        """
        Send webhook notification for batch status update.
        
        Args:
            batch_id: Batch identifier
            status: New status
            progress: Optional progress percentage
            
        Returns:
            True if webhook sent successfully, False otherwise
        """
        if not self.webhook_service:
            return False
        
        try:
            from services.webhook_service import WebhookEvent
            
            event = WebhookEvent(
                event_type="batch_status_update",
                batch_id=batch_id,
                data={
                    "status": status,
                    "progress": progress
                }
            )
            await self.webhook_service.send_event(event)
            return True
        except Exception as e:
            logger.warning(f"Failed to send batch status webhook: {e}")
            return False

