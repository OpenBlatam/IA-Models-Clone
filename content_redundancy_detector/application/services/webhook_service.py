"""
Webhook Application Service
Orchestrates webhook operations
"""

import logging
import time
from typing import Dict, Any, List, Optional
from uuid import uuid4

from ...domain.entities.webhook import (
    WebhookEndpoint, WebhookEventType, WebhookPayload, WebhookDelivery, WebhookDeliveryStatus
)
from ...domain.interfaces.webhook import (
    IWebhookRepository, IWebhookDeliveryRepository, IWebhookSender, IWebhookSigner
)
from ...domain.interfaces import IMessagingService

logger = logging.getLogger(__name__)


class WebhookService:
    """Application service for webhook management"""
    
    def __init__(
        self,
        endpoint_repository: IWebhookRepository,
        delivery_repository: IWebhookDeliveryRepository,
        sender: IWebhookSender,
        signer: IWebhookSigner,
        messaging_service: Optional[IMessagingService] = None
    ):
        self.endpoint_repo = endpoint_repository
        self.delivery_repo = delivery_repository
        self.sender = sender
        self.signer = signer
        self.messaging_service = messaging_service
    
    async def register_endpoint(
        self,
        url: str,
        events: List[str],
        secret: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> WebhookEndpoint:
        """
        Register a new webhook endpoint
        
        Args:
            url: Webhook URL
            events: List of event types to subscribe to
            secret: Optional secret for signature verification
            user_id: Optional user ID
            
        Returns:
            Created WebhookEndpoint
        """
        endpoint_id = str(uuid4())
        event_types = [WebhookEventType(e) for e in events]
        
        endpoint = WebhookEndpoint(
            id=endpoint_id,
            url=url,
            events=event_types,
            secret=secret,
            user_id=user_id
        )
        
        await self.endpoint_repo.save_endpoint(endpoint)
        
        # Publish event (if messaging available)
        if self.messaging_service:
            await self.messaging_service.publish("webhook.endpoint.registered", endpoint.to_dict())
        
        logger.info(f"Webhook endpoint registered: {endpoint_id}")
        return endpoint
    
    async def unregister_endpoint(self, endpoint_id: str) -> bool:
        """Unregister webhook endpoint"""
        success = await self.endpoint_repo.delete_endpoint(endpoint_id)
        
        if success and self.messaging_service:
            await self.messaging_service.publish("webhook.endpoint.unregistered", {
                "endpoint_id": endpoint_id
            })
        
        return success
    
    async def send_event(
        self,
        event_type: WebhookEventType,
        data: Dict[str, Any],
        request_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> List[WebhookDelivery]:
        """
        Send webhook event to all registered endpoints
        
        Args:
            event_type: Type of event
            data: Event data
            request_id: Optional request ID
            user_id: Optional user ID
            
        Returns:
            List of created deliveries
        """
        # Get endpoints that support this event
        endpoints = await self.endpoint_repo.get_endpoints_by_event(event_type)
        
        if not endpoints:
            logger.debug(f"No endpoints registered for event: {event_type.value}")
            return []
        
        # Create payload
        payload = WebhookPayload(
            event=event_type,
            timestamp=time.time(),
            data=data,
            request_id=request_id,
            user_id=user_id
        )
        
        # Create deliveries for each endpoint
        deliveries = []
        for endpoint in endpoints:
            delivery = await self._create_delivery(endpoint, payload)
            deliveries.append(delivery)
            
            # Send asynchronously (fire and forget or queue)
            await self._send_delivery(delivery, endpoint, payload)
        
        return deliveries
    
    async def _create_delivery(
        self,
        endpoint: WebhookEndpoint,
        payload: WebhookPayload
    ) -> WebhookDelivery:
        """Create webhook delivery record"""
        delivery_id = f"{payload.event.value}_{endpoint.id}_{int(time.time() * 1000)}"
        
        delivery = WebhookDelivery(
            id=delivery_id,
            endpoint_id=endpoint.id,
            event=payload.event,
            payload=payload,
            status=WebhookDeliveryStatus.PENDING
        )
        
        await self.delivery_repo.save_delivery(delivery)
        return delivery
    
    async def _send_delivery(
        self,
        delivery: WebhookDelivery,
        endpoint: WebhookEndpoint,
        payload: WebhookPayload
    ):
        """Send webhook delivery"""
        try:
            # Prepare headers
            headers = {
                "Content-Type": "application/json",
                "User-Agent": "Content-Redundancy-Detector/2.0"
            }
            
            # Add signature if secret provided
            if endpoint.secret:
                payload_json = str(payload.to_dict())
                signature = self.signer.sign(payload_json, endpoint.secret)
                headers["X-Webhook-Signature"] = f"sha256={signature}"
            
            # Send webhook
            success = await self.sender.send(endpoint, payload.to_dict(), headers)
            
            if success:
                delivery.mark_delivered()
            else:
                delivery.mark_failed("HTTP request failed")
            
            delivery.increment_attempt()
            await self.delivery_repo.update_delivery(delivery)
            
        except Exception as e:
            logger.error(f"Error sending webhook {delivery.id}: {e}")
            delivery.mark_failed(str(e))
            delivery.increment_attempt()
            await self.delivery_repo.update_delivery(delivery)
            
            # Retry logic if needed
            if delivery.can_retry(endpoint.retry_count):
                delivery.status = WebhookDeliveryStatus.RETRYING
                delivery.next_retry = time.time() + (2 ** delivery.attempts)  # Exponential backoff
                await self.delivery_repo.update_delivery(delivery)
    
    async def retry_failed_deliveries(self, limit: int = 10) -> int:
        """Retry failed webhook deliveries"""
        pending = await self.delivery_repo.list_pending_deliveries(limit)
        
        retry_count = 0
        for delivery in pending:
            if delivery.can_retry(3):  # Default max attempts
                endpoint = await self.endpoint_repo.get_endpoint(delivery.endpoint_id)
                if endpoint:
                    await self._send_delivery(delivery, endpoint, delivery.payload)
                    retry_count += 1
        
        return retry_count
    
    async def get_endpoints(self, user_id: Optional[str] = None) -> List[WebhookEndpoint]:
        """Get all webhook endpoints"""
        return await self.endpoint_repo.list_endpoints(user_id)
    
    async def get_delivery_stats(self) -> Dict[str, Any]:
        """Get webhook delivery statistics"""
        # This would typically aggregate from delivery repository
        # For now, return placeholder
        return {
            "total_deliveries": 0,
            "successful": 0,
            "failed": 0,
            "pending": 0
        }






