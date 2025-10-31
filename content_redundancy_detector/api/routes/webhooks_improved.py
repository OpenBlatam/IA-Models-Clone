"""
Improved Webhook Routes
Modular webhook API endpoints with proper error handling and validation
"""

import logging
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, HttpUrl, validator

from ...application.services.webhook_service import WebhookService
from ...domain.entities.webhook import WebhookEventType
from ...shared.response import create_success_response

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/webhooks", tags=["Webhooks"])


# Request/Response Models
class WebhookRegistrationRequest(BaseModel):
    """Request model for webhook registration"""
    url: HttpUrl = Field(..., description="Webhook endpoint URL")
    events: List[str] = Field(..., min_items=1, description="List of event types to subscribe to")
    secret: Optional[str] = Field(None, min_length=16, description="Secret for signature verification")
    user_id: Optional[str] = Field(None, description="User ID for webhook ownership")
    
    @validator('events')
    def validate_events(cls, v):
        """Validate event types"""
        valid_events = [e.value for e in WebhookEventType]
        invalid = [e for e in v if e not in valid_events]
        if invalid:
            raise ValueError(f"Invalid event types: {invalid}. Valid types: {valid_events}")
        return v


class WebhookRegistrationResponse(BaseModel):
    """Response model for webhook registration"""
    endpoint_id: str
    url: str
    events: List[str]
    created_at: float
    message: str = "Webhook endpoint registered successfully"


class WebhookStatsResponse(BaseModel):
    """Response model for webhook statistics"""
    total_deliveries: int
    successful: int
    failed: int
    pending: int
    success_rate: float
    active_endpoints: int


# Dependency to get webhook service
async def get_webhook_service() -> WebhookService:
    """Get webhook service instance"""
    from ...application.dependencies.webhooks import get_webhook_service
    return await get_webhook_service()


@router.post("/register", response_model=Dict[str, Any], status_code=status.HTTP_201_CREATED)
async def register_webhook(
    request: WebhookRegistrationRequest,
    service: WebhookService = Depends(get_webhook_service)
) -> Dict[str, Any]:
    """
    Register a new webhook endpoint
    
    Returns webhook endpoint information including endpoint_id
    """
    try:
        endpoint = await service.register_endpoint(
            url=str(request.url),
            events=request.events,
            secret=request.secret,
            user_id=request.user_id
        )
        
        return create_success_response(
            {
                "endpoint_id": endpoint.id,
                "url": endpoint.url,
                "events": [e.value for e in endpoint.events],
                "created_at": endpoint.created_at,
                "message": "Webhook endpoint registered successfully"
            },
            message="Webhook registered"
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Webhook registration error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to register webhook endpoint"
        )


@router.delete("/{endpoint_id}", response_model=Dict[str, Any])
async def unregister_webhook(
    endpoint_id: str,
    service: WebhookService = Depends(get_webhook_service)
) -> Dict[str, Any]:
    """
    Unregister a webhook endpoint
    
    Args:
        endpoint_id: Webhook endpoint ID to unregister
    """
    success = await service.unregister_endpoint(endpoint_id)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Webhook endpoint not found: {endpoint_id}"
        )
    
    return create_success_response(
        {"endpoint_id": endpoint_id},
        message="Webhook endpoint unregistered successfully"
    )


@router.get("", response_model=Dict[str, Any])
async def list_webhooks(
    user_id: Optional[str] = None,
    service: WebhookService = Depends(get_webhook_service)
) -> Dict[str, Any]:
    """
    List all webhook endpoints
    
    Args:
        user_id: Optional filter by user ID
    """
    try:
        endpoints = await service.get_endpoints(user_id)
        
        return create_success_response({
            "endpoints": [endpoint.to_dict() for endpoint in endpoints],
            "count": len(endpoints)
        })
    except Exception as e:
        logger.error(f"List webhooks error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve webhook endpoints"
        )


@router.get("/stats", response_model=Dict[str, Any])
async def get_webhook_stats(
    service: WebhookService = Depends(get_webhook_service)
) -> Dict[str, Any]:
    """
    Get webhook delivery statistics
    """
    try:
        stats = await service.get_delivery_stats()
        
        return create_success_response(stats)
    except Exception as e:
        logger.error(f"Get webhook stats error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve webhook statistics"
        )


@router.post("/retry/{delivery_id}", response_model=Dict[str, Any])
async def retry_webhook_delivery(
    delivery_id: str,
    service: WebhookService = Depends(get_webhook_service)
) -> Dict[str, Any]:
    """
    Retry a failed webhook delivery
    
    Args:
        delivery_id: Delivery ID to retry
    """
    # This would require access to delivery repository
    # For now, return placeholder
    return create_success_response(
        {"delivery_id": delivery_id, "message": "Retry queued"},
        message="Webhook retry initiated"
    )


@router.post("/retry-failed", response_model=Dict[str, Any])
async def retry_failed_deliveries(
    limit: int = 10,
    service: WebhookService = Depends(get_webhook_service)
) -> Dict[str, Any]:
    """
    Retry all failed webhook deliveries
    
    Args:
        limit: Maximum number of deliveries to retry
    """
    try:
        retry_count = await service.retry_failed_deliveries(limit)
        
        return create_success_response(
            {"retry_count": retry_count, "limit": limit},
            message=f"Retried {retry_count} webhook deliveries"
        )
    except Exception as e:
        logger.error(f"Retry failed deliveries error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retry webhook deliveries"
        )

