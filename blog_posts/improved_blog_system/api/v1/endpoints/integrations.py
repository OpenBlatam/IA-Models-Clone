"""
Advanced Integration API endpoints
"""

from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query, Request
from pydantic import BaseModel, Field

from ....services.advanced_integration_service import AdvancedIntegrationService
from ....api.dependencies import CurrentUserDep, DatabaseSessionDep
from ....core.exceptions import DatabaseError, ValidationError, ExternalServiceError

router = APIRouter()


class APICallRequest(BaseModel):
    """Request model for API calls."""
    integration_name: str = Field(..., description="Name of the integration")
    endpoint: str = Field(..., description="API endpoint")
    method: str = Field(default="GET", description="HTTP method")
    data: Optional[Dict[str, Any]] = Field(default=None, description="Request data")
    headers: Optional[Dict[str, str]] = Field(default=None, description="Additional headers")


class SocialMediaPostRequest(BaseModel):
    """Request model for social media posts."""
    platform: str = Field(..., description="Social media platform")
    content: str = Field(..., description="Post content")
    media_urls: Optional[List[str]] = Field(default=None, description="Media URLs")
    hashtags: Optional[List[str]] = Field(default=None, description="Hashtags")


class EmailRequest(BaseModel):
    """Request model for email sending."""
    service: str = Field(..., description="Email service")
    to_email: str = Field(..., description="Recipient email")
    subject: str = Field(..., description="Email subject")
    content: str = Field(..., description="Email content")
    from_email: Optional[str] = Field(default=None, description="Sender email")


class AnalyticsEventRequest(BaseModel):
    """Request model for analytics events."""
    service: str = Field(..., description="Analytics service")
    event_name: str = Field(..., description="Event name")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Event properties")
    user_id: Optional[str] = Field(default=None, description="User ID")


class PaymentRequest(BaseModel):
    """Request model for payments."""
    service: str = Field(..., description="Payment service")
    amount: float = Field(..., description="Payment amount")
    currency: str = Field(default="USD", description="Currency")
    description: str = Field(..., description="Payment description")
    customer_email: str = Field(..., description="Customer email")


class WebhookSetupRequest(BaseModel):
    """Request model for webhook setup."""
    integration_name: str = Field(..., description="Integration name")
    webhook_url: str = Field(..., description="Webhook URL")
    events: List[str] = Field(..., description="Events to listen for")


class IntegrationConfigRequest(BaseModel):
    """Request model for integration configuration."""
    integration_name: str = Field(..., description="Integration name")
    api_key: str = Field(..., description="API key")
    additional_config: Optional[Dict[str, Any]] = Field(default=None, description="Additional configuration")


async def get_integration_service(session: DatabaseSessionDep) -> AdvancedIntegrationService:
    """Get integration service instance."""
    return AdvancedIntegrationService(session)


@router.post("/api-call", response_model=Dict[str, Any])
async def make_api_call(
    request: APICallRequest = Depends(),
    integration_service: AdvancedIntegrationService = Depends(get_integration_service),
    current_user: CurrentUserDep = Depends()
):
    """Make an API call to an integrated service."""
    try:
        result = await integration_service.make_api_call(
            integration_name=request.integration_name,
            endpoint=request.endpoint,
            method=request.method,
            data=request.data,
            headers=request.headers
        )
        
        return {
            "success": True,
            "data": result,
            "message": "API call completed successfully"
        }
        
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except ExternalServiceError as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"External service error: {e.detail}"
        )
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to make API call"
        )


@router.post("/social-media/post", response_model=Dict[str, Any])
async def post_to_social_media(
    request: SocialMediaPostRequest = Depends(),
    integration_service: AdvancedIntegrationService = Depends(get_integration_service),
    current_user: CurrentUserDep = Depends()
):
    """Post content to social media platforms."""
    try:
        result = await integration_service.post_to_social_media(
            platform=request.platform,
            content=request.content,
            media_urls=request.media_urls,
            hashtags=request.hashtags
        )
        
        return {
            "success": True,
            "data": result,
            "message": f"Content posted to {request.platform} successfully"
        }
        
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to post to social media"
        )


@router.post("/email/send", response_model=Dict[str, Any])
async def send_email(
    request: EmailRequest = Depends(),
    integration_service: AdvancedIntegrationService = Depends(get_integration_service),
    current_user: CurrentUserDep = Depends()
):
    """Send email via integrated email service."""
    try:
        result = await integration_service.send_email_via_service(
            service=request.service,
            to_email=request.to_email,
            subject=request.subject,
            content=request.content,
            from_email=request.from_email
        )
        
        return {
            "success": True,
            "data": result,
            "message": f"Email sent via {request.service} successfully"
        }
        
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to send email"
        )


@router.post("/analytics/track", response_model=Dict[str, Any])
async def track_analytics_event(
    request: AnalyticsEventRequest = Depends(),
    integration_service: AdvancedIntegrationService = Depends(get_integration_service),
    current_user: CurrentUserDep = Depends()
):
    """Track analytics event via integrated service."""
    try:
        result = await integration_service.track_analytics_event(
            service=request.service,
            event_name=request.event_name,
            properties=request.properties,
            user_id=request.user_id
        )
        
        return {
            "success": True,
            "data": result,
            "message": f"Analytics event tracked via {request.service} successfully"
        }
        
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to track analytics event"
        )


@router.post("/payment/process", response_model=Dict[str, Any])
async def process_payment(
    request: PaymentRequest = Depends(),
    integration_service: AdvancedIntegrationService = Depends(get_integration_service),
    current_user: CurrentUserDep = Depends()
):
    """Process payment via integrated payment service."""
    try:
        result = await integration_service.process_payment(
            service=request.service,
            amount=request.amount,
            currency=request.currency,
            description=request.description,
            customer_email=request.customer_email
        )
        
        return {
            "success": True,
            "data": result,
            "message": f"Payment processed via {request.service} successfully"
        }
        
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process payment"
        )


@router.post("/webhooks/setup", response_model=Dict[str, Any])
async def setup_webhook(
    request: WebhookSetupRequest = Depends(),
    integration_service: AdvancedIntegrationService = Depends(get_integration_service),
    current_user: CurrentUserDep = Depends()
):
    """Setup webhook for an integration."""
    try:
        result = await integration_service.setup_webhook(
            integration_name=request.integration_name,
            webhook_url=request.webhook_url,
            events=request.events
        )
        
        return {
            "success": True,
            "data": result,
            "message": f"Webhook setup for {request.integration_name} completed successfully"
        }
        
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to setup webhook"
        )


@router.post("/webhooks/{integration_name}/handle", response_model=Dict[str, Any])
async def handle_webhook(
    integration_name: str,
    payload: Dict[str, Any],
    signature: Optional[str] = Query(None, description="Webhook signature"),
    integration_service: AdvancedIntegrationService = Depends(get_integration_service)
):
    """Handle incoming webhook."""
    try:
        result = await integration_service.handle_webhook(
            integration_name=integration_name,
            payload=payload,
            signature=signature
        )
        
        return {
            "success": True,
            "data": result,
            "message": f"Webhook handled for {integration_name} successfully"
        }
        
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to handle webhook"
        )


@router.get("/available", response_model=Dict[str, Any])
async def get_available_integrations(
    integration_service: AdvancedIntegrationService = Depends(get_integration_service),
    current_user: CurrentUserDep = Depends()
):
    """Get list of available integrations."""
    try:
        integrations = await integration_service.get_available_integrations()
        
        return {
            "success": True,
            "data": integrations,
            "message": "Available integrations retrieved successfully"
        }
        
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get available integrations"
        )


@router.get("/stats", response_model=Dict[str, Any])
async def get_integration_stats(
    integration_service: AdvancedIntegrationService = Depends(get_integration_service),
    current_user: CurrentUserDep = Depends()
):
    """Get integration statistics."""
    try:
        stats = await integration_service.get_integration_stats()
        
        return {
            "success": True,
            "data": stats,
            "message": "Integration statistics retrieved successfully"
        }
        
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get integration statistics"
        )


@router.post("/configure", response_model=Dict[str, Any])
async def configure_integration(
    request: IntegrationConfigRequest = Depends(),
    integration_service: AdvancedIntegrationService = Depends(get_integration_service),
    current_user: CurrentUserDep = Depends()
):
    """Configure an integration with API key and settings."""
    try:
        result = await integration_service.configure_integration(
            integration_name=request.integration_name,
            api_key=request.api_key,
            additional_config=request.additional_config
        )
        
        return {
            "success": True,
            "data": result,
            "message": f"Integration {request.integration_name} configured successfully"
        }
        
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to configure integration"
        )


@router.post("/test/{integration_name}", response_model=Dict[str, Any])
async def test_integration(
    integration_name: str,
    integration_service: AdvancedIntegrationService = Depends(get_integration_service),
    current_user: CurrentUserDep = Depends()
):
    """Test an integration by making a simple API call."""
    try:
        result = await integration_service.test_integration(integration_name)
        
        return {
            "success": True,
            "data": result,
            "message": f"Integration test for {integration_name} completed"
        }
        
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to test integration"
        )


@router.get("/types", response_model=Dict[str, Any])
async def get_integration_types():
    """Get available integration types and their descriptions."""
    integration_types = {
        "social_media": {
            "name": "Social Media",
            "description": "Integrations with social media platforms",
            "services": ["twitter", "facebook", "linkedin"],
            "capabilities": ["post_content", "schedule_posts", "analytics"]
        },
        "email": {
            "name": "Email Services",
            "description": "Email marketing and transactional email services",
            "services": ["sendgrid", "mailchimp"],
            "capabilities": ["send_emails", "manage_lists", "track_delivery"]
        },
        "analytics": {
            "name": "Analytics",
            "description": "Analytics and tracking services",
            "services": ["google_analytics", "mixpanel"],
            "capabilities": ["track_events", "generate_reports", "user_analytics"]
        },
        "payment": {
            "name": "Payment Processing",
            "description": "Payment processing services",
            "services": ["stripe", "paypal"],
            "capabilities": ["process_payments", "manage_subscriptions", "handle_refunds"]
        },
        "cdn": {
            "name": "Content Delivery Network",
            "description": "CDN and content delivery services",
            "services": ["cloudflare", "aws_cloudfront"],
            "capabilities": ["cache_content", "optimize_delivery", "manage_domains"]
        }
    }
    
    return {
        "success": True,
        "data": {
            "integration_types": integration_types,
            "total_types": len(integration_types)
        },
        "message": "Integration types retrieved successfully"
    }


@router.get("/health", response_model=Dict[str, Any])
async def get_integration_health(
    integration_service: AdvancedIntegrationService = Depends(get_integration_service),
    current_user: CurrentUserDep = Depends()
):
    """Get integration system health status."""
    try:
        # Get integration stats
        stats = await integration_service.get_integration_stats()
        
        # Calculate health metrics
        total_integrations = stats.get("total_integrations", 0)
        active_integrations = stats.get("active_integrations", 0)
        
        health_score = (active_integrations / max(total_integrations, 1)) * 100
        
        # Check API call success rates
        api_calls_stats = stats.get("api_calls_stats", {})
        overall_success_rate = 0
        if api_calls_stats:
            total_calls = sum(stat["total_calls"] for stat in api_calls_stats.values())
            successful_calls = sum(stat["successful_calls"] for stat in api_calls_stats.values())
            overall_success_rate = (successful_calls / max(total_calls, 1)) * 100
        
        health_status = "excellent" if health_score >= 90 and overall_success_rate >= 95 else "good" if health_score >= 70 and overall_success_rate >= 80 else "fair" if health_score >= 50 and overall_success_rate >= 60 else "poor"
        
        return {
            "success": True,
            "data": {
                "health_status": health_status,
                "health_score": health_score,
                "active_integrations": active_integrations,
                "total_integrations": total_integrations,
                "api_success_rate": overall_success_rate,
                "timestamp": "2024-01-15T10:00:00Z"
            },
            "message": "Integration health status retrieved successfully"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get integration health status"
        )

























