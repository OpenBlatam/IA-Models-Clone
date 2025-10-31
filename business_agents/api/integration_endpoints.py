"""
Integration API Endpoints
=========================

REST API endpoints for external system integrations.
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime

from ..services.integration_service import (
    IntegrationService,
    IntegrationType,
    AuthenticationType,
    IntegrationConfig,
    IntegrationRequest,
    IntegrationResponse,
    WebhookConfig
)
from ..middleware.auth_middleware import get_current_user, require_permission, User

# Create API router
router = APIRouter(prefix="/integrations", tags=["Integrations"])

# Pydantic models
class IntegrationConfigModel(BaseModel):
    id: str = Field(..., description="Integration ID")
    name: str = Field(..., description="Integration name")
    type: IntegrationType = Field(..., description="Integration type")
    base_url: str = Field(..., description="Base URL for the integration")
    authentication: AuthenticationType = Field(..., description="Authentication type")
    credentials: Dict[str, Any] = Field(..., description="Authentication credentials")
    headers: Dict[str, str] = Field(default_factory=dict, description="Default headers")
    timeout: int = Field(30, description="Request timeout in seconds")
    retry_count: int = Field(3, description="Number of retries")
    retry_delay: int = Field(1, description="Delay between retries in seconds")
    rate_limit: Optional[int] = Field(None, description="Rate limit per window")
    rate_limit_window: int = Field(60, description="Rate limit window in seconds")
    is_active: bool = Field(True, description="Whether integration is active")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

class IntegrationRequestModel(BaseModel):
    integration_id: str = Field(..., description="Integration ID to use")
    endpoint: str = Field(..., description="API endpoint")
    method: str = Field("GET", description="HTTP method")
    headers: Optional[Dict[str, str]] = Field(None, description="Request headers")
    params: Optional[Dict[str, Any]] = Field(None, description="Query parameters")
    data: Optional[Dict[str, Any]] = Field(None, description="Form data")
    json_data: Optional[Dict[str, Any]] = Field(None, description="JSON data")
    timeout: Optional[int] = Field(None, description="Request timeout")
    retry_count: Optional[int] = Field(None, description="Number of retries")

class WebhookConfigModel(BaseModel):
    id: str = Field(..., description="Webhook ID")
    name: str = Field(..., description="Webhook name")
    url: str = Field(..., description="Webhook URL")
    events: List[str] = Field(..., description="Events to trigger webhook")
    headers: Optional[Dict[str, str]] = Field(None, description="Webhook headers")
    authentication: AuthenticationType = Field(AuthenticationType.NONE, description="Authentication type")
    credentials: Optional[Dict[str, Any]] = Field(None, description="Authentication credentials")
    is_active: bool = Field(True, description="Whether webhook is active")
    retry_count: int = Field(3, description="Number of retries")
    timeout: int = Field(30, description="Request timeout")

class NotificationRequestModel(BaseModel):
    integration_type: IntegrationType = Field(..., description="Integration type for notification")
    message: str = Field(..., description="Notification message")
    recipients: Optional[List[str]] = Field(None, description="Recipients")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

class IntegrationResponseModel(BaseModel):
    status_code: int
    headers: Dict[str, str]
    data: Any
    success: bool
    error_message: Optional[str] = None
    response_time: float = 0.0
    retry_count: int = 0

# Global integration service instance
integration_service = None

def get_integration_service() -> IntegrationService:
    """Get global integration service instance."""
    global integration_service
    if integration_service is None:
        integration_service = IntegrationService({})
    return integration_service

# API Endpoints

@router.post("/", response_model=Dict[str, str])
async def create_integration(
    config: IntegrationConfigModel,
    current_user: User = Depends(require_permission("integrations:create"))
):
    """Create a new integration configuration."""
    
    integration_service = get_integration_service()
    
    try:
        # Convert to service model
        integration_config = IntegrationConfig(
            id=config.id,
            name=config.name,
            type=config.type,
            base_url=config.base_url,
            authentication=config.authentication,
            credentials=config.credentials,
            headers=config.headers,
            timeout=config.timeout,
            retry_count=config.retry_count,
            retry_delay=config.retry_delay,
            rate_limit=config.rate_limit,
            rate_limit_window=config.rate_limit_window,
            is_active=config.is_active,
            metadata=config.metadata
        )
        
        # Create integration
        success = await integration_service.create_integration(integration_config)
        
        if success:
            return {"message": f"Integration {config.id} created successfully"}
        else:
            raise HTTPException(status_code=400, detail="Failed to create integration")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Integration creation failed: {str(e)}")

@router.get("/", response_model=List[Dict[str, Any]])
async def list_integrations(
    integration_type: Optional[IntegrationType] = None,
    current_user: User = Depends(require_permission("integrations:view"))
):
    """List integration configurations."""
    
    integration_service = get_integration_service()
    
    integrations = integration_service.list_integrations(integration_type)
    
    return [
        {
            "id": integration.id,
            "name": integration.name,
            "type": integration.type.value,
            "base_url": integration.base_url,
            "authentication": integration.authentication.value,
            "is_active": integration.is_active,
            "timeout": integration.timeout,
            "retry_count": integration.retry_count,
            "rate_limit": integration.rate_limit,
            "metadata": integration.metadata
        }
        for integration in integrations
    ]

@router.get("/{integration_id}", response_model=Dict[str, Any])
async def get_integration(
    integration_id: str,
    current_user: User = Depends(require_permission("integrations:view"))
):
    """Get integration configuration by ID."""
    
    integration_service = get_integration_service()
    
    integration = integration_service.get_integration(integration_id)
    if not integration:
        raise HTTPException(status_code=404, detail=f"Integration {integration_id} not found")
    
    return {
        "id": integration.id,
        "name": integration.name,
        "type": integration.type.value,
        "base_url": integration.base_url,
        "authentication": integration.authentication.value,
        "credentials": integration.credentials,
        "headers": integration.headers,
        "timeout": integration.timeout,
        "retry_count": integration.retry_count,
        "retry_delay": integration.retry_delay,
        "rate_limit": integration.rate_limit,
        "rate_limit_window": integration.rate_limit_window,
        "is_active": integration.is_active,
        "metadata": integration.metadata
    }

@router.put("/{integration_id}", response_model=Dict[str, str])
async def update_integration(
    integration_id: str,
    config: IntegrationConfigModel,
    current_user: User = Depends(require_permission("integrations:update"))
):
    """Update integration configuration."""
    
    integration_service = get_integration_service()
    
    try:
        # Check if integration exists
        existing = integration_service.get_integration(integration_id)
        if not existing:
            raise HTTPException(status_code=404, detail=f"Integration {integration_id} not found")
        
        # Convert to service model
        integration_config = IntegrationConfig(
            id=config.id,
            name=config.name,
            type=config.type,
            base_url=config.base_url,
            authentication=config.authentication,
            credentials=config.credentials,
            headers=config.headers,
            timeout=config.timeout,
            retry_count=config.retry_count,
            retry_delay=config.retry_delay,
            rate_limit=config.rate_limit,
            rate_limit_window=config.rate_limit_window,
            is_active=config.is_active,
            metadata=config.metadata
        )
        
        # Update integration
        integration_service.integrations[integration_id] = integration_config
        
        return {"message": f"Integration {integration_id} updated successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Integration update failed: {str(e)}")

@router.delete("/{integration_id}", response_model=Dict[str, str])
async def delete_integration(
    integration_id: str,
    current_user: User = Depends(require_permission("integrations:delete"))
):
    """Delete integration configuration."""
    
    integration_service = get_integration_service()
    
    if integration_id not in integration_service.integrations:
        raise HTTPException(status_code=404, detail=f"Integration {integration_id} not found")
    
    # Close connection pool if exists
    if integration_id in integration_service.connection_pools:
        await integration_service.connection_pools[integration_id].aclose()
        del integration_service.connection_pools[integration_id]
    
    # Remove integration
    del integration_service.integrations[integration_id]
    
    return {"message": f"Integration {integration_id} deleted successfully"}

@router.post("/{integration_id}/request", response_model=IntegrationResponseModel)
async def make_integration_request(
    integration_id: str,
    request: IntegrationRequestModel,
    current_user: User = Depends(require_permission("integrations:execute"))
):
    """Make a request to an integration."""
    
    integration_service = get_integration_service()
    
    try:
        # Convert to service model
        integration_request = IntegrationRequest(
            integration_id=request.integration_id,
            endpoint=request.endpoint,
            method=request.method,
            headers=request.headers,
            params=request.params,
            data=request.data,
            json_data=request.json_data,
            timeout=request.timeout,
            retry_count=request.retry_count
        )
        
        # Make request
        response = await integration_service.make_request(integration_request)
        
        return IntegrationResponseModel(
            status_code=response.status_code,
            headers=response.headers,
            data=response.data,
            success=response.success,
            error_message=response.error_message,
            response_time=response.response_time,
            retry_count=response.retry_count
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Integration request failed: {str(e)}")

@router.get("/{integration_id}/test", response_model=Dict[str, Any])
async def test_integration(
    integration_id: str,
    current_user: User = Depends(require_permission("integrations:test"))
):
    """Test integration connectivity and configuration."""
    
    integration_service = get_integration_service()
    
    try:
        result = await integration_service.test_integration(integration_id)
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Integration test failed: {str(e)}")

@router.post("/webhooks/", response_model=Dict[str, str])
async def create_webhook(
    config: WebhookConfigModel,
    current_user: User = Depends(require_permission("integrations:create"))
):
    """Create a new webhook configuration."""
    
    integration_service = get_integration_service()
    
    try:
        # Convert to service model
        webhook_config = WebhookConfig(
            id=config.id,
            name=config.name,
            url=config.url,
            events=config.events,
            headers=config.headers,
            authentication=config.authentication,
            credentials=config.credentials,
            is_active=config.is_active,
            retry_count=config.retry_count,
            timeout=config.timeout
        )
        
        # Create webhook
        success = await integration_service.create_webhook(webhook_config)
        
        if success:
            return {"message": f"Webhook {config.id} created successfully"}
        else:
            raise HTTPException(status_code=400, detail="Failed to create webhook")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Webhook creation failed: {str(e)}")

@router.get("/webhooks/", response_model=List[Dict[str, Any]])
async def list_webhooks(
    current_user: User = Depends(require_permission("integrations:view"))
):
    """List webhook configurations."""
    
    integration_service = get_integration_service()
    
    webhooks = integration_service.list_webhooks()
    
    return [
        {
            "id": webhook.id,
            "name": webhook.name,
            "url": webhook.url,
            "events": webhook.events,
            "authentication": webhook.authentication.value,
            "is_active": webhook.is_active,
            "retry_count": webhook.retry_count,
            "timeout": webhook.timeout
        }
        for webhook in webhooks
    ]

@router.post("/webhooks/{webhook_id}/send", response_model=IntegrationResponseModel)
async def send_webhook(
    webhook_id: str,
    payload: Dict[str, Any],
    current_user: User = Depends(require_permission("integrations:execute"))
):
    """Send webhook notification."""
    
    integration_service = get_integration_service()
    
    try:
        response = await integration_service.send_webhook(webhook_id, payload)
        
        return IntegrationResponseModel(
            status_code=response.status_code,
            headers=response.headers,
            data=response.data,
            success=response.success,
            error_message=response.error_message,
            response_time=response.response_time,
            retry_count=response.retry_count
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Webhook send failed: {str(e)}")

@router.post("/notifications/send", response_model=IntegrationResponseModel)
async def send_notification(
    request: NotificationRequestModel,
    current_user: User = Depends(require_permission("integrations:execute"))
):
    """Send notification through specified integration type."""
    
    integration_service = get_integration_service()
    
    try:
        response = await integration_service.send_notification(
            integration_type=request.integration_type,
            message=request.message,
            recipients=request.recipients,
            metadata=request.metadata
        )
        
        return IntegrationResponseModel(
            status_code=response.status_code,
            headers=response.headers,
            data=response.data,
            success=response.success,
            error_message=response.error_message,
            response_time=response.response_time,
            retry_count=response.retry_count
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Notification send failed: {str(e)}")

@router.get("/types", response_model=List[Dict[str, str]])
async def get_integration_types():
    """Get available integration types."""
    
    integration_types = [
        {
            "type": integration_type.value,
            "name": integration_type.value.replace("_", " ").title(),
            "description": get_integration_description(integration_type)
        }
        for integration_type in IntegrationType
    ]
    
    return integration_types

def get_integration_description(integration_type: IntegrationType) -> str:
    """Get description for integration type."""
    
    descriptions = {
        IntegrationType.REST_API: "REST API integration for external services",
        IntegrationType.WEBHOOK: "Webhook integration for real-time notifications",
        IntegrationType.DATABASE: "Database integration for data storage",
        IntegrationType.EMAIL: "Email service integration",
        IntegrationType.SMS: "SMS service integration",
        IntegrationType.SLACK: "Slack workspace integration",
        IntegrationType.TEAMS: "Microsoft Teams integration",
        IntegrationType.CRM: "Customer Relationship Management integration",
        IntegrationType.ERP: "Enterprise Resource Planning integration",
        IntegrationType.ANALYTICS: "Analytics service integration",
        IntegrationType.PAYMENT: "Payment gateway integration",
        IntegrationType.FILE_STORAGE: "File storage service integration"
    }
    
    return descriptions.get(integration_type, "External system integration")

@router.get("/auth-types", response_model=List[Dict[str, str]])
async def get_authentication_types():
    """Get available authentication types."""
    
    auth_types = [
        {
            "type": auth_type.value,
            "name": auth_type.value.replace("_", " ").title(),
            "description": get_auth_description(auth_type)
        }
        for auth_type in AuthenticationType
    ]
    
    return auth_types

def get_auth_description(auth_type: AuthenticationType) -> str:
    """Get description for authentication type."""
    
    descriptions = {
        AuthenticationType.NONE: "No authentication required",
        AuthenticationType.API_KEY: "API key authentication",
        AuthenticationType.BEARER_TOKEN: "Bearer token authentication",
        AuthenticationType.BASIC_AUTH: "Basic username/password authentication",
        AuthenticationType.OAUTH2: "OAuth 2.0 authentication",
        AuthenticationType.JWT: "JSON Web Token authentication",
        AuthenticationType.CUSTOM: "Custom authentication method"
    }
    
    return descriptions.get(auth_type, "Authentication method")

@router.get("/health")
async def integration_health_check():
    """Health check for integration service."""
    
    integration_service = get_integration_service()
    
    try:
        # Test service availability
        integrations_count = len(integration_service.integrations)
        webhooks_count = len(integration_service.webhooks)
        
        return {
            "status": "healthy",
            "service": "Integration Service",
            "timestamp": datetime.now().isoformat(),
            "integrations_count": integrations_count,
            "webhooks_count": webhooks_count,
            "active_integrations": len([i for i in integration_service.integrations.values() if i.is_active])
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "service": "Integration Service",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }





























