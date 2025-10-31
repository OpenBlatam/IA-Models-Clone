"""
Gamma App - Integration API Routes
Advanced integration endpoints for external services and APIs
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from fastapi import APIRouter, HTTPException, Depends, status, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
import httpx
import aiohttp
from sqlalchemy.orm import Session

from ..models.database import get_db
from ..services.advanced_ai_service import get_advanced_ai_service, AdvancedAIService
from ..services.advanced_analytics_service import get_advanced_analytics_service, AdvancedAnalyticsService
from ..services.workflow_automation_service import get_workflow_automation_service, WorkflowAutomationService
from ..utils.auth import get_current_user
from ..utils.config import get_settings

logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Pydantic models for request/response
class IntegrationConfig(BaseModel):
    """Integration configuration"""
    service_name: str = Field(..., description="Name of the external service")
    api_key: str = Field(..., description="API key for the service")
    base_url: str = Field(..., description="Base URL of the service")
    timeout: int = Field(30, description="Request timeout in seconds")
    retry_count: int = Field(3, description="Number of retries on failure")
    headers: Dict[str, str] = Field(default_factory=dict, description="Additional headers")

class WebhookConfig(BaseModel):
    """Webhook configuration"""
    url: str = Field(..., description="Webhook URL")
    events: List[str] = Field(..., description="Events to subscribe to")
    secret: Optional[str] = Field(None, description="Webhook secret for verification")
    timeout: int = Field(30, description="Request timeout in seconds")
    retry_count: int = Field(3, description="Number of retries on failure")

class DataSyncConfig(BaseModel):
    """Data synchronization configuration"""
    source_service: str = Field(..., description="Source service name")
    target_service: str = Field(..., description="Target service name")
    sync_interval: int = Field(3600, description="Sync interval in seconds")
    batch_size: int = Field(100, description="Batch size for sync operations")
    filters: Dict[str, Any] = Field(default_factory=dict, description="Sync filters")

class APIGatewayConfig(BaseModel):
    """API Gateway configuration"""
    upstream_url: str = Field(..., description="Upstream API URL")
    rate_limit: int = Field(100, description="Requests per minute")
    timeout: int = Field(30, description="Request timeout")
    retry_count: int = Field(3, description="Number of retries")
    circuit_breaker: bool = Field(True, description="Enable circuit breaker")
    authentication: Dict[str, Any] = Field(default_factory=dict, description="Authentication config")

# Integration endpoints
@router.post("/integrations/configure")
async def configure_integration(
    config: IntegrationConfig,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Configure an external service integration"""
    
    try:
        # Store integration configuration
        integration_data = {
            "service_name": config.service_name,
            "api_key": config.api_key,
            "base_url": config.base_url,
            "timeout": config.timeout,
            "retry_count": config.retry_count,
            "headers": config.headers,
            "created_by": current_user.get("user_id"),
            "created_at": datetime.now().isoformat()
        }
        
        # Test the integration
        test_result = await test_integration_connection(config)
        
        if not test_result["success"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Integration test failed: {test_result['error']}"
            )
        
        # Store in database (simplified)
        # In a real implementation, you would store this in a proper table
        
        return {
            "success": True,
            "message": f"Integration configured successfully for {config.service_name}",
            "integration_id": f"int_{config.service_name}_{int(datetime.now().timestamp())}",
            "test_result": test_result
        }
        
    except Exception as e:
        logger.error(f"Integration configuration failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.get("/integrations/test/{service_name}")
async def test_integration(
    service_name: str,
    current_user: dict = Depends(get_current_user)
):
    """Test an integration connection"""
    
    try:
        # Get integration configuration (simplified)
        config = await get_integration_config(service_name)
        
        if not config:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Integration not found: {service_name}"
            )
        
        test_result = await test_integration_connection(config)
        
        return {
            "service_name": service_name,
            "test_result": test_result,
            "tested_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Integration test failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.post("/webhooks/configure")
async def configure_webhook(
    config: WebhookConfig,
    current_user: dict = Depends(get_current_user)
):
    """Configure a webhook for external services"""
    
    try:
        # Validate webhook URL
        if not config.url.startswith(("http://", "https://")):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid webhook URL"
            )
        
        # Test webhook endpoint
        test_result = await test_webhook_endpoint(config.url, config.secret)
        
        if not test_result["success"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Webhook test failed: {test_result['error']}"
            )
        
        # Store webhook configuration
        webhook_data = {
            "url": config.url,
            "events": config.events,
            "secret": config.secret,
            "timeout": config.timeout,
            "retry_count": config.retry_count,
            "created_by": current_user.get("user_id"),
            "created_at": datetime.now().isoformat(),
            "active": True
        }
        
        return {
            "success": True,
            "message": "Webhook configured successfully",
            "webhook_id": f"wh_{int(datetime.now().timestamp())}",
            "test_result": test_result
        }
        
    except Exception as e:
        logger.error(f"Webhook configuration failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.post("/webhooks/trigger/{webhook_id}")
async def trigger_webhook(
    webhook_id: str,
    event_data: Dict[str, Any],
    current_user: dict = Depends(get_current_user)
):
    """Manually trigger a webhook"""
    
    try:
        # Get webhook configuration
        webhook_config = await get_webhook_config(webhook_id)
        
        if not webhook_config:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Webhook not found: {webhook_id}"
            )
        
        # Trigger webhook
        result = await send_webhook(webhook_config, event_data)
        
        return {
            "success": True,
            "webhook_id": webhook_id,
            "result": result,
            "triggered_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Webhook trigger failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.post("/sync/configure")
async def configure_data_sync(
    config: DataSyncConfig,
    current_user: dict = Depends(get_current_user)
):
    """Configure data synchronization between services"""
    
    try:
        # Validate services
        source_config = await get_integration_config(config.source_service)
        target_config = await get_integration_config(config.target_service)
        
        if not source_config or not target_config:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Source or target service not configured"
            )
        
        # Create sync job
        sync_job = {
            "source_service": config.source_service,
            "target_service": config.target_service,
            "sync_interval": config.sync_interval,
            "batch_size": config.batch_size,
            "filters": config.filters,
            "created_by": current_user.get("user_id"),
            "created_at": datetime.now().isoformat(),
            "status": "active"
        }
        
        # Start sync job in background
        sync_id = f"sync_{int(datetime.now().timestamp())}"
        asyncio.create_task(run_data_sync(sync_id, sync_job))
        
        return {
            "success": True,
            "message": "Data synchronization configured successfully",
            "sync_id": sync_id,
            "sync_job": sync_job
        }
        
    except Exception as e:
        logger.error(f"Data sync configuration failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.get("/sync/status/{sync_id}")
async def get_sync_status(
    sync_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get data synchronization status"""
    
    try:
        # Get sync status (simplified)
        sync_status = await get_sync_job_status(sync_id)
        
        if not sync_status:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Sync job not found: {sync_id}"
            )
        
        return {
            "sync_id": sync_id,
            "status": sync_status,
            "checked_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Sync status check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.post("/gateway/configure")
async def configure_api_gateway(
    config: APIGatewayConfig,
    current_user: dict = Depends(get_current_user)
):
    """Configure API Gateway for external services"""
    
    try:
        # Validate upstream URL
        if not config.upstream_url.startswith(("http://", "https://")):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid upstream URL"
            )
        
        # Test upstream connection
        test_result = await test_upstream_connection(config.upstream_url)
        
        if not test_result["success"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Upstream connection test failed: {test_result['error']}"
            )
        
        # Configure API Gateway
        gateway_config = {
            "upstream_url": config.upstream_url,
            "rate_limit": config.rate_limit,
            "timeout": config.timeout,
            "retry_count": config.retry_count,
            "circuit_breaker": config.circuit_breaker,
            "authentication": config.authentication,
            "created_by": current_user.get("user_id"),
            "created_at": datetime.now().isoformat(),
            "active": True
        }
        
        return {
            "success": True,
            "message": "API Gateway configured successfully",
            "gateway_id": f"gw_{int(datetime.now().timestamp())}",
            "test_result": test_result
        }
        
    except Exception as e:
        logger.error(f"API Gateway configuration failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.get("/gateway/proxy/{gateway_id}/{path:path}")
async def proxy_request(
    gateway_id: str,
    path: str,
    request_data: Dict[str, Any] = None,
    current_user: dict = Depends(get_current_user)
):
    """Proxy request through API Gateway"""
    
    try:
        # Get gateway configuration
        gateway_config = await get_gateway_config(gateway_id)
        
        if not gateway_config:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Gateway not found: {gateway_id}"
            )
        
        # Check rate limiting
        rate_limit_result = await check_rate_limit(gateway_id, current_user.get("user_id"))
        
        if not rate_limit_result["allowed"]:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded"
            )
        
        # Proxy request
        proxy_result = await proxy_http_request(
            gateway_config,
            path,
            request_data or {}
        )
        
        return proxy_result
        
    except Exception as e:
        logger.error(f"Proxy request failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.get("/integrations/list")
async def list_integrations(
    current_user: dict = Depends(get_current_user)
):
    """List all configured integrations"""
    
    try:
        # Get all integrations (simplified)
        integrations = await get_all_integrations()
        
        return {
            "success": True,
            "integrations": integrations,
            "count": len(integrations)
        }
        
    except Exception as e:
        logger.error(f"List integrations failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.get("/integrations/health")
async def check_integrations_health(
    current_user: dict = Depends(get_current_user)
):
    """Check health of all integrations"""
    
    try:
        integrations = await get_all_integrations()
        health_results = []
        
        for integration in integrations:
            try:
                test_result = await test_integration_connection(integration)
                health_results.append({
                    "service_name": integration["service_name"],
                    "status": "healthy" if test_result["success"] else "unhealthy",
                    "response_time": test_result.get("response_time", 0),
                    "error": test_result.get("error")
                })
            except Exception as e:
                health_results.append({
                    "service_name": integration["service_name"],
                    "status": "error",
                    "error": str(e)
                })
        
        return {
            "success": True,
            "health_check": health_results,
            "checked_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.post("/integrations/export")
async def export_integration_data(
    service_name: str,
    export_format: str = "json",
    current_user: dict = Depends(get_current_user)
):
    """Export data from an integration"""
    
    try:
        # Get integration configuration
        config = await get_integration_config(service_name)
        
        if not config:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Integration not found: {service_name}"
            )
        
        # Export data
        export_result = await export_service_data(config, export_format)
        
        return {
            "success": True,
            "service_name": service_name,
            "export_format": export_format,
            "export_result": export_result,
            "exported_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Data export failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.post("/integrations/import")
async def import_integration_data(
    service_name: str,
    data: Dict[str, Any],
    current_user: dict = Depends(get_current_user)
):
    """Import data to an integration"""
    
    try:
        # Get integration configuration
        config = await get_integration_config(service_name)
        
        if not config:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Integration not found: {service_name}"
            )
        
        # Import data
        import_result = await import_service_data(config, data)
        
        return {
            "success": True,
            "service_name": service_name,
            "import_result": import_result,
            "imported_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Data import failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

# Helper functions
async def test_integration_connection(config: IntegrationConfig) -> Dict[str, Any]:
    """Test integration connection"""
    
    try:
        async with httpx.AsyncClient(timeout=config.timeout) as client:
            headers = {
                "Authorization": f"Bearer {config.api_key}",
                **config.headers
            }
            
            start_time = datetime.now()
            response = await client.get(f"{config.base_url}/health", headers=headers)
            response_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "success": response.status_code < 400,
                "status_code": response.status_code,
                "response_time": response_time,
                "response_data": response.json() if response.headers.get("content-type", "").startswith("application/json") else response.text
            }
            
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "response_time": 0
        }

async def test_webhook_endpoint(url: str, secret: Optional[str] = None) -> Dict[str, Any]:
    """Test webhook endpoint"""
    
    try:
        test_data = {
            "event": "test",
            "timestamp": datetime.now().isoformat(),
            "data": {"test": True}
        }
        
        headers = {"Content-Type": "application/json"}
        if secret:
            headers["X-Webhook-Secret"] = secret
        
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(url, json=test_data, headers=headers)
            
            return {
                "success": response.status_code < 400,
                "status_code": response.status_code,
                "response": response.text
            }
            
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

async def test_upstream_connection(url: str) -> Dict[str, Any]:
    """Test upstream connection"""
    
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(f"{url}/health")
            
            return {
                "success": response.status_code < 400,
                "status_code": response.status_code,
                "response_time": response.elapsed.total_seconds()
            }
            
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

async def send_webhook(webhook_config: Dict[str, Any], event_data: Dict[str, Any]) -> Dict[str, Any]:
    """Send webhook"""
    
    try:
        headers = {"Content-Type": "application/json"}
        if webhook_config.get("secret"):
            headers["X-Webhook-Secret"] = webhook_config["secret"]
        
        async with httpx.AsyncClient(timeout=webhook_config.get("timeout", 30)) as client:
            response = await client.post(
                webhook_config["url"],
                json=event_data,
                headers=headers
            )
            
            return {
                "success": response.status_code < 400,
                "status_code": response.status_code,
                "response": response.text
            }
            
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

async def proxy_http_request(
    gateway_config: Dict[str, Any],
    path: str,
    request_data: Dict[str, Any]
) -> Dict[str, Any]:
    """Proxy HTTP request"""
    
    try:
        url = f"{gateway_config['upstream_url']}/{path}"
        
        headers = {}
        if gateway_config.get("authentication", {}).get("type") == "bearer":
            headers["Authorization"] = f"Bearer {gateway_config['authentication']['token']}"
        
        async with httpx.AsyncClient(timeout=gateway_config.get("timeout", 30)) as client:
            response = await client.post(url, json=request_data, headers=headers)
            
            return {
                "success": response.status_code < 400,
                "status_code": response.status_code,
                "data": response.json() if response.headers.get("content-type", "").startswith("application/json") else response.text
            }
            
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

async def check_rate_limit(gateway_id: str, user_id: str) -> Dict[str, Any]:
    """Check rate limit for gateway"""
    
    # Simplified rate limiting
    # In a real implementation, you would use Redis or similar
    return {"allowed": True, "remaining": 100}

async def run_data_sync(sync_id: str, sync_job: Dict[str, Any]):
    """Run data synchronization job"""
    
    try:
        logger.info(f"Starting data sync: {sync_id}")
        
        # Get source and target configurations
        source_config = await get_integration_config(sync_job["source_service"])
        target_config = await get_integration_config(sync_job["target_service"])
        
        # Perform sync (simplified)
        # In a real implementation, you would implement actual data sync logic
        
        logger.info(f"Data sync completed: {sync_id}")
        
    except Exception as e:
        logger.error(f"Data sync failed: {sync_id} - {e}")

# Simplified data access functions
async def get_integration_config(service_name: str) -> Optional[Dict[str, Any]]:
    """Get integration configuration"""
    # Simplified - in real implementation, this would query the database
    return {
        "service_name": service_name,
        "api_key": "test-key",
        "base_url": "https://api.example.com",
        "timeout": 30,
        "retry_count": 3,
        "headers": {}
    }

async def get_webhook_config(webhook_id: str) -> Optional[Dict[str, Any]]:
    """Get webhook configuration"""
    # Simplified - in real implementation, this would query the database
    return {
        "url": "https://webhook.example.com/endpoint",
        "events": ["content_created", "user_updated"],
        "secret": "webhook-secret",
        "timeout": 30,
        "retry_count": 3
    }

async def get_gateway_config(gateway_id: str) -> Optional[Dict[str, Any]]:
    """Get gateway configuration"""
    # Simplified - in real implementation, this would query the database
    return {
        "upstream_url": "https://api.example.com",
        "rate_limit": 100,
        "timeout": 30,
        "retry_count": 3,
        "circuit_breaker": True,
        "authentication": {"type": "bearer", "token": "test-token"}
    }

async def get_all_integrations() -> List[Dict[str, Any]]:
    """Get all integrations"""
    # Simplified - in real implementation, this would query the database
    return [
        {
            "service_name": "openai",
            "api_key": "sk-***",
            "base_url": "https://api.openai.com/v1",
            "timeout": 30,
            "retry_count": 3,
            "headers": {},
            "created_at": datetime.now().isoformat()
        },
        {
            "service_name": "anthropic",
            "api_key": "sk-ant-***",
            "base_url": "https://api.anthropic.com",
            "timeout": 30,
            "retry_count": 3,
            "headers": {},
            "created_at": datetime.now().isoformat()
        }
    ]

async def get_sync_job_status(sync_id: str) -> Optional[Dict[str, Any]]:
    """Get sync job status"""
    # Simplified - in real implementation, this would query the database
    return {
        "sync_id": sync_id,
        "status": "running",
        "last_sync": datetime.now().isoformat(),
        "records_synced": 150,
        "errors": 0
    }

async def export_service_data(config: Dict[str, Any], format: str) -> Dict[str, Any]:
    """Export data from service"""
    # Simplified implementation
    return {
        "format": format,
        "records_exported": 100,
        "file_path": f"/exports/{config['service_name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{format}"
    }

async def import_service_data(config: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
    """Import data to service"""
    # Simplified implementation
    return {
        "records_imported": len(data.get("records", [])),
        "status": "success"
    }



