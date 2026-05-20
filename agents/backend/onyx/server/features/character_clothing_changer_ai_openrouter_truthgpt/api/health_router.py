"""
Health Check Router
===================

FastAPI router for health check endpoints.

Endpoints:
- GET /health - Basic health check
- GET /health/detailed - Detailed health check with service status
"""

import logging
from fastapi import APIRouter, Depends, HTTPException, status
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime

from services.clothing_service import ClothingChangeService
from services.health_service import get_health_service

logger = logging.getLogger(__name__)

router = APIRouter()

# Service version
SERVICE_VERSION = "1.0.0"
SERVICE_NAME = "character_clothing_changer_ai_openrouter_truthgpt"


# Response Models
class HealthResponse(BaseModel):
    """Basic health check response"""
    
    status: str = Field(..., description="Service status", example="healthy")
    service: str = Field(..., description="Service name", example=SERVICE_NAME)
    version: str = Field(..., description="Service version", example=SERVICE_VERSION)
    timestamp: str = Field(..., description="Current timestamp")


class DetailedHealthResponse(BaseModel):
    """Detailed health check response"""
    
    status: str = Field(..., description="Service status", example="healthy")
    service: str = Field(..., description="Service name", example=SERVICE_NAME)
    version: str = Field(..., description="Service version", example=SERVICE_VERSION)
    timestamp: str = Field(..., description="Current timestamp")
    services: Dict[str, Any] = Field(..., description="Service status and configuration")
    uptime: Optional[str] = Field(None, description="Service uptime")


# Dependency Injection
def get_clothing_service() -> ClothingChangeService:
    """
    Dependency to get ClothingChangeService instance.
    
    Returns:
        ClothingChangeService instance
    """
    return ClothingChangeService()


# Endpoints
@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Basic Health Check",
    description="Simple health check endpoint to verify service is running",
    tags=["health"]
)
async def health_check() -> HealthResponse:
    """
    Basic health check endpoint.
    
    Returns:
        HealthResponse with service status
        
    This endpoint provides a quick way to verify the service is running
    without checking external dependencies.
    """
    return HealthResponse(
        status="healthy",
        service=SERVICE_NAME,
        version=SERVICE_VERSION,
        timestamp=datetime.utcnow().isoformat()
    )


@router.get(
    "/health/detailed",
    response_model=DetailedHealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Detailed Health Check",
    description="Comprehensive health check including all service dependencies",
    tags=["health"],
    responses={
        200: {"description": "Service is healthy"},
        503: {"description": "Service is unhealthy"}
    }
)
async def detailed_health_check(
    service: ClothingChangeService = Depends(get_clothing_service)
) -> DetailedHealthResponse:
    """
    Detailed health check with service status.
    
    This endpoint checks:
    - Service availability
    - OpenRouter configuration
    - TruthGPT configuration
    - ComfyUI connectivity
    
    Args:
        service: ClothingChangeService instance (injected)
        
    Returns:
        DetailedHealthResponse with comprehensive status
        
    Raises:
        HTTPException: If critical services are unavailable
    """
    try:
        logger.debug("Performing detailed health check")
        
        # Get service analytics
        analytics = await service.get_analytics()
        
        # Determine overall health status
        health_status = "healthy"
        issues = []
        
        # Check ComfyUI (required)
        comfyui_url = analytics.get("comfyui_url")
        if not comfyui_url:
            health_status = "degraded"
            issues.append("ComfyUI URL not configured")
        
        # Check optional services
        if not analytics.get("openrouter_enabled"):
            issues.append("OpenRouter is disabled")
        
        if not analytics.get("truthgpt_enabled"):
            issues.append("TruthGPT is disabled")
        
        # Build response
        response_data = {
            "status": health_status,
            "service": SERVICE_NAME,
            "version": SERVICE_VERSION,
            "timestamp": datetime.utcnow().isoformat(),
            "services": analytics
        }
        
        if issues:
            response_data["issues"] = issues
            logger.warning(f"Health check found issues: {issues}")
        
        # Return 503 if critical services are down
        if health_status == "unhealthy":
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Service is unhealthy",
                headers={"X-Health-Status": "unhealthy"}
            )
        
        return DetailedHealthResponse(**response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in detailed health check: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Health check failed: {str(e)}"
        )


@router.get(
    "/health/components",
    status_code=status.HTTP_200_OK,
    summary="Component Health Check",
    description="Check health of individual components with response times",
    tags=["health"]
)
async def component_health_check(
    health_service = Depends(get_health_service)
) -> Dict[str, Any]:
    """
    Check health of all components.
    
    Args:
        health_service: HealthService instance (injected)
        
    Returns:
        Dictionary with component health status
    """
    try:
        return await health_service.check_all()
    except Exception as e:
        logger.error(f"Error in component health check: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Component health check failed: {str(e)}"
        )

