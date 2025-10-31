"""
System API Router
=================

API endpoints for system information and management.
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import List
import logging

from ..schemas.common_schemas import (
    HealthCheckResponse, SystemInfoResponse, MetricsResponse
)
from ..schemas.agent_schemas import BusinessAreaResponse
from ..core.dependencies import (
    get_health_service, get_system_info_service, get_metrics_service,
    get_agent_service
)
from ..services.health_service import HealthService
from ..services.system_info_service import SystemInfoService
from ..services.metrics_service import MetricsService
from ..services.agent_service import AgentService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/system", tags=["System"])

@router.get("/health", response_model=HealthCheckResponse)
async def health_check(
    health_service: HealthService = Depends(get_health_service)
):
    """Get system health status."""
    
    try:
        result = await health_service.get_health_status()
        return HealthCheckResponse(**result)
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=503, detail="Health check failed")

@router.get("/info", response_model=SystemInfoResponse)
async def system_info(
    system_info_service: SystemInfoService = Depends(get_system_info_service)
):
    """Get detailed system information."""
    
    try:
        result = await system_info_service.get_system_info()
        return SystemInfoResponse(**result)
        
    except Exception as e:
        logger.error(f"Failed to get system info: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get system info")

@router.get("/metrics", response_model=MetricsResponse)
async def system_metrics(
    metrics_service: MetricsService = Depends(get_metrics_service)
):
    """Get system metrics."""
    
    try:
        result = await metrics_service.get_system_metrics()
        return MetricsResponse(**result)
        
    except Exception as e:
        logger.error(f"Failed to get system metrics: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get system metrics")

@router.get("/business-areas", response_model=List[BusinessAreaResponse])
async def get_business_areas(
    agent_service: AgentService = Depends(get_agent_service)
):
    """Get all available business areas."""
    
    try:
        # This would need to be implemented in the agent service
        # For now, return a placeholder response
        business_areas = [
            BusinessAreaResponse(
                value="marketing",
                name="Marketing",
                agents_count=1,
                description="Marketing strategy, campaigns, and brand management"
            ),
            BusinessAreaResponse(
                value="sales",
                name="Sales", 
                agents_count=1,
                description="Sales processes, lead generation, and customer acquisition"
            ),
            BusinessAreaResponse(
                value="operations",
                name="Operations",
                agents_count=1,
                description="Business operations and process optimization"
            ),
            BusinessAreaResponse(
                value="hr",
                name="Human Resources",
                agents_count=1,
                description="Human resources and employee lifecycle management"
            ),
            BusinessAreaResponse(
                value="finance",
                name="Finance",
                agents_count=1,
                description="Financial planning, analysis, and reporting"
            ),
            BusinessAreaResponse(
                value="legal",
                name="Legal",
                agents_count=1,
                description="Legal documents, compliance, and contract review"
            ),
            BusinessAreaResponse(
                value="technical",
                name="Technical",
                agents_count=1,
                description="Technical documentation and system analysis"
            ),
            BusinessAreaResponse(
                value="content",
                name="Content",
                agents_count=1,
                description="Content creation and management across platforms"
            )
        ]
        
        return business_areas
        
    except Exception as e:
        logger.error(f"Failed to get business areas: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get business areas")
