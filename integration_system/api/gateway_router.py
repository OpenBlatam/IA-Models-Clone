"""
Gateway Router
==============

Unified API Gateway for routing requests to all integrated systems.
"""

from fastapi import APIRouter, HTTPException, Depends, Request, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
import logging
import uuid
from datetime import datetime

from ..core.integration_manager import IntegrationManager, SystemType, SystemStatus

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/gateway", tags=["Gateway"])

# Dependency to get integration manager
def get_integration_manager() -> IntegrationManager:
    """Get the global integration manager instance."""
    from ..main import app
    return app.state.integration_manager

# Request/Response Models
class GatewayRequest(BaseModel):
    target_system: str = Field(..., description="Target system to route request to")
    endpoint: str = Field(..., description="Endpoint path")
    method: str = Field("GET", description="HTTP method")
    data: Optional[Dict[str, Any]] = Field(None, description="Request data")
    headers: Optional[Dict[str, str]] = Field(None, description="Request headers")

class GatewayResponse(BaseModel):
    request_id: str
    target_system: str
    endpoint: str
    method: str
    status_code: int
    response: Dict[str, Any]
    processing_time: float
    timestamp: str

class SystemRouteRequest(BaseModel):
    source_system: str = Field(..., description="Source system")
    target_system: str = Field(..., description="Target system")
    operation: str = Field(..., description="Operation to perform")
    data: Dict[str, Any] = Field(..., description="Operation data")

class SystemRouteResponse(BaseModel):
    request_id: str
    source_system: str
    target_system: str
    operation: str
    status: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: str

# Endpoints
@router.post("/route", response_model=GatewayResponse)
async def route_request(
    request: GatewayRequest,
    background_tasks: BackgroundTasks,
    integration_manager: IntegrationManager = Depends(get_integration_manager)
):
    """Route a request to a specific system."""
    
    try:
        # Validate target system
        try:
            system_type = SystemType(request.target_system)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid target system: {request.target_system}. Available systems: {[s.value for s in SystemType]}"
            )
        
        # Check if system is available
        system_info = await integration_manager.get_system_status(system_type)
        if system_info.status == SystemStatus.OFFLINE:
            raise HTTPException(
                status_code=503,
                detail=f"System {system_info.name} is currently offline"
            )
        
        # Route the request
        start_time = datetime.now()
        
        response = await integration_manager.route_request(
            system_type=system_type,
            endpoint=request.endpoint,
            method=request.method,
            data=request.data,
            headers=request.headers
        )
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        # Generate request ID
        request_id = str(uuid.uuid4())
        
        return GatewayResponse(
            request_id=request_id,
            target_system=request.target_system,
            endpoint=request.endpoint,
            method=request.method,
            status_code=200,
            response=response,
            processing_time=processing_time,
            timestamp=end_time.isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Gateway routing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Gateway routing failed: {str(e)}")

@router.post("/systems/route", response_model=SystemRouteResponse)
async def route_system_request(
    request: SystemRouteRequest,
    background_tasks: BackgroundTasks,
    integration_manager: IntegrationManager = Depends(get_integration_manager)
):
    """Route a request between systems."""
    
    try:
        # Validate systems
        try:
            source_system = SystemType(request.source_system)
            target_system = SystemType(request.target_system)
        except ValueError as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid system type: {str(e)}. Available systems: {[s.value for s in SystemType]}"
            )
        
        # Create integration request
        integration_request = await integration_manager.create_integration_request(
            source_system=source_system,
            target_system=target_system,
            operation=request.operation,
            data=request.data
        )
        
        return SystemRouteResponse(
            request_id=integration_request.id,
            source_system=request.source_system,
            target_system=request.target_system,
            operation=request.operation,
            status=integration_request.status,
            result=integration_request.result,
            error=integration_request.error,
            created_at=integration_request.created_at.isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"System routing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"System routing failed: {str(e)}")

@router.get("/systems", response_model=Dict[str, Any])
async def get_available_systems(
    integration_manager: IntegrationManager = Depends(get_integration_manager)
):
    """Get all available systems and their status."""
    
    try:
        systems_status = await integration_manager.get_all_systems_status()
        
        systems_info = {}
        for system_type, system_info in systems_status.items():
            systems_info[system_type.value] = {
                "name": system_info.name,
                "status": system_info.status.value,
                "endpoint": system_info.endpoint,
                "description": system_info.description,
                "version": system_info.version,
                "capabilities": system_info.capabilities,
                "last_check": system_info.last_check.isoformat(),
                "response_time": system_info.response_time,
                "error_message": system_info.error_message
            }
        
        return {
            "systems": systems_info,
            "total_systems": len(systems_info),
            "healthy_systems": len([s for s in systems_status.values() if s.status == SystemStatus.HEALTHY]),
            "offline_systems": len([s for s in systems_status.values() if s.status == SystemStatus.OFFLINE])
        }
        
    except Exception as e:
        logger.error(f"Failed to get systems status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get systems status: {str(e)}")

@router.get("/systems/{system_name}", response_model=Dict[str, Any])
async def get_system_info(
    system_name: str,
    integration_manager: IntegrationManager = Depends(get_integration_manager)
):
    """Get detailed information about a specific system."""
    
    try:
        # Validate system type
        try:
            system_type = SystemType(system_name)
        except ValueError:
            raise HTTPException(
                status_code=404,
                detail=f"System {system_name} not found. Available systems: {[s.value for s in SystemType]}"
            )
        
        # Get system status
        system_info = await integration_manager.get_system_status(system_type)
        
        # Get system capabilities
        capabilities = await integration_manager.get_system_capabilities(system_type)
        
        # Get available operations
        operations = await integration_manager.get_available_operations()
        system_operations = operations.get(system_type, [])
        
        return {
            "name": system_info.name,
            "type": system_info.type.value,
            "status": system_info.status.value,
            "endpoint": system_info.endpoint,
            "description": system_info.description,
            "version": system_info.version,
            "capabilities": capabilities,
            "operations": system_operations,
            "last_check": system_info.last_check.isoformat(),
            "response_time": system_info.response_time,
            "error_message": system_info.error_message,
            "metadata": system_info.metadata
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get system info for {system_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get system info: {str(e)}")

@router.get("/operations", response_model=Dict[str, List[str]])
async def get_available_operations(
    integration_manager: IntegrationManager = Depends(get_integration_manager)
):
    """Get available operations for all systems."""
    
    try:
        operations = await integration_manager.get_available_operations()
        
        # Convert to string keys for JSON serialization
        operations_dict = {}
        for system_type, ops in operations.items():
            operations_dict[system_type.value] = ops
        
        return operations_dict
        
    except Exception as e:
        logger.error(f"Failed to get available operations: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get available operations: {str(e)}")

@router.get("/requests", response_model=List[Dict[str, Any]])
async def get_integration_requests(
    source_system: Optional[str] = None,
    target_system: Optional[str] = None,
    status: Optional[str] = None,
    integration_manager: IntegrationManager = Depends(get_integration_manager)
):
    """Get integration requests with optional filtering."""
    
    try:
        # Convert string parameters to SystemType if provided
        source_system_type = None
        target_system_type = None
        
        if source_system:
            try:
                source_system_type = SystemType(source_system)
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid source system: {source_system}"
                )
        
        if target_system:
            try:
                target_system_type = SystemType(target_system)
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid target system: {target_system}"
                )
        
        # Get requests
        requests = integration_manager.list_integration_requests(
            source_system=source_system_type,
            target_system=target_system_type,
            status=status
        )
        
        # Convert to response format
        requests_data = []
        for request in requests:
            requests_data.append({
                "id": request.id,
                "source_system": request.source_system.value,
                "target_system": request.target_system.value,
                "operation": request.operation,
                "status": request.status,
                "result": request.result,
                "error": request.error,
                "created_at": request.created_at.isoformat()
            })
        
        return requests_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get integration requests: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get integration requests: {str(e)}")

@router.get("/requests/{request_id}", response_model=Dict[str, Any])
async def get_integration_request(
    request_id: str,
    integration_manager: IntegrationManager = Depends(get_integration_manager)
):
    """Get a specific integration request by ID."""
    
    try:
        request = integration_manager.get_integration_request(request_id)
        if not request:
            raise HTTPException(status_code=404, detail=f"Integration request {request_id} not found")
        
        return {
            "id": request.id,
            "source_system": request.source_system.value,
            "target_system": request.target_system.value,
            "operation": request.operation,
            "data": request.data,
            "status": request.status,
            "result": request.result,
            "error": request.error,
            "created_at": request.created_at.isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get integration request {request_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get integration request: {str(e)}")

@router.get("/statistics", response_model=Dict[str, Any])
async def get_gateway_statistics(
    integration_manager: IntegrationManager = Depends(get_integration_manager)
):
    """Get gateway and integration statistics."""
    
    try:
        statistics = await integration_manager.get_integration_statistics()
        return statistics
        
    except Exception as e:
        logger.error(f"Failed to get gateway statistics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get gateway statistics: {str(e)}")

@router.post("/systems/{system_name}/health-check")
async def trigger_health_check(
    system_name: str,
    integration_manager: IntegrationManager = Depends(get_integration_manager)
):
    """Trigger a health check for a specific system."""
    
    try:
        # Validate system type
        try:
            system_type = SystemType(system_name)
        except ValueError:
            raise HTTPException(
                status_code=404,
                detail=f"System {system_name} not found. Available systems: {[s.value for s in SystemType]}"
            )
        
        # Perform health check
        system_info = await integration_manager.get_system_status(system_type)
        
        return {
            "system": system_name,
            "status": system_info.status.value,
            "response_time": system_info.response_time,
            "error_message": system_info.error_message,
            "last_check": system_info.last_check.isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to trigger health check for {system_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to trigger health check: {str(e)}")

@router.post("/systems/health-check-all")
async def trigger_all_health_checks(
    integration_manager: IntegrationManager = Depends(get_integration_manager)
):
    """Trigger health checks for all systems."""
    
    try:
        # Perform health checks
        systems_status = await integration_manager.get_all_systems_status()
        
        results = {}
        for system_type, system_info in systems_status.items():
            results[system_type.value] = {
                "status": system_info.status.value,
                "response_time": system_info.response_time,
                "error_message": system_info.error_message,
                "last_check": system_info.last_check.isoformat()
            }
        
        return {
            "message": "Health checks completed",
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to trigger all health checks: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to trigger health checks: {str(e)}")

