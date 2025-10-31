"""
Holographic Computing API Endpoints
===================================

API endpoints for holographic computing service.
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import Dict, List, Any, Optional, Tuple
from pydantic import BaseModel, Field
from datetime import datetime
import logging

from ..services.holographic_computing_service import (
    HolographicComputingService,
    HolographicData,
    HolographicDisplay,
    HolographicProjection,
    HolographicProcessing,
    HolographicType,
    HolographicAlgorithm,
    SpatialDimension
)

logger = logging.getLogger(__name__)

# Create router
holographic_router = APIRouter(prefix="/holographic", tags=["Holographic Computing"])

# Pydantic models for request/response
class HolographicDataRequest(BaseModel):
    name: str
    data_type: str
    dimensions: Tuple[int, ...]
    resolution: Tuple[int, ...]
    data_array: List[List[float]]
    holographic_properties: Dict[str, Any] = {}
    metadata: Dict[str, Any] = {}

class HolographicDisplayRequest(BaseModel):
    name: str
    display_type: HolographicType
    resolution: Tuple[int, int, int]
    field_of_view: float = 60.0
    refresh_rate: float = 60.0
    color_depth: int = 24
    brightness: float = 100.0
    contrast: float = 50.0
    metadata: Dict[str, Any] = {}

class HolographicProjectionRequest(BaseModel):
    name: str
    source_data: str
    target_display: str
    projection_type: str
    parameters: Dict[str, Any] = {}
    metadata: Dict[str, Any] = {}

class HolographicProcessingRequest(BaseModel):
    name: str
    algorithm: HolographicAlgorithm
    input_data: str
    output_data: str
    parameters: Dict[str, Any] = {}
    metadata: Dict[str, Any] = {}

class HolographicDataResponse(BaseModel):
    data_id: str
    name: str
    data_type: str
    dimensions: Tuple[int, ...]
    resolution: Tuple[int, ...]
    holographic_properties: Dict[str, Any]
    created_at: datetime
    metadata: Dict[str, Any]

class HolographicDisplayResponse(BaseModel):
    display_id: str
    name: str
    display_type: str
    resolution: Tuple[int, int, int]
    field_of_view: float
    refresh_rate: float
    color_depth: int
    brightness: float
    contrast: float
    status: str
    created_at: datetime
    metadata: Dict[str, Any]

class HolographicProjectionResponse(BaseModel):
    projection_id: str
    name: str
    source_data: str
    target_display: str
    projection_type: str
    parameters: Dict[str, Any]
    quality: float
    status: str
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    metadata: Dict[str, Any]

class HolographicProcessingResponse(BaseModel):
    processing_id: str
    name: str
    algorithm: str
    input_data: str
    output_data: str
    parameters: Dict[str, Any]
    processing_time: float
    quality_metrics: Dict[str, float]
    status: str
    created_at: datetime
    completed_at: Optional[datetime]
    metadata: Dict[str, Any]

class ServiceStatusResponse(BaseModel):
    service_status: str
    total_data: int
    total_displays: int
    total_projections: int
    total_processing: int
    active_displays: int
    running_projections: int
    running_processing: int
    spatial_algorithms: int
    holographic_engines: int
    holographic_ai_enabled: bool
    spatial_computing_enabled: bool
    real_time_processing: bool
    max_data_objects: int
    max_displays: int
    timestamp: str

# Dependency to get holographic computing service
async def get_holographic_service() -> HolographicComputingService:
    """Get holographic computing service instance."""
    # This would be injected from your dependency injection system
    # For now, we'll create a mock instance
    from ..main import get_holographic_computing_service
    return await get_holographic_computing_service()

@holographic_router.post("/data", response_model=Dict[str, str])
async def create_holographic_data(
    request: HolographicDataRequest,
    holographic_service: HolographicComputingService = Depends(get_holographic_service)
):
    """Create holographic data."""
    try:
        import numpy as np
        
        data = HolographicData(
            data_id="",
            name=request.name,
            data_type=request.data_type,
            dimensions=request.dimensions,
            resolution=request.resolution,
            data_array=np.array(request.data_array),
            holographic_properties=request.holographic_properties,
            created_at=datetime.utcnow(),
            metadata=request.metadata
        )
        
        data_id = await holographic_service.create_holographic_data(data)
        
        return {"data_id": data_id, "status": "created"}
        
    except Exception as e:
        logger.error(f"Failed to create holographic data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@holographic_router.get("/data/{data_id}", response_model=HolographicDataResponse)
async def get_holographic_data(
    data_id: str,
    holographic_service: HolographicComputingService = Depends(get_holographic_service)
):
    """Get holographic data."""
    try:
        data = await holographic_service.get_holographic_data(data_id)
        
        if not data:
            raise HTTPException(status_code=404, detail="Holographic data not found")
            
        return HolographicDataResponse(
            data_id=data.data_id,
            name=data.name,
            data_type=data.data_type,
            dimensions=data.dimensions,
            resolution=data.resolution,
            holographic_properties=data.holographic_properties,
            created_at=data.created_at,
            metadata=data.metadata
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get holographic data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@holographic_router.get("/data", response_model=List[HolographicDataResponse])
async def list_holographic_data(
    data_type: Optional[str] = None,
    limit: int = 100,
    holographic_service: HolographicComputingService = Depends(get_holographic_service)
):
    """List holographic data."""
    try:
        data_list = await holographic_service.list_holographic_data(data_type)
        
        return [
            HolographicDataResponse(
                data_id=data.data_id,
                name=data.name,
                data_type=data.data_type,
                dimensions=data.dimensions,
                resolution=data.resolution,
                holographic_properties=data.holographic_properties,
                created_at=data.created_at,
                metadata=data.metadata
            )
            for data in data_list[:limit]
        ]
        
    except Exception as e:
        logger.error(f"Failed to list holographic data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@holographic_router.post("/displays", response_model=Dict[str, str])
async def create_holographic_display(
    request: HolographicDisplayRequest,
    holographic_service: HolographicComputingService = Depends(get_holographic_service)
):
    """Create holographic display."""
    try:
        display = HolographicDisplay(
            display_id="",
            name=request.name,
            display_type=request.display_type,
            resolution=request.resolution,
            field_of_view=request.field_of_view,
            refresh_rate=request.refresh_rate,
            color_depth=request.color_depth,
            brightness=request.brightness,
            contrast=request.contrast,
            status="active",
            created_at=datetime.utcnow(),
            metadata=request.metadata
        )
        
        display_id = await holographic_service.create_holographic_display(display)
        
        return {"display_id": display_id, "status": "created"}
        
    except Exception as e:
        logger.error(f"Failed to create holographic display: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@holographic_router.get("/displays/{display_id}", response_model=HolographicDisplayResponse)
async def get_holographic_display(
    display_id: str,
    holographic_service: HolographicComputingService = Depends(get_holographic_service)
):
    """Get holographic display."""
    try:
        display = await holographic_service.get_holographic_display(display_id)
        
        if not display:
            raise HTTPException(status_code=404, detail="Holographic display not found")
            
        return HolographicDisplayResponse(
            display_id=display.display_id,
            name=display.name,
            display_type=display.display_type.value,
            resolution=display.resolution,
            field_of_view=display.field_of_view,
            refresh_rate=display.refresh_rate,
            color_depth=display.color_depth,
            brightness=display.brightness,
            contrast=display.contrast,
            status=display.status,
            created_at=display.created_at,
            metadata=display.metadata
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get holographic display: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@holographic_router.get("/displays", response_model=List[HolographicDisplayResponse])
async def list_holographic_displays(
    status: Optional[str] = None,
    limit: int = 100,
    holographic_service: HolographicComputingService = Depends(get_holographic_service)
):
    """List holographic displays."""
    try:
        displays = await holographic_service.list_holographic_displays(status)
        
        return [
            HolographicDisplayResponse(
                display_id=display.display_id,
                name=display.name,
                display_type=display.display_type.value,
                resolution=display.resolution,
                field_of_view=display.field_of_view,
                refresh_rate=display.refresh_rate,
                color_depth=display.color_depth,
                brightness=display.brightness,
                contrast=display.contrast,
                status=display.status,
                created_at=display.created_at,
                metadata=display.metadata
            )
            for display in displays[:limit]
        ]
        
    except Exception as e:
        logger.error(f"Failed to list holographic displays: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@holographic_router.post("/projections", response_model=Dict[str, str])
async def create_holographic_projection(
    request: HolographicProjectionRequest,
    background_tasks: BackgroundTasks,
    holographic_service: HolographicComputingService = Depends(get_holographic_service)
):
    """Create holographic projection."""
    try:
        projection = HolographicProjection(
            projection_id="",
            name=request.name,
            source_data=request.source_data,
            target_display=request.target_display,
            projection_type=request.projection_type,
            parameters=request.parameters,
            quality=0.0,
            status="pending",
            created_at=datetime.utcnow(),
            started_at=None,
            completed_at=None,
            metadata=request.metadata
        )
        
        projection_id = await holographic_service.create_holographic_projection(projection)
        
        return {"projection_id": projection_id, "status": "created"}
        
    except Exception as e:
        logger.error(f"Failed to create holographic projection: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@holographic_router.get("/projections/{projection_id}", response_model=HolographicProjectionResponse)
async def get_holographic_projection(
    projection_id: str,
    holographic_service: HolographicComputingService = Depends(get_holographic_service)
):
    """Get holographic projection."""
    try:
        projection = await holographic_service.get_holographic_projection(projection_id)
        
        if not projection:
            raise HTTPException(status_code=404, detail="Holographic projection not found")
            
        return HolographicProjectionResponse(
            projection_id=projection.projection_id,
            name=projection.name,
            source_data=projection.source_data,
            target_display=projection.target_display,
            projection_type=projection.projection_type,
            parameters=projection.parameters,
            quality=projection.quality,
            status=projection.status,
            created_at=projection.created_at,
            started_at=projection.started_at,
            completed_at=projection.completed_at,
            metadata=projection.metadata
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get holographic projection: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@holographic_router.get("/projections", response_model=List[HolographicProjectionResponse])
async def list_holographic_projections(
    status: Optional[str] = None,
    limit: int = 100,
    holographic_service: HolographicComputingService = Depends(get_holographic_service)
):
    """List holographic projections."""
    try:
        projections = await holographic_service.list_holographic_projections(status)
        
        return [
            HolographicProjectionResponse(
                projection_id=projection.projection_id,
                name=projection.name,
                source_data=projection.source_data,
                target_display=projection.target_display,
                projection_type=projection.projection_type,
                parameters=projection.parameters,
                quality=projection.quality,
                status=projection.status,
                created_at=projection.created_at,
                started_at=projection.started_at,
                completed_at=projection.completed_at,
                metadata=projection.metadata
            )
            for projection in projections[:limit]
        ]
        
    except Exception as e:
        logger.error(f"Failed to list holographic projections: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@holographic_router.post("/processing", response_model=Dict[str, str])
async def process_holographic_data(
    request: HolographicProcessingRequest,
    background_tasks: BackgroundTasks,
    holographic_service: HolographicComputingService = Depends(get_holographic_service)
):
    """Process holographic data."""
    try:
        processing = HolographicProcessing(
            processing_id="",
            name=request.name,
            algorithm=request.algorithm,
            input_data=request.input_data,
            output_data=request.output_data,
            parameters=request.parameters,
            processing_time=0.0,
            quality_metrics={},
            status="pending",
            created_at=datetime.utcnow(),
            completed_at=None,
            metadata=request.metadata
        )
        
        processing_id = await holographic_service.process_holographic_data(processing)
        
        return {"processing_id": processing_id, "status": "started"}
        
    except Exception as e:
        logger.error(f"Failed to process holographic data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@holographic_router.get("/processing/{processing_id}", response_model=HolographicProcessingResponse)
async def get_holographic_processing(
    processing_id: str,
    holographic_service: HolographicComputingService = Depends(get_holographic_service)
):
    """Get holographic processing result."""
    try:
        processing = await holographic_service.get_holographic_processing(processing_id)
        
        if not processing:
            raise HTTPException(status_code=404, detail="Holographic processing not found")
            
        return HolographicProcessingResponse(
            processing_id=processing.processing_id,
            name=processing.name,
            algorithm=processing.algorithm.value,
            input_data=processing.input_data,
            output_data=processing.output_data,
            parameters=processing.parameters,
            processing_time=processing.processing_time,
            quality_metrics=processing.quality_metrics,
            status=processing.status,
            created_at=processing.created_at,
            completed_at=processing.completed_at,
            metadata=processing.metadata
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get holographic processing: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@holographic_router.get("/processing", response_model=List[HolographicProcessingResponse])
async def list_holographic_processing(
    status: Optional[str] = None,
    limit: int = 100,
    holographic_service: HolographicComputingService = Depends(get_holographic_service)
):
    """List holographic processing tasks."""
    try:
        processing_tasks = await holographic_service.list_holographic_processing(status)
        
        return [
            HolographicProcessingResponse(
                processing_id=processing.processing_id,
                name=processing.name,
                algorithm=processing.algorithm.value,
                input_data=processing.input_data,
                output_data=processing.output_data,
                parameters=processing.parameters,
                processing_time=processing.processing_time,
                quality_metrics=processing.quality_metrics,
                status=processing.status,
                created_at=processing.created_at,
                completed_at=processing.completed_at,
                metadata=processing.metadata
            )
            for processing in processing_tasks[:limit]
        ]
        
    except Exception as e:
        logger.error(f"Failed to list holographic processing: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@holographic_router.get("/status", response_model=ServiceStatusResponse)
async def get_service_status(
    holographic_service: HolographicComputingService = Depends(get_holographic_service)
):
    """Get holographic computing service status."""
    try:
        status = await holographic_service.get_service_status()
        
        return ServiceStatusResponse(
            service_status=status["service_status"],
            total_data=status["total_data"],
            total_displays=status["total_displays"],
            total_projections=status["total_projections"],
            total_processing=status["total_processing"],
            active_displays=status["active_displays"],
            running_projections=status["running_projections"],
            running_processing=status["running_processing"],
            spatial_algorithms=status["spatial_algorithms"],
            holographic_engines=status["holographic_engines"],
            holographic_ai_enabled=status["holographic_ai_enabled"],
            spatial_computing_enabled=status["spatial_computing_enabled"],
            real_time_processing=status["real_time_processing"],
            max_data_objects=status["max_data_objects"],
            max_displays=status["max_displays"],
            timestamp=status["timestamp"]
        )
        
    except Exception as e:
        logger.error(f"Failed to get service status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@holographic_router.get("/algorithms", response_model=Dict[str, Any])
async def get_spatial_algorithms(
    holographic_service: HolographicComputingService = Depends(get_holographic_service)
):
    """Get available spatial algorithms."""
    try:
        return holographic_service.spatial_algorithms
        
    except Exception as e:
        logger.error(f"Failed to get spatial algorithms: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@holographic_router.get("/engines", response_model=Dict[str, Any])
async def get_holographic_engines(
    holographic_service: HolographicComputingService = Depends(get_holographic_service)
):
    """Get available holographic engines."""
    try:
        return holographic_service.holographic_engines
        
    except Exception as e:
        logger.error(f"Failed to get holographic engines: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@holographic_router.get("/types", response_model=List[str])
async def get_holographic_types():
    """Get available holographic types."""
    return [htype.value for htype in HolographicType]

@holographic_router.get("/algorithms", response_model=List[str])
async def get_holographic_algorithms():
    """Get available holographic algorithms."""
    return [algorithm.value for algorithm in HolographicAlgorithm]

@holographic_router.get("/dimensions", response_model=List[str])
async def get_spatial_dimensions():
    """Get available spatial dimensions."""
    return [dimension.value for dimension in SpatialDimension]

@holographic_router.delete("/data/{data_id}")
async def delete_holographic_data(
    data_id: str,
    holographic_service: HolographicComputingService = Depends(get_holographic_service)
):
    """Delete holographic data."""
    try:
        if data_id not in holographic_service.holographic_data:
            raise HTTPException(status_code=404, detail="Holographic data not found")
            
        del holographic_service.holographic_data[data_id]
        
        return {"status": "deleted", "data_id": data_id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete holographic data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@holographic_router.delete("/displays/{display_id}")
async def delete_holographic_display(
    display_id: str,
    holographic_service: HolographicComputingService = Depends(get_holographic_service)
):
    """Delete holographic display."""
    try:
        if display_id not in holographic_service.holographic_displays:
            raise HTTPException(status_code=404, detail="Holographic display not found")
            
        del holographic_service.holographic_displays[display_id]
        
        return {"status": "deleted", "display_id": display_id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete holographic display: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

























