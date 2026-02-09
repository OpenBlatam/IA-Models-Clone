"""API Routes"""
from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Query, WebSocket
from typing import List, Optional
import numpy as np

from ...application.dto import InspectionRequest, InspectionResponse
from ...application.services import InspectionApplicationService
from ...config.app_settings import get_settings
from ..schemas import (
    InspectionRequestSchema,
    InspectionResponseSchema,
    BatchInspectionRequestSchema,
    BatchInspectionResponseSchema,
)
from ..dependencies import get_inspection_service
from .websocket import websocket_inspection_stream

router = APIRouter()
settings = get_settings()

@router.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "description": "API for quality control and defect detection",
        "endpoints": {
            "inspect": "/api/v1/inspections",
            "inspect_upload": "/api/v1/inspections/upload",
            "batch": "/api/v1/inspections/batch",
            "health": "/api/v1/health",
            "metrics": "/api/v1/metrics",
            "docs": "/docs",
        },
        "status": "operational"
    }

@router.post("/inspections", response_model=InspectionResponseSchema)
async def inspect_image(
    request: InspectionRequestSchema,
    service: InspectionApplicationService = Depends(get_inspection_service)
):
    """Inspect a single image."""
    try:
        # Convert schema to DTO
        app_request = InspectionRequest(
            image_data=request.image_data,
            image_format=request.image_format,
            config_overrides=request.config_overrides,
            include_visualization=request.include_visualization,
            timeout_seconds=request.timeout_seconds or settings.inspection_timeout,
        )
        
        # Execute use case
        response = service.inspect_image(app_request)
        
        # Convert to schema
        return InspectionResponseSchema(**response.to_dict())
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/inspections/upload", response_model=InspectionResponseSchema)
async def inspect_uploaded_image(
    file: UploadFile = File(...),
    include_visualization: bool = Query(default=False),
    service: InspectionApplicationService = Depends(get_inspection_service)
):
    """Inspect an uploaded image file."""
    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400,
                detail="File must be an image"
            )
        
        # Read file content
        contents = await file.read()
        
        # Validate file size (max 10MB)
        max_size = 10 * 1024 * 1024  # 10MB
        if len(contents) > max_size:
            raise HTTPException(
                status_code=400,
                detail=f"File size exceeds maximum of {max_size / 1024 / 1024}MB"
            )
        
        # Create request
        app_request = InspectionRequest(
            image_data=contents,
            image_format="bytes",
            include_visualization=include_visualization,
        )
        
        # Execute use case
        response = service.inspect_image(app_request)
        
        # Convert to schema
        return InspectionResponseSchema(**response.to_dict())
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/inspections/batch", response_model=BatchInspectionResponseSchema)
async def inspect_batch(
    request: BatchInspectionRequestSchema,
    service: InspectionApplicationService = Depends(get_inspection_service)
):
    """Inspect multiple images in batch."""
    try:
        from ...application.dto import BatchInspectionRequest
        
        # Use default batch size from settings if not provided
        batch_size = request.batch_size or settings.inspection_batch_size
        
        app_request = BatchInspectionRequest(
            images=[InspectionRequest(**img.dict()) for img in request.images],
            batch_size=batch_size,
            parallel=request.parallel,
            max_workers=request.max_workers,
        )
        
        response = service.inspect_batch(app_request)
        return BatchInspectionResponseSchema(**response.to_dict())
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/inspections/{inspection_id}")
async def get_inspection(
    inspection_id: str,
    service: InspectionApplicationService = Depends(get_inspection_service)
):
    """Get inspection by ID."""
    try:
        # This would use repository to fetch inspection
        # For now, placeholder
        raise HTTPException(status_code=501, detail="Not yet implemented")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """Health check endpoint with detailed system checks."""
    from ...infrastructure.metrics import get_metrics_collector
    from ...infrastructure.health import get_health_checker
    
    metrics = get_metrics_collector()
    metrics_data = metrics.get_metrics()
    
    # Run health checks
    health_checker = get_health_checker()
    health_status = health_checker.check_all()
    
    # Calculate success rate
    total = metrics_data["counters"].get("inspections.total", 0)
    successful = metrics_data["counters"].get("inspections.successful", 0)
    success_rate = (successful / max(total, 1)) * 100
    
    return {
        "status": health_status["status"],
        "version": settings.app_version,
        "service": settings.app_name,
        "uptime_seconds": metrics_data["uptime_seconds"],
        "total_inspections": total,
        "success_rate": round(success_rate, 2),
        "checks": health_status["checks"],
        "timestamp": health_status["timestamp"],
    }


@router.get("/metrics")
async def get_metrics():
    """Get system metrics."""
    from ...infrastructure.metrics import get_metrics_collector
    from ...infrastructure.monitoring import get_system_monitor
    
    metrics = get_metrics_collector()
    system_monitor = get_system_monitor()
    
    metrics_data = metrics.get_metrics()
    system_info = system_monitor.check_resources()
    
    return {
        **metrics_data,
        "system": system_info,
    }


@router.get("/settings")
async def get_settings_endpoint():
    """Get application settings (non-sensitive)."""
    return {
        "app_name": settings.app_name,
        "app_version": settings.app_version,
        "api_port": settings.api_port,
        "cache_enabled": settings.cache_enabled,
        "metrics_enabled": settings.metrics_enabled,
        "log_level": settings.log_level,
    }


@router.websocket("/ws/inspection")
async def websocket_endpoint(
    websocket: WebSocket,
    service: InspectionApplicationService = Depends(get_inspection_service)
):
    """WebSocket endpoint for real-time inspection streaming."""
    await websocket_inspection_stream(websocket, service)
