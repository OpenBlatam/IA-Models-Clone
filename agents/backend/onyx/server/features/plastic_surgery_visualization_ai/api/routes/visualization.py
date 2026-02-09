"""Visualization endpoints for plastic surgery preview."""

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Query
from fastapi.responses import FileResponse
from typing import Optional

from api.schemas.visualization import (
    VisualizationRequest,
    VisualizationResponse,
    SurgeryType
)
from core.dependencies import get_visualization_service
from core.constants import SURGERY_TYPES_METADATA, DEFAULT_INTENSITY
from core.exceptions import PlasticSurgeryAIException
from services.visualization_service import VisualizationService
from utils.decorators_advanced import track_metrics, handle_exceptions
from utils.metrics import metrics_collector
from utils.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)


@router.post("/visualize", response_model=VisualizationResponse)
async def create_visualization(
    request: VisualizationRequest,
    service: VisualizationService = Depends(get_visualization_service)
) -> VisualizationResponse:
    """
    Create a visualization of how the user will look after plastic surgery.
    
    Args:
        request: Visualization request with surgery type and parameters
        service: Visualization service instance
        
    Returns:
        VisualizationResponse with preview image URL
    """
    metrics_collector.increment("api.requests.visualize")
    
    try:
        result = await service.create_visualization(request)
        return result
    except PlasticSurgeryAIException:
        metrics_collector.increment("api.errors.visualize")
        raise
    except Exception as e:
        metrics_collector.increment("api.errors.visualize")
        logger.error(f"Unexpected error creating visualization: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/visualize/upload", response_model=VisualizationResponse)
@track_metrics("api.upload")
async def visualize_from_upload(
    file: UploadFile = File(...),
    surgery_type: SurgeryType = Query(...),
    intensity: Optional[float] = Query(None, ge=0.0, le=1.0),
    service: VisualizationService = Depends(get_visualization_service)
) -> VisualizationResponse:
    """
    Create visualization from uploaded image.
    
    Args:
        file: Uploaded image file
        surgery_type: Type of surgery to visualize
        intensity: Intensity of the surgery effect (0.0 to 1.0)
        service: Visualization service instance
        
    Returns:
        VisualizationResponse with preview image
    """
    from utils.validation_utils import validate_uploaded_file, validate_intensity
    
    image_data = await validate_uploaded_file(file)
    validated_intensity = validate_intensity(intensity, DEFAULT_INTENSITY)
    
    request = VisualizationRequest(
        surgery_type=surgery_type,
        intensity=validated_intensity,
        image_data=image_data
    )
    
    return await service.create_visualization(request)


@router.get("/visualize/{visualization_id}")
@track_metrics("api.get_visualization")
async def get_visualization(
    visualization_id: str,
    service: VisualizationService = Depends(get_visualization_service)
) -> FileResponse:
    """
    Get a previously created visualization.
    
    Args:
        visualization_id: ID of the visualization
        service: Visualization service instance
        
    Returns:
        Visualization image file
    """
    from utils.validation_utils import validate_visualization_id
    
    validated_id = validate_visualization_id(visualization_id)
    image_path = await service.get_visualization(validated_id)
    return FileResponse(image_path)


@router.get("/surgery-types")
async def get_surgery_types() -> dict:
    """
    Get available surgery types.
    
    Returns:
        List of available surgery types with descriptions
    """
    surgery_types = [
        {
            "type": surgery_type.value,
            **metadata
        }
        for surgery_type, metadata in SURGERY_TYPES_METADATA.items()
    ]
    
    return {"surgery_types": surgery_types}

