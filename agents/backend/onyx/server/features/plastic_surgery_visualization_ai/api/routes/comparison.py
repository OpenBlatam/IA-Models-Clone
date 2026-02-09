"""Comparison endpoints for before/after visualization."""

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import FileResponse
from pathlib import Path
from PIL import Image
import uuid
from datetime import datetime

from api.schemas.comparison import ComparisonRequest, ComparisonResponse
from services.visualization_service import VisualizationService
from core.dependencies import get_visualization_service
from config.settings import settings
from utils.decorators_advanced import track_metrics
from utils.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)


@router.post("/compare", response_model=ComparisonResponse)
@track_metrics("api.compare")
async def create_comparison(
    request: ComparisonRequest,
    service: VisualizationService = Depends(get_visualization_service)
) -> ComparisonResponse:
    """
    Create a before/after comparison image.
    
    Args:
        request: Comparison request
        service: Visualization service instance
        
    Returns:
        ComparisonResponse with comparison image URL
    """
    try:
        # Get visualization image
        viz_path = await service.get_visualization(request.visualization_id)
        if not viz_path:
            raise HTTPException(status_code=404, detail="Visualization not found")
        
        # Load images
        visualization_image = Image.open(viz_path)
        
        # Create comparison based on layout
        if request.layout == "side_by_side":
            comparison_image = _create_side_by_side(
                visualization_image,
                include_original=request.include_original
            )
        elif request.layout == "overlay":
            comparison_image = _create_overlay(
                visualization_image,
                include_original=request.include_original
            )
        else:
            comparison_image = visualization_image
        
        # Save comparison
        comparison_id = str(uuid.uuid4())
        output_path = Path(settings.output_dir) / f"comparison_{comparison_id}.{settings.output_format}"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        comparison_image.save(output_path, format=settings.output_format.upper())
        
        return ComparisonResponse(
            comparison_id=comparison_id,
            image_url=f"/api/v1/compare/{comparison_id}",
            original_url=f"/api/v1/visualize/{request.visualization_id}" if request.include_original else None,
            visualization_url=f"/api/v1/visualize/{request.visualization_id}",
            layout=request.layout,
            created_at=datetime.utcnow().isoformat()
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating comparison: {e}")
        raise HTTPException(status_code=500, detail="Failed to create comparison")


@router.get("/compare/{comparison_id}")
async def get_comparison(comparison_id: str) -> FileResponse:
    """Get comparison image."""
    comparison_path = Path(settings.output_dir) / f"comparison_{comparison_id}.{settings.output_format}"
    if not comparison_path.exists():
        raise HTTPException(status_code=404, detail="Comparison not found")
    return FileResponse(comparison_path)


def _create_side_by_side(viz_image: Image.Image, include_original: bool = True) -> Image.Image:
    """Create side-by-side comparison."""
    if not include_original:
        return viz_image
    
    # For now, return visualization image
    # In production, would load original and combine
    return viz_image


def _create_overlay(viz_image: Image.Image, include_original: bool = True) -> Image.Image:
    """Create overlay comparison."""
    # For now, return visualization image
    # In production, would create overlay effect
    return viz_image
