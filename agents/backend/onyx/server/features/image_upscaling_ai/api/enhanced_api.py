"""
Enhanced API Endpoints
======================

Enhanced API with all intelligent features.
"""

from fastapi import APIRouter, HTTPException, File, UploadFile, Form
from fastapi.responses import JSONResponse
from typing import Optional
import logging
import io
from PIL import Image

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/enhanced", tags=["enhanced"])

try:
    from ..core.enhanced_service import EnhancedUpscalingService
    from ..config.upscaling_config import UpscalingConfig
except ImportError:
    EnhancedUpscalingService = None
    UpscalingConfig = None

# Global service instance
_service: Optional[EnhancedUpscalingService] = None


def get_service() -> EnhancedUpscalingService:
    """Get or create enhanced service instance."""
    global _service
    if _service is None:
        config = UpscalingConfig.from_env() if UpscalingConfig else None
        _service = EnhancedUpscalingService(config=config)
    return _service


@router.post("/upscale")
async def enhanced_upscale(
    image: UploadFile = File(..., description="Image to upscale"),
    scale_factor: Optional[float] = Form(None, description="Scale factor"),
    use_recommendations: bool = Form(True, description="Use smart recommendations"),
    validate_quality: bool = Form(True, description="Validate output quality"),
):
    """
    Enhanced upscaling with all intelligent features.
    
    Args:
        image: Image file
        scale_factor: Scale factor (optional, uses recommendation if not provided)
        use_recommendations: Use smart recommendations
        validate_quality: Validate output quality
        
    Returns:
        Enhanced upscaling result
    """
    if not EnhancedUpscalingService:
        raise HTTPException(
            status_code=503,
            detail="Enhanced service not available"
        )
    
    try:
        service = get_service()
        
        # Load image
        image_bytes = await image.read()
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Upscale
        result = await service.upscale_image_enhanced(
            pil_image,
            scale_factor=scale_factor,
            use_recommendations=use_recommendations,
            validate_quality=validate_quality
        )
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result.get("error", "Upscaling failed"))
        
        # Convert image to base64 for response
        import base64
        from io import BytesIO
        
        buffer = BytesIO()
        result["upscaled_image"].save(buffer, format="PNG")
        img_str = base64.b64encode(buffer.getvalue()).decode()
        
        # Remove PIL image from response
        response = result.copy()
        response["upscaled_image_base64"] = img_str
        del response["upscaled_image"]
        
        return JSONResponse(content=response)
        
    except Exception as e:
        logger.error(f"Error in enhanced upscale: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/feedback")
async def submit_feedback(
    operation_id: str = Form(..., description="Operation ID"),
    satisfaction: float = Form(..., description="Satisfaction (0.0-1.0)"),
    quality_rating: float = Form(..., description="Quality rating (0.0-1.0)"),
    speed_rating: float = Form(..., description="Speed rating (0.0-1.0)"),
    comments: Optional[str] = Form(None, description="Comments"),
):
    """
    Submit user feedback.
    
    Args:
        operation_id: Operation ID
        satisfaction: Overall satisfaction
        quality_rating: Quality rating
        speed_rating: Speed rating
        comments: Optional comments
        
    Returns:
        Feedback submission status
    """
    if not EnhancedUpscalingService:
        raise HTTPException(
            status_code=503,
            detail="Enhanced service not available"
        )
    
    try:
        service = get_service()
        service.submit_feedback(
            operation_id,
            satisfaction,
            quality_rating,
            speed_rating,
            comments
        )
        
        return JSONResponse(content={
            "status": "success",
            "message": "Feedback submitted"
        })
        
    except Exception as e:
        logger.error(f"Error submitting feedback: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def get_status():
    """
    Get comprehensive system status.
    
    Returns:
        System status with all metrics
    """
    if not EnhancedUpscalingService:
        raise HTTPException(
            status_code=503,
            detail="Enhanced service not available"
        )
    
    try:
        service = get_service()
        status = service.get_system_status()
        
        return JSONResponse(content=status)
        
    except Exception as e:
        logger.error(f"Error getting status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/recommendations")
async def get_recommendations(
    image: UploadFile = File(..., description="Image to analyze"),
    target_scale: float = Form(4.0, description="Target scale factor"),
    prioritize_speed: bool = Form(False, description="Prioritize speed"),
):
    """
    Get upscaling recommendations for an image.
    
    Args:
        image: Image file
        target_scale: Target scale factor
        prioritize_speed: Prioritize speed over quality
        
    Returns:
        Recommendations
    """
    if not EnhancedUpscalingService:
        raise HTTPException(
            status_code=503,
            detail="Enhanced service not available"
        )
    
    try:
        service = get_service()
        
        # Load image
        image_bytes = await image.read()
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Get recommendation
        recommendation = service.recommender.recommend(
            pil_image,
            target_scale,
            prioritize_speed=prioritize_speed
        )
        
        return JSONResponse(content={
            "method": recommendation.method,
            "scale_factor": recommendation.scale_factor,
            "preprocessing_mode": recommendation.preprocessing_mode,
            "postprocessing_mode": recommendation.postprocessing_mode,
            "expected_quality": recommendation.expected_quality,
            "expected_time": recommendation.expected_time,
            "confidence": recommendation.confidence,
            "reasoning": recommendation.reasoning
        })
        
    except Exception as e:
        logger.error(f"Error getting recommendations: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


