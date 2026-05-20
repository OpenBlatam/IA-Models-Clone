"""
Real-ESRGAN Advanced API Endpoints
===================================

Advanced API endpoints for Real-ESRGAN features.
"""

from fastapi import APIRouter, HTTPException, Form, File, UploadFile
from fastapi.responses import JSONResponse
from typing import Optional, List, Dict, Any
import logging
import io
import time
from PIL import Image

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/realesrgan/advanced", tags=["realesrgan-advanced"])

try:
    from ..models.realesrgan_manager import RealESRGANModelManager
    from ..models.realesrgan_comparison import ModelComparison
    from ..models.realesrgan_integration import REALESRGAN_AVAILABLE
except ImportError:
    REALESRGAN_AVAILABLE = False
    RealESRGANModelManager = None
    ModelComparison = None

# Global manager instance
_model_manager: Optional[RealESRGANModelManager] = None


def get_model_manager() -> RealESRGANModelManager:
    """Get or create model manager instance."""
    global _model_manager
    if _model_manager is None:
        _model_manager = RealESRGANModelManager(
            max_cached_models=3,
            auto_download=False
        )
    return _model_manager


@router.post("/upscale-smart")
async def upscale_smart(
    image: UploadFile = File(..., description="Image to upscale"),
    scale_factor: float = Form(..., description="Scale factor"),
    auto_select_model: bool = Form(True, description="Auto-select best model"),
    model_name: Optional[str] = Form(None, description="Specific model name"),
):
    """
    Smart upscaling with automatic model selection.
    
    Args:
        image: Image file
        scale_factor: Scale factor
        auto_select_model: Auto-select best model based on image type
        model_name: Specific model (overrides auto-select)
        
    Returns:
        Upscaling result
    """
    if not REALESRGAN_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Real-ESRGAN not available"
        )
    
    try:
        # Load image
        image_bytes = await image.read()
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Get manager
        manager = get_model_manager()
        
        # Upscale
        upscaled = await manager.upscale_async(
            pil_image,
            scale_factor,
            model_name=model_name,
            auto_select=auto_select_model
        )
        
        # Save result
        output_path = f"/tmp/upscaled_{int(time.time())}.png"
        upscaled.save(output_path)
        
        return JSONResponse(content={
            "status": "success",
            "output_path": output_path,
            "original_size": pil_image.size,
            "upscaled_size": upscaled.size,
            "scale_factor": scale_factor,
            "model_used": manager.select_best_model(pil_image, scale_factor) if auto_select_model else model_name,
        })
        
    except Exception as e:
        logger.error(f"Error in smart upscale: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/compare-models")
async def compare_models(
    image: UploadFile = File(..., description="Image to test"),
    scale_factor: float = Form(4.0, description="Scale factor"),
    model_names: Optional[str] = Form(None, description="Comma-separated model names"),
):
    """
    Compare different Real-ESRGAN models on the same image.
    
    Args:
        image: Image file
        scale_factor: Scale factor
        model_names: Comma-separated list of model names
        
    Returns:
        Comparison results
    """
    if not REALESRGAN_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Real-ESRGAN not available"
        )
    
    try:
        # Load image
        image_bytes = await image.read()
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Parse model names
        models = None
        if model_names:
            models = [m.strip() for m in model_names.split(",")]
        
        # Compare
        comparison = ModelComparison()
        results = comparison.compare_models(
            pil_image,
            scale_factor,
            model_names=models
        )
        
        # Convert PIL images to base64 for JSON response
        import base64
        from io import BytesIO
        
        for model_name, result in results["results"].items():
            if result.get("success") and result.get("upscaled_image"):
                img = result["upscaled_image"]
                buffer = BytesIO()
                img.save(buffer, format="PNG")
                img_str = base64.b64encode(buffer.getvalue()).decode()
                result["upscaled_image_base64"] = img_str
                del result["upscaled_image"]  # Remove PIL image from response
        
        return JSONResponse(content=results)
        
    except Exception as e:
        logger.error(f"Error comparing models: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/manager/stats")
async def get_manager_stats():
    """Get model manager statistics."""
    if not REALESRGAN_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Real-ESRGAN not available"
        )
    
    try:
        manager = get_model_manager()
        stats = manager.get_stats()
        cache_info = manager.get_cache_info()
        
        return JSONResponse(content={
            "stats": stats,
            "cache": cache_info,
        })
    except Exception as e:
        logger.error(f"Error getting stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/manager/clear-cache")
async def clear_cache():
    """Clear model cache."""
    if not REALESRGAN_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Real-ESRGAN not available"
        )
    
    try:
        manager = get_model_manager()
        manager.clear_cache()
        
        return JSONResponse(content={
            "status": "success",
            "message": "Cache cleared"
        })
    except Exception as e:
        logger.error(f"Error clearing cache: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/detect-image-type")
async def detect_image_type(
    image: UploadFile = File(..., description="Image to analyze")
):
    """
    Detect image type (anime/photo/artwork).
    
    Args:
        image: Image file
        
    Returns:
        Detected image type and recommended model
    """
    if not REALESRGAN_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Real-ESRGAN not available"
        )
    
    try:
        # Load image
        image_bytes = await image.read()
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Detect type
        manager = get_model_manager()
        image_type = manager.detect_image_type(pil_image)
        
        # Recommend model
        recommended_model = manager.select_best_model(pil_image, 4.0, image_type)
        
        return JSONResponse(content={
            "image_type": image_type,
            "recommended_model": recommended_model,
            "image_size": pil_image.size,
        })
        
    except Exception as e:
        logger.error(f"Error detecting image type: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch-upscale")
async def batch_upscale(
    images: List[UploadFile] = File(..., description="Images to upscale"),
    scale_factor: float = Form(..., description="Scale factor"),
    max_concurrent: int = Form(2, description="Max concurrent upscales"),
):
    """
    Upscale multiple images in parallel.
    
    Args:
        images: List of image files
        scale_factor: Scale factor
        max_concurrent: Maximum concurrent upscales
        
    Returns:
        Batch upscaling results
    """
    if not REALESRGAN_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Real-ESRGAN not available"
        )
    
    try:
        # Load images
        pil_images = []
        for img_file in images:
            image_bytes = await img_file.read()
            pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            pil_images.append(pil_image)
        
        # Batch upscale
        manager = get_model_manager()
        upscaled_images = await manager.batch_upscale_async(
            pil_images,
            scale_factor,
            max_concurrent=max_concurrent
        )
        
        # Save results
        results = []
        import time
        base_path = f"/tmp/batch_{int(time.time())}"
        
        for idx, upscaled in enumerate(upscaled_images):
            output_path = f"{base_path}_{idx}.png"
            upscaled.save(output_path)
            results.append({
                "index": idx,
                "output_path": output_path,
                "original_size": pil_images[idx].size,
                "upscaled_size": upscaled.size,
            })
        
        return JSONResponse(content={
            "status": "success",
            "total_images": len(results),
            "results": results,
        })
        
    except Exception as e:
        logger.error(f"Error in batch upscale: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

