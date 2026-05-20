"""
Image Upscaling API
===================

FastAPI endpoints for image upscaling operations.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, Dict, Any
from pathlib import Path
import logging
from PIL import Image
import io

from ..core.upscaling_service import UpscalingService
from ..config.upscaling_config import UpscalingConfig
from ..models.presets import PresetManager
from ..models.image_comparison import ImageComparison

logger = logging.getLogger(__name__)

# Initialize service
config = UpscalingConfig.from_env()
service = UpscalingService(config=config)

# Create FastAPI app
app = FastAPI(
    title="Image Upscaling AI API",
    description="API for upscaling images using AI and optimization_core",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create router
from fastapi import APIRouter
router = APIRouter(prefix="/api/v1", tags=["upscaling"])


@router.post("/upscale", response_model=Dict[str, Any])
async def upscale_image(
    image: UploadFile = File(..., description="Image to upscale"),
    scale_factor: float = Form(2.0, description="Scale factor (e.g., 2.0 for 2x)"),
    use_ai: Optional[bool] = Form(None, description="Use AI enhancement"),
    use_optimization_core: Optional[bool] = Form(None, description="Use optimization_core"),
    save_result: bool = Form(True, description="Save result to disk"),
):
    """
    Upscale an image.
    
    Args:
        image: Image file to upscale
        scale_factor: Scale factor (e.g., 2.0 for 2x upscaling)
        use_ai: Whether to use AI enhancement
        use_optimization_core: Whether to use optimization_core
        save_result: Whether to save the result
        
    Returns:
        Upscaling result information
    """
    try:
        # Load image
        image_bytes = await image.read()
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Upscale
        result = await service.upscale_image(
            image=pil_image,
            scale_factor=scale_factor,
            use_ai=use_ai,
            use_optimization_core=use_optimization_core,
            save_result=save_result,
        )
        
        return JSONResponse(content=result)
    
    except Exception as e:
        logger.error(f"Error upscaling image: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upscale-batch", response_model=List[Dict[str, Any]])
async def upscale_batch(
    images: list[UploadFile] = File(..., description="Images to upscale"),
    scale_factor: float = Form(2.0, description="Scale factor"),
    use_ai: Optional[bool] = Form(None, description="Use AI enhancement"),
    use_optimization_core: Optional[bool] = Form(None, description="Use optimization_core"),
):
    """
    Upscale multiple images in batch.
    
    Args:
        images: List of image files to upscale
        scale_factor: Scale factor
        use_ai: Whether to use AI enhancement
        use_optimization_core: Whether to use optimization_core
        
    Returns:
        List of upscaling results
    """
    try:
        # Load images
        pil_images = []
        for img_file in images:
            image_bytes = await img_file.read()
            pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            pil_images.append(pil_image)
        
        # Upscale batch
        results = await service.batch_upscale(
            images=pil_images,
            scale_factor=scale_factor,
            use_ai=use_ai,
            use_optimization_core=use_optimization_core,
        )
        
        return JSONResponse(content=results)
    
    except Exception as e:
        logger.error(f"Error upscaling batch: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/model/info", response_model=Dict[str, Any])
async def get_model_info():
    """
    Get model information.
    
    Returns:
        Model information
    """
    try:
        info = service.get_model_info()
        return JSONResponse(content=info)
    
    except Exception as e:
        logger.error(f"Error getting model info: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/initialize")
async def initialize_model():
    """
    Initialize the upscaling model.
    
    Returns:
        Initialization status
    """
    try:
        service.initialize_model()
        return JSONResponse(content={"status": "initialized", "message": "Model initialized successfully"})
    
    except Exception as e:
        logger.error(f"Error initializing model: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        Health status
    """
    try:
        health = await service.health_check()
        return JSONResponse(content=health)
    
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": str(e)}
        )


@router.post("/upscale-preset", response_model=Dict[str, Any])
async def upscale_with_preset(
    image: UploadFile = File(..., description="Image to upscale"),
    preset: str = Form(..., description="Preset name"),
    scale_factor: Optional[float] = Form(None, description="Override scale factor"),
):
    """
    Upscale image using a preset.
    
    Args:
        image: Image file to upscale
        preset: Preset name (photo_enhancement, artwork_upscale, pixel_art, document_scan, video_frame)
        scale_factor: Optional override for scale factor
        
    Returns:
        Upscaling result
    """
    try:
        # Get preset configuration
        preset_config = PresetManager.apply_preset(
            preset,
            scale_factor=scale_factor
        )
        
        # Load image
        image_bytes = await image.read()
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Upscale with preset
        result = await service.upscale_image(
            image=pil_image,
            scale_factor=preset_config["scale_factor"],
            use_ai=preset_config["use_ai"],
            use_optimization_core=preset_config["use_optimization_core"],
            save_result=True,
        )
        
        result["preset"] = preset
        return JSONResponse(content=result)
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error upscaling with preset: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/presets", response_model=Dict[str, Any])
async def list_presets():
    """
    List all available presets.
    
    Returns:
        Dictionary of available presets
    """
    try:
        presets = PresetManager.list_presets()
        return JSONResponse(content=presets)
    
    except Exception as e:
        logger.error(f"Error listing presets: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/comparison")
async def create_comparison(
    original: UploadFile = File(..., description="Original image"),
    upscaled: UploadFile = File(..., description="Upscaled image"),
    save_result: bool = Form(True, description="Save comparison image"),
):
    """
    Create side-by-side comparison of original and upscaled images.
    
    Args:
        original: Original image
        upscaled: Upscaled image
        save_result: Whether to save the comparison
        
    Returns:
        Comparison image or file path
    """
    try:
        # Load images
        orig_bytes = await original.read()
        upscaled_bytes = await upscaled.read()
        
        orig_img = Image.open(io.BytesIO(orig_bytes)).convert("RGB")
        upscaled_img = Image.open(io.BytesIO(upscaled_bytes)).convert("RGB")
        
        # Create comparison
        comparison = service.create_comparison(
            original=orig_img,
            upscaled=upscaled_img,
            save_path=str(service.output_dir / "comparison.png") if save_result else None
        )
        
        if save_result:
            return JSONResponse(content={
                "saved": True,
                "path": str(service.output_dir / "comparison.png")
            })
        else:
            # Return image as bytes
            img_bytes = io.BytesIO()
            comparison.save(img_bytes, format="PNG")
            img_bytes.seek(0)
            return FileResponse(
                img_bytes,
                media_type="image/png",
                filename="comparison.png"
            )
    
    except Exception as e:
        logger.error(f"Error creating comparison: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/cache/stats", response_model=Dict[str, Any])
async def get_cache_stats():
    """
    Get cache statistics.
    
    Returns:
        Cache statistics
    """
    try:
        stats = service.get_cache_stats()
        return JSONResponse(content=stats)
    
    except Exception as e:
        logger.error(f"Error getting cache stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cache/clear")
async def clear_cache():
    """
    Clear result cache.
    
    Returns:
        Clear status
    """
    try:
        service.clear_cache()
        return JSONResponse(content={"status": "cleared", "message": "Cache cleared successfully"})
    
    except Exception as e:
        logger.error(f"Error clearing cache: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# Include routers
app.include_router(router)

# Include Real-ESRGAN routers if available
try:
    from .realesrgan_api import router as realesrgan_router
    app.include_router(realesrgan_router)
except ImportError:
    logger.warning("Real-ESRGAN API router not available")

try:
    from .realesrgan_advanced_api import router as realesrgan_advanced_router
    app.include_router(realesrgan_advanced_router)
except ImportError:
    logger.warning("Real-ESRGAN advanced API router not available")

# Include enhanced API if available
try:
    from .enhanced_api import router as enhanced_router
    app.include_router(enhanced_router)
except ImportError:
    logger.warning("Enhanced API router not available")

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "Image Upscaling AI",
        "version": "1.0.0",
        "status": "running",
    }

