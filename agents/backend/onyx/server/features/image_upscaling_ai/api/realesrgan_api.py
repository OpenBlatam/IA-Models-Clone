"""
Real-ESRGAN API Endpoints
==========================

API endpoints for Real-ESRGAN model management.
"""

from fastapi import APIRouter, HTTPException, Form
from fastapi.responses import JSONResponse
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/realesrgan", tags=["realesrgan"])

try:
    from ..models.realesrgan_integration import (
        RealESRGANWrapper,
        RealESRGANUpscaler,
        REALESRGAN_AVAILABLE
    )
except ImportError:
    REALESRGAN_AVAILABLE = False
    RealESRGANWrapper = None
    RealESRGANUpscaler = None


@router.get("/available")
async def check_available():
    """
    Check if Real-ESRGAN is available.
    
    Returns:
        Availability status
    """
    return JSONResponse(content={
        "available": REALESRGAN_AVAILABLE,
        "message": (
            "Real-ESRGAN is available" if REALESRGAN_AVAILABLE
            else "Real-ESRGAN not installed. Install with: pip install realesrgan basicsr"
        )
    })


@router.get("/models")
async def list_models():
    """
    List all available Real-ESRGAN models.
    
    Returns:
        List of available models
    """
    if not REALESRGAN_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Real-ESRGAN not available. Install with: pip install realesrgan basicsr"
        )
    
    try:
        models = RealESRGANWrapper.list_available_models()
        return JSONResponse(content=models)
    except Exception as e:
        logger.error(f"Error listing models: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/download")
async def download_model(
    model_name: str = Form(..., description="Model name to download"),
    output_path: Optional[str] = Form(None, description="Output path (optional)")
):
    """
    Download a Real-ESRGAN model.
    
    Args:
        model_name: Model name to download
        output_path: Optional output path
        
    Returns:
        Download status
    """
    if not REALESRGAN_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Real-ESRGAN not available. Install with: pip install realesrgan basicsr"
        )
    
    try:
        model_path = RealESRGANWrapper.download_model(model_name, output_path)
        return JSONResponse(content={
            "status": "downloaded",
            "model_name": model_name,
            "model_path": model_path,
            "message": f"Model {model_name} downloaded successfully"
        })
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error downloading model: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/model/{model_name}/info")
async def get_model_info(model_name: str):
    """
    Get information about a specific model.
    
    Args:
        model_name: Model name
        
    Returns:
        Model information
    """
    if not REALESRGAN_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Real-ESRGAN not available"
        )
    
    try:
        models = RealESRGANWrapper.list_available_models()
        if model_name not in models:
            raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
        
        return JSONResponse(content=models[model_name])
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model info: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


