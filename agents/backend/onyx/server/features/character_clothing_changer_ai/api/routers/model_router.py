"""
Model Router
===========

API endpoints for model operations.
"""

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse
from typing import Dict, Any
import logging

from ...core.clothing_changer_service import ClothingChangerService
from ...api.dependencies import get_service
from ...api.utils.error_handler import APIErrorHandler

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["model"])


@router.get("/model/info", response_model=Dict[str, Any])
async def get_model_info(service: ClothingChangerService = Depends(get_service)):
    """
    Get model information.
    
    Returns:
        Model information
    """
    try:
        info = service.get_model_info()
        return JSONResponse(content=info)
    
    except Exception as e:
        raise APIErrorHandler.handle_error(e, context="get_model_info")


@router.post("/initialize")
async def initialize_model(service: ClothingChangerService = Depends(get_service)):
    """
    Initialize the Flux2 model (with DeepSeek fallback).
    
    Returns:
        Initialization status
    """
    try:
        service.initialize_model()
        
        # Check which model is being used
        model_info = service.get_model_info()
        if model_info.get("fallback_mode"):
            return JSONResponse(content={
                "status": "initialized",
                "message": "DeepSeek model initialized as fallback (Flux2 unavailable)",
                "using_deepseek_fallback": True,
                "model_type": "DeepSeek"
            })
        else:
            return JSONResponse(content={
                "status": "initialized",
                "message": "Flux2 model initialized successfully",
                "using_deepseek_fallback": False,
                "model_type": "Flux2"
            })
    
    except RuntimeError as e:
        # RuntimeError is raised when both Flux2 and DeepSeek fail
        raise APIErrorHandler.handle_runtime_error(e, context="initialize_model")
    except Exception as e:
        # For other errors, check if DeepSeek fallback is available
        logger.warning(f"Error during initialization: {e}")
        model_info = service.get_model_info()
        if model_info.get("fallback_mode"):
            return JSONResponse(content={
                "status": "initialized",
                "message": "Using DeepSeek fallback (Flux2 failed to initialize)",
                "using_deepseek_fallback": True,
                "model_type": "DeepSeek",
                "warning": str(e)
            })
        else:
            raise APIErrorHandler.handle_error(e, context="initialize_model")

