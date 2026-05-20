"""
Service Routes
==============

API routes for enhancement services.
"""

import logging
from fastapi import APIRouter

from ..dependencies import get_agent
from ..models import (
    EnhanceImageRequest,
    EnhanceVideoRequest,
    UpscaleRequest,
    DenoiseRequest,
    RestoreRequest,
    ColorCorrectionRequest
)
from ...utils.validators import ParameterValidator
from ..route_helpers import handle_route_error, create_success_response

logger = logging.getLogger(__name__)

router = APIRouter(tags=["services"])


async def _process_service_request(
    request,
    service_method: str,
    service_name: str
):
    """
    Common handler for service requests.
    
    Args:
        request: Request object
        service_method: Agent method name
        service_name: Service name for messages
        
    Returns:
        Success response
    """
    agent = get_agent()
    
    # Validate common parameters
    ParameterValidator.validate_file_path(request.file_path)
    ParameterValidator.validate_priority(request.priority)
    
    # Get service method
    method = getattr(agent, service_method)
    
    # Prepare parameters
    params = {
        "file_path": request.file_path,
        "options": request.options or {},
        "priority": request.priority
    }
    
    # Add service-specific parameters
    if hasattr(request, "enhancement_type"):
        ParameterValidator.validate_enhancement_type(request.enhancement_type)
        params["enhancement_type"] = request.enhancement_type
    
    if hasattr(request, "scale_factor"):
        ParameterValidator.validate_scale_factor(request.scale_factor)
        params["scale_factor"] = request.scale_factor
    
    if hasattr(request, "noise_level"):
        ParameterValidator.validate_noise_level(request.noise_level)
        params["noise_level"] = request.noise_level
    
    if hasattr(request, "damage_type") and request.damage_type:
        params["damage_type"] = request.damage_type
    
    if hasattr(request, "correction_type"):
        params["correction_type"] = request.correction_type
    
    # Execute service
    task_id = await method(**params)
    
    return create_success_response(
        data={"task_id": task_id},
        message=f"{service_name} task created"
    )


@router.post("/enhance-image")
async def enhance_image(request: EnhanceImageRequest):
    """Enhance an image from file path."""
    try:
        return await _process_service_request(
            request,
            "enhance_image",
            "Image enhancement"
        )
    except Exception as e:
        raise handle_route_error(e, "Error enhancing image")


@router.post("/enhance-video")
async def enhance_video(request: EnhanceVideoRequest):
    """Enhance a video from file path."""
    try:
        return await _process_service_request(
            request,
            "enhance_video",
            "Video enhancement"
        )
    except Exception as e:
        raise handle_route_error(e, "Error enhancing video")


@router.post("/upscale")
async def upscale(request: UpscaleRequest):
    """Upscale an image or video."""
    try:
        return await _process_service_request(
            request,
            "upscale",
            "Upscaling"
        )
    except Exception as e:
        raise handle_route_error(e, "Error upscaling")


@router.post("/denoise")
async def denoise(request: DenoiseRequest):
    """Denoise an image or video."""
    try:
        return await _process_service_request(
            request,
            "denoise",
            "Denoising"
        )
    except Exception as e:
        raise handle_route_error(e, "Error denoising")


@router.post("/restore")
async def restore(request: RestoreRequest):
    """Restore an image."""
    try:
        return await _process_service_request(
            request,
            "restore",
            "Image restoration"
        )
    except Exception as e:
        raise handle_route_error(e, "Error restoring")


@router.post("/color-correction")
async def color_correction(request: ColorCorrectionRequest):
    """Apply color correction to an image or video."""
    try:
        return await _process_service_request(
            request,
            "color_correction",
            "Color correction"
        )
    except Exception as e:
        raise handle_route_error(e, "Error applying color correction")
