"""
Clothing Router
==============

API endpoints for clothing change operations.
"""

from fastapi import APIRouter, File, UploadFile, HTTPException, Form, Depends
from fastapi.responses import JSONResponse
from typing import Optional, Dict, Any
from PIL import Image
import io
import logging

from ...api.dependencies import ServiceDep
from ...api.utils import add_image_urls_to_result
from ...api.utils.error_handler import APIErrorHandler
from ...api.utils.request_processor import RequestProcessor
from ...core.exceptions import ValidationError, ImageValidationError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["clothing"])


@router.post("/change-clothing", response_model=Dict[str, Any])
async def change_clothing(
    service: ServiceDep,
    image: UploadFile = File(..., description="Character image"),
    clothing_description: str = Form(..., description="Description of new clothing"),
    character_name: Optional[str] = Form(None, description="Character name"),
    prompt: Optional[str] = Form(None, description="Optional full prompt"),
    negative_prompt: Optional[str] = Form(None, description="Negative prompt"),
    num_inference_steps: Optional[int] = Form(None, description="Number of inference steps"),
    guidance_scale: Optional[float] = Form(None, description="Guidance scale"),
    strength: Optional[float] = Form(None, description="Inpainting strength"),
    save_tensor: bool = Form(True, description="Save as ComfyUI safe tensor"),
):
    """
    Change clothing in character image.
    
    Returns:
        Clothing change result information
    """
    try:
        # Process image upload
        pil_image = await RequestProcessor.process_image_upload(image)
        
        # Validate request
        image_bytes = await image.read()
        await image.seek(0)  # Reset file pointer
        
        validation_errors = RequestProcessor.validate_clothing_request(
            image_bytes=image_bytes,
            clothing_description=clothing_description,
            character_name=character_name,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            strength=strength,
        )
        
        if validation_errors:
            raise ValidationError(
                message="Validation failed",
                details={"errors": validation_errors}
            )
        
        # Change clothing
        result = service.change_clothing(
            image=pil_image,
            clothing_description=clothing_description,
            character_name=character_name,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            strength=strength,
            save_tensor=save_tensor,
        )
        
        # Add image URLs to result
        result = add_image_urls_to_result(result)
        
        return JSONResponse(content=result)
    
    except (ValidationError, ImageValidationError, RuntimeError) as e:
        raise APIErrorHandler.handle_error(e, context="change_clothing")
    except Exception as e:
        raise APIErrorHandler.handle_error(e, context="change_clothing")

