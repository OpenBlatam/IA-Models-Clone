"""
Request Processor Utility
=========================

Utilities for processing API requests.
"""

import logging
from typing import Dict, Any, Optional
from fastapi import UploadFile
from PIL import Image
import io

from ...core.utils.image_loader import ImageLoader
from ...core.validators import RequestValidator
from ...core.exceptions import ValidationError

logger = logging.getLogger(__name__)


class RequestProcessor:
    """Processes API requests consistently."""
    
    @staticmethod
    async def process_image_upload(
        image: UploadFile,
        validate: bool = True
    ) -> Image.Image:
        """
        Process uploaded image file.
        
        Args:
            image: Uploaded file
            validate: Whether to validate the image
            
        Returns:
            PIL Image
            
        Raises:
            ValidationError: If validation fails
        """
        # Read image bytes
        image_bytes = await image.read()
        
        if validate:
            # Validate image
            is_valid, error_msg = RequestValidator.validate_image_file(
                image_bytes=image_bytes,
                filename=image.filename
            )
            
            if not is_valid:
                raise ValidationError(
                    message=error_msg or "Invalid image file",
                    field="image"
                )
        
        # Convert to PIL Image
        try:
            pil_image = Image.open(io.BytesIO(image_bytes))
            # Validate and convert using UnifiedImageValidator
            from ...core.utils.image_validator_unified import UnifiedImageValidator
            return UnifiedImageValidator.validate_and_convert(pil_image)
        except Exception as e:
            logger.error(f"Error loading image: {e}")
            raise ValidationError(
                message=f"Cannot load image: {str(e)}",
                field="image"
            )
    
    @staticmethod
    def validate_clothing_request(
        image_bytes: bytes,
        clothing_description: str,
        character_name: Optional[str] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        strength: Optional[float] = None,
    ) -> list:
        """
        Validate clothing change request.
        
        Args:
            image_bytes: Image file bytes
            clothing_description: Clothing description
            character_name: Optional character name
            num_inference_steps: Optional inference steps
            guidance_scale: Optional guidance scale
            strength: Optional strength
            
        Returns:
            List of validation errors (empty if valid)
        """
        return RequestValidator.validate_change_clothing_request(
            image_bytes=image_bytes,
            clothing_description=clothing_description,
            character_name=character_name,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            strength=strength,
        )
    
    @staticmethod
    def create_error_response(
        message: str,
        errors: Optional[list] = None,
        status_code: int = 400
    ) -> Dict[str, Any]:
        """
        Create standardized error response.
        
        Args:
            message: Error message
            errors: Optional list of specific errors
            status_code: HTTP status code
            
        Returns:
            Error response dictionary
        """
        response = {
            "success": False,
            "error": message,
            "status_code": status_code
        }
        
        if errors:
            response["errors"] = errors
        
        return response
    
    @staticmethod
    def create_success_response(
        data: Dict[str, Any],
        message: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create standardized success response.
        
        Args:
            data: Response data
            message: Optional success message
            
        Returns:
            Success response dictionary
        """
        response = {
            "success": True,
            "data": data
        }
        
        if message:
            response["message"] = message
        
        return response

