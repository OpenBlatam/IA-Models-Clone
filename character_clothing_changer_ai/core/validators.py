"""
Validators Module
=================
Centralized validation functions for the Character Clothing Changer AI.
"""

import os
from pathlib import Path
from typing import Optional, Tuple, List
from PIL import Image
import io
import logging

from .constants import (
    MAX_IMAGE_SIZE,
    MIN_IMAGE_SIZE,
    SUPPORTED_IMAGE_FORMATS,
    MAX_IMAGE_FILE_SIZE,
    MIN_CLOTHING_DESCRIPTION_LENGTH,
    MAX_CLOTHING_DESCRIPTION_LENGTH,
    MIN_CHARACTER_NAME_LENGTH,
    MAX_CHARACTER_NAME_LENGTH,
    ERROR_IMAGE_NOT_PROVIDED,
    ERROR_INVALID_IMAGE_FORMAT,
    ERROR_IMAGE_TOO_LARGE,
    ERROR_INVALID_DESCRIPTION,
)

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Custom exception for validation errors."""
    
    def __init__(self, message: str, field: Optional[str] = None):
        self.message = message
        self.field = field
        super().__init__(self.message)


class ImageValidator:
    """Validator for image-related operations."""
    
    @staticmethod
    def validate_image_file(image_bytes: bytes, filename: Optional[str] = None) -> Tuple[bool, Optional[str]]:
        """
        Validate image file.
        
        Args:
            image_bytes: Image file bytes
            filename: Optional filename for format detection
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not image_bytes:
            return False, ERROR_IMAGE_NOT_PROVIDED
        
        # Check file size
        if len(image_bytes) > MAX_IMAGE_FILE_SIZE:
            return False, ERROR_IMAGE_TOO_LARGE
        
        # Try to open and validate image
        try:
            image = Image.open(io.BytesIO(image_bytes))
            image_format = image.format
            
            # Check format
            if image_format not in SUPPORTED_IMAGE_FORMATS:
                return False, ERROR_INVALID_IMAGE_FORMAT
            
            # Check dimensions
            width, height = image.size
            max_dimension = max(width, height)
            min_dimension = min(width, height)
            
            if max_dimension > MAX_IMAGE_SIZE:
                return False, f"Image dimensions ({width}x{height}) exceed maximum size of {MAX_IMAGE_SIZE}px"
            
            if min_dimension < MIN_IMAGE_SIZE:
                return False, f"Image dimensions ({width}x{height}) are below minimum size of {MIN_IMAGE_SIZE}px"
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            return True, None
            
        except Exception as e:
            logger.error(f"Error validating image: {e}")
            return False, f"Invalid image file: {str(e)}"
    
    @staticmethod
    def validate_image_path(image_path: Path) -> Tuple[bool, Optional[str]]:
        """
        Validate image file path.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not image_path.exists():
            return False, f"Image file not found: {image_path}"
        
        if not image_path.is_file():
            return False, f"Path is not a file: {image_path}"
        
        # Check file size
        file_size = image_path.stat().st_size
        if file_size > MAX_IMAGE_FILE_SIZE:
            return False, ERROR_IMAGE_TOO_LARGE
        
        # Check extension
        extension = image_path.suffix.upper().lstrip('.')
        if extension not in SUPPORTED_IMAGE_FORMATS:
            return False, ERROR_INVALID_IMAGE_FORMAT
        
        return True, None


class TextValidator:
    """Validator for text inputs."""
    
    @staticmethod
    def validate_clothing_description(description: str) -> Tuple[bool, Optional[str]]:
        """
        Validate clothing description.
        
        Args:
            description: Clothing description text
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not description:
            return False, ERROR_INVALID_DESCRIPTION
        
        description = description.strip()
        
        if len(description) < MIN_CLOTHING_DESCRIPTION_LENGTH:
            return False, f"Description must be at least {MIN_CLOTHING_DESCRIPTION_LENGTH} characters"
        
        if len(description) > MAX_CLOTHING_DESCRIPTION_LENGTH:
            return False, f"Description must not exceed {MAX_CLOTHING_DESCRIPTION_LENGTH} characters"
        
        return True, None
    
    @staticmethod
    def validate_character_name(name: Optional[str]) -> Tuple[bool, Optional[str]]:
        """
        Validate character name (optional field).
        
        Args:
            name: Character name (can be None)
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if name is None:
            return True, None
        
        name = name.strip()
        
        if len(name) < MIN_CHARACTER_NAME_LENGTH:
            return False, f"Character name must be at least {MIN_CHARACTER_NAME_LENGTH} character"
        
        if len(name) > MAX_CHARACTER_NAME_LENGTH:
            return False, f"Character name must not exceed {MAX_CHARACTER_NAME_LENGTH} characters"
        
        return True, None
    
    @staticmethod
    def validate_prompt(prompt: Optional[str], max_length: int = 1000) -> Tuple[bool, Optional[str]]:
        """
        Validate prompt (optional field).
        
        Args:
            prompt: Prompt text (can be None)
            max_length: Maximum prompt length
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if prompt is None:
            return True, None
        
        prompt = prompt.strip()
        
        if len(prompt) > max_length:
            return False, f"Prompt must not exceed {max_length} characters"
        
        return True, None


class ParameterValidator:
    """Validator for generation parameters."""
    
    @staticmethod
    def validate_num_inference_steps(steps: Optional[int]) -> Tuple[bool, Optional[str]]:
        """
        Validate number of inference steps.
        
        Args:
            steps: Number of inference steps
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if steps is None:
            return True, None
        
        if steps < 1:
            return False, "Number of inference steps must be at least 1"
        
        if steps > 100:
            return False, "Number of inference steps must not exceed 100"
        
        return True, None
    
    @staticmethod
    def validate_guidance_scale(scale: Optional[float]) -> Tuple[bool, Optional[str]]:
        """
        Validate guidance scale.
        
        Args:
            scale: Guidance scale value
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if scale is None:
            return True, None
        
        if scale < 1.0:
            return False, "Guidance scale must be at least 1.0"
        
        if scale > 20.0:
            return False, "Guidance scale must not exceed 20.0"
        
        return True, None
    
    @staticmethod
    def validate_strength(strength: Optional[float]) -> Tuple[bool, Optional[str]]:
        """
        Validate inpainting strength.
        
        Args:
            strength: Inpainting strength value
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if strength is None:
            return True, None
        
        if strength < 0.0:
            return False, "Strength must be at least 0.0"
        
        if strength > 1.0:
            return False, "Strength must not exceed 1.0"
        
        return True, None


class RequestValidator:
    """Validator for API requests."""
    
    @staticmethod
    def validate_change_clothing_request(
        image_bytes: Optional[bytes],
        clothing_description: Optional[str],
        character_name: Optional[str] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        strength: Optional[float] = None,
    ) -> List[str]:
        """
        Validate a complete change clothing request.
        
        Args:
            image_bytes: Image file bytes
            clothing_description: Clothing description
            character_name: Optional character name
            num_inference_steps: Optional inference steps
            guidance_scale: Optional guidance scale
            strength: Optional strength
            
        Returns:
            List of error messages (empty if valid)
        """
        errors = []
        
        # Validate image
        if image_bytes:
            is_valid, error = ImageValidator.validate_image_file(image_bytes)
            if not is_valid:
                errors.append(f"Image: {error}")
        else:
            errors.append(f"Image: {ERROR_IMAGE_NOT_PROVIDED}")
        
        # Validate clothing description
        if clothing_description:
            is_valid, error = TextValidator.validate_clothing_description(clothing_description)
            if not is_valid:
                errors.append(f"Clothing description: {error}")
        else:
            errors.append(f"Clothing description: {ERROR_INVALID_DESCRIPTION}")
        
        # Validate character name (optional)
        if character_name:
            is_valid, error = TextValidator.validate_character_name(character_name)
            if not is_valid:
                errors.append(f"Character name: {error}")
        
        # Validate parameters
        if num_inference_steps is not None:
            is_valid, error = ParameterValidator.validate_num_inference_steps(num_inference_steps)
            if not is_valid:
                errors.append(f"Inference steps: {error}")
        
        if guidance_scale is not None:
            is_valid, error = ParameterValidator.validate_guidance_scale(guidance_scale)
            if not is_valid:
                errors.append(f"Guidance scale: {error}")
        
        if strength is not None:
            is_valid, error = ParameterValidator.validate_strength(strength)
            if not is_valid:
                errors.append(f"Strength: {error}")
        
        return errors

