"""
Input Validators for Autonomous SAM3 Agent
==========================================

Validation utilities for ensuring input quality and preventing errors.
"""

import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from PIL import Image

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


class ImageValidator:
    """Validator for image inputs."""
    
    # Supported image formats
    SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp', '.tiff'}
    
    # Maximum image size (in MB)
    MAX_IMAGE_SIZE_MB = 50
    
    # Maximum dimensions (pixels)
    MAX_DIMENSION = 10000
    
    @staticmethod
    def validate_image_path(image_path: str) -> Tuple[bool, Optional[str]]:
        """
        Validate image file path and existence.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not image_path:
            return False, "Image path is required"
        
        if not isinstance(image_path, str):
            return False, "Image path must be a string"
        
        path = Path(image_path)
        
        # Check if file exists
        if not path.exists():
            return False, f"Image file not found: {image_path}"
        
        # Check if it's a file
        if not path.is_file():
            return False, f"Path is not a file: {image_path}"
        
        # Check file extension
        ext = path.suffix.lower()
        if ext not in ImageValidator.SUPPORTED_FORMATS:
            return False, f"Unsupported image format: {ext}. Supported: {ImageValidator.SUPPORTED_FORMATS}"
        
        # Check file size
        file_size_mb = path.stat().st_size / (1024 * 1024)
        if file_size_mb > ImageValidator.MAX_IMAGE_SIZE_MB:
            return False, f"Image file too large: {file_size_mb:.2f}MB (max: {ImageValidator.MAX_IMAGE_SIZE_MB}MB)"
        
        # Try to open and validate image
        try:
            with Image.open(path) as img:
                width, height = img.size
                
                # Check dimensions
                if width > ImageValidator.MAX_DIMENSION or height > ImageValidator.MAX_DIMENSION:
                    return False, f"Image dimensions too large: {width}x{height} (max: {ImageValidator.MAX_DIMENSION})"
                
                # Verify image can be loaded
                img.verify()
                
        except Exception as e:
            return False, f"Invalid image file: {str(e)}"
        
        return True, None
    
    @staticmethod
    def validate_image(image_path: str) -> None:
        """
        Validate image and raise ValidationError if invalid.
        
        Args:
            image_path: Path to image file
            
        Raises:
            ValidationError: If image is invalid
        """
        is_valid, error_msg = ImageValidator.validate_image_path(image_path)
        if not is_valid:
            raise ValidationError(error_msg)


class PromptValidator:
    """Validator for text prompts."""
    
    MIN_LENGTH = 1
    MAX_LENGTH = 1000
    
    @staticmethod
    def validate_prompt(text_prompt: str) -> Tuple[bool, Optional[str]]:
        """
        Validate text prompt.
        
        Args:
            text_prompt: Text prompt to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not text_prompt:
            return False, "Text prompt is required"
        
        if not isinstance(text_prompt, str):
            return False, "Text prompt must be a string"
        
        # Check length
        prompt_length = len(text_prompt.strip())
        if prompt_length < PromptValidator.MIN_LENGTH:
            return False, f"Text prompt too short (min: {PromptValidator.MIN_LENGTH} characters)"
        
        if prompt_length > PromptValidator.MAX_LENGTH:
            return False, f"Text prompt too long (max: {PromptValidator.MAX_LENGTH} characters)"
        
        # Check for potentially harmful content (basic check)
        if text_prompt.strip() == "":
            return False, "Text prompt cannot be empty or whitespace only"
        
        return True, None
    
    @staticmethod
    def validate_prompt_raise(text_prompt: str) -> None:
        """
        Validate prompt and raise ValidationError if invalid.
        
        Args:
            text_prompt: Text prompt to validate
            
        Raises:
            ValidationError: If prompt is invalid
        """
        is_valid, error_msg = PromptValidator.validate_prompt(text_prompt)
        if not is_valid:
            raise ValidationError(error_msg)


class TaskValidator:
    """Validator for task inputs."""
    
    @staticmethod
    def validate_task_inputs(
        image_path: str,
        text_prompt: str,
        priority: int = 0
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate all task inputs.
        
        Args:
            image_path: Path to image file
            text_prompt: Text prompt for segmentation
            priority: Task priority
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Validate image
        img_valid, img_error = ImageValidator.validate_image_path(image_path)
        if not img_valid:
            return False, f"Image validation failed: {img_error}"
        
        # Validate prompt
        prompt_valid, prompt_error = PromptValidator.validate_prompt(text_prompt)
        if not prompt_valid:
            return False, f"Prompt validation failed: {prompt_error}"
        
        # Validate priority
        if not isinstance(priority, int):
            return False, "Priority must be an integer"
        
        if priority < -100 or priority > 100:
            return False, "Priority must be between -100 and 100"
        
        return True, None
    
    @staticmethod
    def validate_task_inputs_raise(
        image_path: str,
        text_prompt: str,
        priority: int = 0
    ) -> None:
        """
        Validate task inputs and raise ValidationError if invalid.
        
        Args:
            image_path: Path to image file
            text_prompt: Text prompt for segmentation
            priority: Task priority
            
        Raises:
            ValidationError: If any input is invalid
        """
        is_valid, error_msg = TaskValidator.validate_task_inputs(
            image_path, text_prompt, priority
        )
        if not is_valid:
            raise ValidationError(error_msg)
