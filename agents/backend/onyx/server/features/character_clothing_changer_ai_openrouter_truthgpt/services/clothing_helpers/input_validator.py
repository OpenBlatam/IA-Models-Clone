"""
Input Validator
===============
Validates input parameters for clothing change operations
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class InputValidator:
    """
    Validates input parameters for clothing change operations.
    """
    
    @staticmethod
    def validate_clothing_change_inputs(
        image_url: str,
        clothing_description: str,
        guidance_scale: float,
        num_steps: int
    ) -> None:
        """
        Validate input parameters for clothing change.
        
        Args:
            image_url: URL or path to input image
            clothing_description: Description of desired clothing
            guidance_scale: Guidance scale value
            num_steps: Number of inference steps
            
        Raises:
            ValueError: If any parameter is invalid
        """
        if not image_url or not image_url.strip():
            raise ValueError("image_url is required and cannot be empty")
        
        if not clothing_description or not clothing_description.strip():
            raise ValueError("clothing_description is required and cannot be empty")
        
        if not (1.0 <= guidance_scale <= 100.0):
            raise ValueError(
                f"guidance_scale must be between 1.0 and 100.0, got {guidance_scale}"
            )
        
        if not (1 <= num_steps <= 100):
            raise ValueError(
                f"num_steps must be between 1 and 100, got {num_steps}"
            )
    
    @staticmethod
    def validate_face_swap_inputs(
        image_url: str,
        face_url: str,
        guidance_scale: float,
        num_steps: int
    ) -> None:
        """
        Validate input parameters for face swap.
        
        Args:
            image_url: URL or path to input image
            face_url: URL or path to face image
            guidance_scale: Guidance scale value
            num_steps: Number of inference steps
            
        Raises:
            ValueError: If any parameter is invalid
        """
        if not image_url or not image_url.strip():
            raise ValueError("image_url is required and cannot be empty")
        
        if not face_url or not face_url.strip():
            raise ValueError("face_url is required and cannot be empty")
        
        if not (1.0 <= guidance_scale <= 100.0):
            raise ValueError(
                f"guidance_scale must be between 1.0 and 100.0, got {guidance_scale}"
            )
        
        if not (1 <= num_steps <= 100):
            raise ValueError(
                f"num_steps must be between 1 and 100, got {num_steps}"
            )
    
    @staticmethod
    def validate_prompt_id(prompt_id: Optional[str]) -> None:
        """
        Validate prompt ID.
        
        Args:
            prompt_id: Prompt ID to validate
            
        Raises:
            ValueError: If prompt_id is invalid
        """
        if not prompt_id or not prompt_id.strip():
            raise ValueError("prompt_id is required and cannot be empty")

