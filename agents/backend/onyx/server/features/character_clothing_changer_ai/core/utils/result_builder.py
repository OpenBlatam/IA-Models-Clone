"""
Result Builder
=============

Utility for building consistent result dictionaries from clothing change operations.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union
from PIL import Image

from .image_saver import ImageSaver

logger = logging.getLogger(__name__)


class ResultBuilder:
    """Builds standardized result dictionaries for clothing change operations."""
    
    @staticmethod
    def build_result(
        clothing_description: str,
        changed_image: Image.Image,
        character_name: Optional[str] = None,
        style: Optional[str] = None,
        prompt: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        metrics: Optional[Dict[str, Any]] = None,
        saved_path: Optional[Union[str, Path]] = None,
        save_tensor: bool = True,
    ) -> Dict[str, Any]:
        """
        Build a standardized result dictionary.
        
        Args:
            clothing_description: Description of clothing change
            changed_image: The modified image
            character_name: Optional character name
            style: Optional style information
            prompt: Prompt used for generation
            negative_prompt: Negative prompt used
            metrics: Quality metrics if calculated
            saved_path: Path where tensor was saved
            save_tensor: Whether tensor was saved
            
        Returns:
            Standardized result dictionary
        """
        result = {
            "clothing_description": clothing_description,
            "character_name": character_name,
            "style": style,
            "changed": True,
            "prompt_used": prompt,
            "negative_prompt_used": negative_prompt,
            "saved": bool(saved_path),
        }
        
        # Add metrics if available
        if metrics:
            result["quality_metrics"] = metrics
        
        # Add saved path if tensor was saved
        if saved_path:
            result["saved_path"] = str(saved_path)
        
        # Save image temporarily if not saved as tensor
        if not save_tensor or not saved_path:
            try:
                image_path = ImageSaver.save_temp(
                    image=changed_image,
                    prefix="clothing_change",
                    suffix=character_name
                )
                result["image_path"] = str(image_path)
            except Exception as e:
                logger.warning(f"Failed to save temporary image: {e}")
        
        return result

