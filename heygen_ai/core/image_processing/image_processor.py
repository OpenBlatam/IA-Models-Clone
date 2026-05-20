"""
Image Processor
===============

Utility class for image processing operations.
"""

import logging
from typing import Any, Optional

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class ImageProcessor:
    """Utility class for image processing operations."""
    
    @staticmethod
    def post_process_image(
        image: Image.Image,
        config: Any,
        face_service: Optional[Any] = None,
    ) -> Image.Image:
        """Post-process generated image.
        
        Args:
            image: Generated PIL Image
            config: Generation configuration (must have quality, enable_expressions)
            face_service: Optional face processing service
        
        Returns:
            Post-processed image
        """
        try:
            # Convert to numpy array
            img_array = np.array(image)
            
            # Face enhancement if available
            if face_service and hasattr(config, 'enable_expressions') and config.enable_expressions:
                img_array = face_service.enhance_face(img_array)
            
            # Quality enhancements
            if hasattr(config, 'quality'):
                from shared.enums import AvatarQuality
                if config.quality in [AvatarQuality.HIGH, AvatarQuality.ULTRA]:
                    img_array = ImageProcessor._apply_quality_enhancements(img_array)
            
            # Convert back to PIL Image
            return Image.fromarray(img_array)
            
        except Exception as e:
            logging.warning(f"Post-processing failed: {e}")
            return image
    
    @staticmethod
    def _apply_quality_enhancements(img_array: np.ndarray) -> np.ndarray:
        """Apply quality enhancements to image.
        
        Args:
            img_array: Image as numpy array
        
        Returns:
            Enhanced image array
        """
        try:
            # Apply noise reduction
            enhanced = cv2.fastNlMeansDenoisingColored(
                img_array, None, 10, 10, 7, 21
            )
            
            # Apply subtle sharpening
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            enhanced = cv2.filter2D(enhanced, -1, kernel)
            
            return enhanced
            
        except Exception as e:
            logging.warning(f"Quality enhancement failed: {e}")
            return img_array

