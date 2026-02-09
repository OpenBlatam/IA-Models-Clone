"""
Mask Processor
==============

Advanced mask processing and handling for clothing change operations.
"""

import logging
from typing import Optional, Union
from PIL import Image
import numpy as np

from .mask_generator import MaskGenerator

logger = logging.getLogger(__name__)


class MaskProcessor:
    """Advanced mask processing and handling."""
    
    def __init__(self):
        """Initialize mask processor."""
        self.mask_generator = MaskGenerator()
    
    def prepare_mask(
        self,
        image: Image.Image,
        mask: Optional[Union[Image.Image, np.ndarray]] = None,
        use_smart_detection: bool = True,
    ) -> Image.Image:
        """
        Prepare mask for inpainting operation.
        
        Args:
            image: Input image
            mask: Optional mask (if None, will auto-generate)
            use_smart_detection: Use smart mask generation
            
        Returns:
            Prepared mask image
        """
        if mask is None:
            # Auto-generate mask
            if use_smart_detection:
                try:
                    mask_image = self.mask_generator.generate_smart_mask(image)
                    mask_image = self.mask_generator.refine_mask(mask_image, image)
                    logger.debug("Smart mask generated successfully")
                except Exception as e:
                    logger.warning(f"Smart mask generation failed: {e}, using simple mask")
                    mask_image = self.mask_generator.generate_simple_mask(image)
            else:
                mask_image = self.mask_generator.generate_simple_mask(image)
        else:
            # Process provided mask
            mask_image = self._process_provided_mask(mask, image)
        
        # Ensure mask matches image size
        if mask_image.size != image.size:
            mask_image = mask_image.resize(image.size, Image.Resampling.LANCZOS)
            logger.debug(f"Resized mask to match image size: {image.size}")
        
        return mask_image
    
    def _process_provided_mask(
        self,
        mask: Union[Image.Image, np.ndarray],
        reference_image: Image.Image,
    ) -> Image.Image:
        """
        Process a provided mask.
        
        Args:
            mask: Mask in various formats
            reference_image: Reference image for size matching
            
        Returns:
            Processed mask image
        """
        if isinstance(mask, np.ndarray):
            mask_image = Image.fromarray(mask).convert("L")
        else:
            mask_image = mask.convert("L")
        
        return mask_image
    
    def validate_mask(
        self,
        mask: Image.Image,
        image: Image.Image,
        min_coverage: float = 0.1,
        max_coverage: float = 0.9,
    ) -> tuple[bool, str]:
        """
        Validate mask quality and coverage.
        
        Args:
            mask: Mask image
            image: Reference image
            min_coverage: Minimum coverage ratio
            max_coverage: Maximum coverage ratio
            
        Returns:
            Tuple of (is_valid, message)
        """
        mask_array = np.array(mask)
        image_array = np.array(image)
        
        # Calculate coverage
        mask_pixels = np.sum(mask_array > 0)
        total_pixels = mask_array.size
        coverage = mask_pixels / total_pixels
        
        if coverage < min_coverage:
            return False, f"Mask coverage too low: {coverage:.2%} < {min_coverage:.2%}"
        
        if coverage > max_coverage:
            return False, f"Mask coverage too high: {coverage:.2%} > {max_coverage:.2%}"
        
        # Check size match
        if mask.size != image.size:
            return False, f"Mask size {mask.size} doesn't match image size {image.size}"
        
        return True, f"Mask valid with coverage: {coverage:.2%}"


