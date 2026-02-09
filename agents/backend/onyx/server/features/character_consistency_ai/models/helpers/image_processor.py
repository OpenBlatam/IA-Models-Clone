"""
Image Processing Helper
=======================

Helper utilities for image preprocessing and conversion.
"""

from typing import Union
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import logging

logger = logging.getLogger(__name__)


class ImageProcessor:
    """Helper class for image preprocessing operations."""
    
    @staticmethod
    def to_pil_image(image: Union[Image.Image, str, Path, np.ndarray]) -> Image.Image:
        """
        Convert various image formats to PIL Image.
        
        Args:
            image: Input image in various formats
            
        Returns:
            PIL Image in RGB format
            
        Raises:
            ValueError: If image type is not supported
        """
        if isinstance(image, (str, Path)):
            return Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            # Handle different numpy array formats
            if image.dtype != np.uint8:
                # Normalize if not uint8
                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)
                else:
                    image = image.astype(np.uint8)
            return Image.fromarray(image).convert("RGB")
        elif isinstance(image, Image.Image):
            return image.convert("RGB")
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
    
    @staticmethod
    def process_with_clip(
        image: Union[Image.Image, str, Path, np.ndarray],
        clip_processor,
        device: torch.device
    ) -> torch.Tensor:
        """
        Process image with CLIP processor.
        
        Args:
            image: Input image
            clip_processor: CLIP image processor
            device: Target device
            
        Returns:
            Preprocessed tensor
        """
        pil_image = ImageProcessor.to_pil_image(image)
        inputs = clip_processor(images=pil_image, return_tensors="pt")
        return inputs["pixel_values"].to(device)

