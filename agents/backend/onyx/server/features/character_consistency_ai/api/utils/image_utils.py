"""
Image Utilities for API
=======================

Utilities for handling image uploads and processing in API endpoints.
"""

import io
import logging
from typing import List
from PIL import Image
from fastapi import UploadFile

logger = logging.getLogger(__name__)


async def process_uploaded_images(images: List[UploadFile]) -> List[Image.Image]:
    """
    Process uploaded images from FastAPI UploadFile objects.
    
    Args:
        images: List of UploadFile objects
        
    Returns:
        List of PIL Image objects in RGB format
        
    Raises:
        ValueError: If image processing fails
    """
    if not images:
        raise ValueError("At least one image is required")
    
    image_list = []
    for image_file in images:
        pil_image = await _process_single_image(image_file)
        image_list.append(pil_image)
    
    return image_list


async def _process_single_image(image_file: UploadFile) -> Image.Image:
    """Process a single uploaded image file."""
    try:
        image_bytes = await image_file.read()
        if not image_bytes:
            raise ValueError(f"Empty image file: {image_file.filename}")
        
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        _validate_image(pil_image, image_file.filename)
        return pil_image
    except ValueError:
        raise
    except Exception as e:
        logger.error(f"Error processing image {image_file.filename}: {e}")
        raise ValueError(f"Failed to process image {image_file.filename}: {e}")


def _validate_image(image: Image.Image, filename: str) -> None:
    """Validate image dimensions and format."""
    if image.size[0] == 0 or image.size[1] == 0:
        raise ValueError(f"Invalid image dimensions in {filename}")
    if image.mode != "RGB":
        logger.warning(f"Image {filename} converted from {image.mode} to RGB")


