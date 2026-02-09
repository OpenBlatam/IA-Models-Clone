"""Image processing utilities."""

import aiohttp
from io import BytesIO
from pathlib import Path
from PIL import Image, ImageOps
from typing import Optional
import aiofiles

from config.settings import settings
from core.exceptions import ImageProcessingError, ImageValidationError
from core.constants import MIN_IMAGE_DIMENSION, MAX_IMAGE_DIMENSION
from core.interfaces import IImageProcessor
from utils.image_utils import resize_image, optimize_image_quality
from utils.network_utils import validate_image_url
from utils.retry import retry_async
from utils.logger import get_logger

logger = get_logger(__name__)


class ImageProcessor(IImageProcessor):
    """Handles image loading, validation, and saving."""
    
    def __init__(self):
        self.max_size_mb = settings.max_image_size_mb
        self.supported_formats = settings.supported_formats
    
    async def load_from_bytes(self, image_data: bytes) -> Image.Image:
        """
        Load image from bytes.
        
        Args:
            image_data: Image data as bytes
            
        Returns:
            PIL Image object
            
        Raises:
            ImageValidationError: If image is invalid
            ImageProcessingError: If processing fails
        """
        try:
            # Validate size
            size_mb = len(image_data) / (1024 * 1024)
            if size_mb > self.max_size_mb:
                raise ImageValidationError(
                    f"Image size ({size_mb:.2f}MB) exceeds maximum ({self.max_size_mb}MB)"
                )
            
            # Load image
            image = Image.open(BytesIO(image_data))
            
            # Validate image
            self.validate_image(image)
            
            # Normalize to RGB and auto-orient
            image = image.convert("RGB")
            image = ImageOps.exif_transpose(image)  # Fix orientation
            
            return image
        except ImageValidationError:
            raise
        except Exception as e:
            logger.error(f"Error loading image from bytes: {e}")
            raise ImageProcessingError(f"Failed to load image: {str(e)}")
    
    @retry_async(max_attempts=3, initial_wait=1.0, retry_on=(ImageProcessingError,))
    async def load_from_url(self, image_url: str) -> Image.Image:
        """
        Load image from URL.
        
        Args:
            image_url: URL of the image
            
        Returns:
            PIL Image object
            
        Raises:
            ImageProcessingError: If fetching or processing fails
            ImageValidationError: If URL is invalid
        """
        # Validate URL first
        validate_image_url(image_url)
        
        try:
            timeout = aiohttp.ClientTimeout(total=30)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(image_url) as response:
                    if response.status != 200:
                        raise ImageProcessingError(
                            f"Failed to fetch image from URL: HTTP {response.status}"
                        )
                    
                    image_data = await response.read()
                    return await self.load_from_bytes(image_data)
        except ImageValidationError:
            raise
        except aiohttp.ClientError as e:
            logger.error(f"Error fetching image from URL: {e}")
            raise ImageProcessingError(f"Failed to fetch image: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error loading image from URL: {e}")
            raise ImageProcessingError(f"Failed to load image: {str(e)}")
    
    async def save_image(self, image: Image.Image, output_path: Path) -> None:
        """
        Save image to file.
        
        Args:
            image: PIL Image object
            output_path: Path to save the image
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save image
        image.save(output_path, format=settings.output_format.upper())
        logger.info(f"Saved image to {output_path}")
    
    def validate_image(self, image: Image.Image) -> bool:
        """
        Validate image format and properties.
        
        Args:
            image: PIL Image object
            
        Returns:
            True if valid
            
        Raises:
            ImageValidationError: If image is invalid
        """
        if not image:
            raise ImageValidationError("Image is None or empty")
        
        if image.format and image.format.lower() not in self.supported_formats:
            raise ImageValidationError(
                f"Unsupported image format: {image.format}. "
                f"Supported formats: {', '.join(self.supported_formats)}"
            )
        
        # Check dimensions
        width, height = image.size
        if width < MIN_IMAGE_DIMENSION or height < MIN_IMAGE_DIMENSION:
            raise ImageValidationError(
                f"Image dimensions too small ({width}x{height}). "
                f"Minimum size: {MIN_IMAGE_DIMENSION}x{MIN_IMAGE_DIMENSION} pixels"
            )
        
        if width > MAX_IMAGE_DIMENSION or height > MAX_IMAGE_DIMENSION:
            raise ImageValidationError(
                f"Image dimensions too large ({width}x{height}). "
                f"Maximum size: {MAX_IMAGE_DIMENSION}x{MAX_IMAGE_DIMENSION} pixels"
            )
        
        # Check if image is corrupted
        try:
            image.verify()
        except Exception as e:
            raise ImageValidationError(f"Image appears to be corrupted: {str(e)}")
        
        return True

