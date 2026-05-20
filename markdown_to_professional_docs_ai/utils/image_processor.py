"""Image Processor - Process and embed images in documents"""
import base64
import io
import requests
from pathlib import Path
from typing import Optional, Dict, Any
from PIL import Image
import logging

logger = logging.getLogger(__name__)


class ImageProcessor:
    """Process images for embedding in documents"""
    
    def __init__(self, max_size_mb: int = 5, max_dimension: int = 2000):
        """
        Initialize image processor
        
        Args:
            max_size_mb: Maximum image size in MB
            max_dimension: Maximum dimension (width or height) in pixels
        """
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.max_dimension = max_dimension
    
    def download_image(self, url: str, timeout: int = 10) -> Optional[bytes]:
        """
        Download image from URL
        
        Args:
            url: Image URL
            timeout: Request timeout in seconds
            
        Returns:
            Image bytes or None
        """
        try:
            response = requests.get(url, timeout=timeout, stream=True)
            response.raise_for_status()
            
            # Check content type
            content_type = response.headers.get('content-type', '')
            if not content_type.startswith('image/'):
                logger.warning(f"URL does not appear to be an image: {url}")
                return None
            
            # Read image
            image_data = response.content
            
            # Check size
            if len(image_data) > self.max_size_bytes:
                logger.warning(f"Image too large: {len(image_data)} bytes")
                return None
            
            return image_data
        except Exception as e:
            logger.error(f"Error downloading image {url}: {e}")
            return None
    
    def process_image(
        self,
        image_data: bytes,
        max_width: Optional[int] = None,
        max_height: Optional[int] = None,
        quality: int = 85
    ) -> Optional[bytes]:
        """
        Process and optimize image
        
        Args:
            image_data: Raw image bytes
            max_width: Maximum width (None for auto)
            max_height: Maximum height (None for auto)
            quality: JPEG quality (1-100)
            
        Returns:
            Processed image bytes or None
        """
        try:
            # Open image
            img = Image.open(io.BytesIO(image_data))
            
            # Get format
            original_format = img.format
            
            # Resize if needed
            if max_width or max_height:
                img.thumbnail(
                    (max_width or self.max_dimension, max_height or self.max_dimension),
                    Image.Resampling.LANCZOS
                )
            
            # Convert to RGB if needed (for JPEG)
            if original_format == 'JPEG' and img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Save to bytes
            output = io.BytesIO()
            
            if original_format == 'JPEG' or original_format == 'JPG':
                img.save(output, format='JPEG', quality=quality, optimize=True)
            elif original_format == 'PNG':
                img.save(output, format='PNG', optimize=True)
            else:
                # Convert to JPEG
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img.save(output, format='JPEG', quality=quality, optimize=True)
            
            return output.getvalue()
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            return None
    
    def image_to_base64(self, image_data: bytes) -> str:
        """Convert image to base64 string"""
        return base64.b64encode(image_data).decode('utf-8')
    
    def get_image_info(self, image_data: bytes) -> Dict[str, Any]:
        """
        Get image information
        
        Args:
            image_data: Image bytes
            
        Returns:
            Dictionary with image info
        """
        try:
            img = Image.open(io.BytesIO(image_data))
            return {
                "width": img.width,
                "height": img.height,
                "format": img.format,
                "mode": img.mode,
                "size_bytes": len(image_data)
            }
        except Exception as e:
            logger.error(f"Error getting image info: {e}")
            return {}

