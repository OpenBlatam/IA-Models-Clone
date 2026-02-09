"""
Image Processor

Service for processing and preparing images for ML models.
"""

import logging
from typing import Optional, Tuple
import numpy as np
import cv2
from pathlib import Path
import base64
from io import BytesIO
from PIL import Image

logger = logging.getLogger(__name__)


class ImageProcessor:
    """
    Service for image processing operations.
    
    Handles loading, preprocessing, and format conversion.
    """
    
    def __init__(self):
        """Initialize image processor."""
        self.default_size = (224, 224)
        self.default_channels = 3
    
    def load_image(
        self,
        image_data: any,
        image_format: str = "numpy"
    ) -> Tuple[np.ndarray, dict]:
        """
        Load image from various formats.
        
        Args:
            image_data: Image data (numpy array, bytes, file path, base64)
            image_format: Format type ('numpy', 'bytes', 'file_path', 'base64')
        
        Returns:
            Tuple of (image array, metadata)
        
        Raises:
            ValueError: If format is invalid or image cannot be loaded
        """
        try:
            if image_format == "numpy":
                if not isinstance(image_data, np.ndarray):
                    raise ValueError("Expected numpy array for 'numpy' format")
                image = image_data
                metadata = self._get_image_metadata(image)
            
            elif image_format == "bytes":
                image = self._load_from_bytes(image_data)
                metadata = self._get_image_metadata(image)
            
            elif image_format == "file_path":
                image = self._load_from_file(image_data)
                metadata = self._get_image_metadata(image)
            
            elif image_format == "base64":
                image = self._load_from_base64(image_data)
                metadata = self._get_image_metadata(image)
            
            else:
                raise ValueError(f"Unsupported image format: {image_format}")
            
            # Validate image
            if image.size == 0:
                raise ValueError("Image is empty")
            
            return image, metadata
        
        except Exception as e:
            logger.error(f"Failed to load image: {str(e)}", exc_info=True)
            raise ValueError(f"Failed to load image: {str(e)}")
    
    def _load_from_bytes(self, image_bytes: bytes) -> np.ndarray:
        """Load image from bytes."""
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Failed to decode image from bytes")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    def _load_from_file(self, file_path: str) -> np.ndarray:
        """Load image from file path."""
        path = Path(file_path)
        if not path.exists():
            raise ValueError(f"Image file not found: {file_path}")
        
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Failed to load image from: {file_path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    def _load_from_base64(self, base64_string: str) -> np.ndarray:
        """Load image from base64 string."""
        try:
            # Remove data URL prefix if present
            if ',' in base64_string:
                base64_string = base64_string.split(',')[1]
            
            image_bytes = base64.b64decode(base64_string)
            return self._load_from_bytes(image_bytes)
        except Exception as e:
            raise ValueError(f"Failed to decode base64 image: {str(e)}")
    
    def _get_image_metadata(self, image: np.ndarray) -> dict:
        """Get metadata from image."""
        height, width = image.shape[:2]
        channels = image.shape[2] if len(image.shape) == 3 else 1
        
        return {
            "width": width,
            "height": height,
            "channels": channels,
            "total_pixels": width * height,
        }
    
    def preprocess_for_model(
        self,
        image: np.ndarray,
        target_size: Optional[Tuple[int, int]] = None,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Preprocess image for ML model input.
        
        Args:
            image: Input image
            target_size: Target size (width, height)
            normalize: Whether to normalize to [0, 1]
        
        Returns:
            Preprocessed image
        """
        target_size = target_size or self.default_size
        
        # Resize
        resized = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
        
        # Normalize if requested
        if normalize:
            resized = resized.astype(np.float32) / 255.0
        
        return resized
    
    def resize_image(
        self,
        image: np.ndarray,
        size: Tuple[int, int],
        maintain_aspect: bool = False
    ) -> np.ndarray:
        """
        Resize image.
        
        Args:
            image: Input image
            size: Target size (width, height)
            maintain_aspect: Whether to maintain aspect ratio
        
        Returns:
            Resized image
        """
        if maintain_aspect:
            h, w = image.shape[:2]
            target_w, target_h = size
            
            # Calculate scaling factor
            scale = min(target_w / w, target_h / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            resized = cv2.resize(image, (new_w, new_h))
            
            # Pad if necessary
            if new_w != target_w or new_h != target_h:
                pad_w = (target_w - new_w) // 2
                pad_h = (target_h - new_h) // 2
                resized = cv2.copyMakeBorder(
                    resized, pad_h, target_h - new_h - pad_h,
                    pad_w, target_w - new_w - pad_w,
                    cv2.BORDER_CONSTANT, value=[0, 0, 0]
                )
            return resized
        else:
            return cv2.resize(image, size)
    
    def convert_to_base64(
        self,
        image: np.ndarray,
        format: str = "PNG"
    ) -> str:
        """
        Convert image to base64 string.
        
        Args:
            image: Image array
            format: Image format ('PNG', 'JPEG')
        
        Returns:
            Base64 encoded string
        """
        # Convert BGR to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Assume it's already RGB or convert from BGR
            pil_image = Image.fromarray(image.astype(np.uint8))
        else:
            pil_image = Image.fromarray(image)
        
        buffer = BytesIO()
        pil_image.save(buffer, format=format)
        img_bytes = buffer.getvalue()
        
        return base64.b64encode(img_bytes).decode('utf-8')



