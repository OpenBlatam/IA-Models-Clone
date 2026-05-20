"""
Format Converter
================

Convert between different formats and representations.
"""

import logging
from typing import Union, Optional
from PIL import Image
import base64
import io
import numpy as np

logger = logging.getLogger(__name__)

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


class FormatConverter:
    """
    Convert between different image formats.
    
    Features:
    - PIL <-> NumPy
    - PIL <-> OpenCV
    - PIL <-> Base64
    - PIL <-> Bytes
    """
    
    @staticmethod
    def pil_to_numpy(image: Image.Image) -> np.ndarray:
        """Convert PIL Image to NumPy array."""
        return np.array(image.convert("RGB"))
    
    @staticmethod
    def numpy_to_pil(array: np.ndarray) -> Image.Image:
        """Convert NumPy array to PIL Image."""
        if len(array.shape) == 2:
            # Grayscale
            return Image.fromarray(array, mode="L")
        elif len(array.shape) == 3:
            # Color
            return Image.fromarray(array, mode="RGB")
        else:
            raise ValueError(f"Unsupported array shape: {array.shape}")
    
    @staticmethod
    def pil_to_cv2(image: Image.Image) -> np.ndarray:
        """Convert PIL Image to OpenCV format (BGR)."""
        if not CV2_AVAILABLE:
            raise ImportError("OpenCV not available")
        
        array = np.array(image.convert("RGB"))
        return cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    
    @staticmethod
    def cv2_to_pil(image_cv: np.ndarray) -> Image.Image:
        """Convert OpenCV format (BGR) to PIL Image."""
        if not CV2_AVAILABLE:
            raise ImportError("OpenCV not available")
        
        rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)
    
    @staticmethod
    def pil_to_base64(image: Image.Image, format: str = "PNG") -> str:
        """Convert PIL Image to base64 string."""
        buffer = io.BytesIO()
        image.save(buffer, format=format)
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    @staticmethod
    def base64_to_pil(base64_string: str) -> Image.Image:
        """Convert base64 string to PIL Image."""
        img_bytes = base64.b64decode(base64_string)
        return Image.open(io.BytesIO(img_bytes)).convert("RGB")
    
    @staticmethod
    def pil_to_bytes(image: Image.Image, format: str = "PNG") -> bytes:
        """Convert PIL Image to bytes."""
        buffer = io.BytesIO()
        image.save(buffer, format=format)
        return buffer.getvalue()
    
    @staticmethod
    def bytes_to_pil(image_bytes: bytes) -> Image.Image:
        """Convert bytes to PIL Image."""
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    @staticmethod
    def convert_format(
        image: Union[Image.Image, np.ndarray, str, bytes],
        target_format: str
    ) -> Union[Image.Image, np.ndarray, str, bytes]:
        """
        Convert image between formats.
        
        Args:
            image: Input image in any format
            target_format: Target format ('pil', 'numpy', 'cv2', 'base64', 'bytes')
            
        Returns:
            Image in target format
        """
        # Detect input format
        if isinstance(image, Image.Image):
            current_format = "pil"
        elif isinstance(image, np.ndarray):
            current_format = "numpy"
        elif isinstance(image, str):
            current_format = "base64"
        elif isinstance(image, bytes):
            current_format = "bytes"
        else:
            raise ValueError(f"Unsupported input format: {type(image)}")
        
        # Convert to PIL first (intermediate format)
        if current_format == "pil":
            pil_image = image
        elif current_format == "numpy":
            pil_image = FormatConverter.numpy_to_pil(image)
        elif current_format == "base64":
            pil_image = FormatConverter.base64_to_pil(image)
        elif current_format == "bytes":
            pil_image = FormatConverter.bytes_to_pil(image)
        
        # Convert to target format
        if target_format == "pil":
            return pil_image
        elif target_format == "numpy":
            return FormatConverter.pil_to_numpy(pil_image)
        elif target_format == "cv2":
            return FormatConverter.pil_to_cv2(pil_image)
        elif target_format == "base64":
            return FormatConverter.pil_to_base64(pil_image)
        elif target_format == "bytes":
            return FormatConverter.pil_to_bytes(pil_image)
        else:
            raise ValueError(f"Unsupported target format: {target_format}")


