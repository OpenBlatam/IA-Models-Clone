"""
Integration Helper
==================

Helper functions for integrating with other systems.
"""

import logging
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
from PIL import Image
import base64
import io

logger = logging.getLogger(__name__)


class IntegrationHelper:
    """
    Helper for integrating with other systems.
    
    Features:
    - Format conversion
    - Base64 encoding/decoding
    - Batch processing helpers
    - Integration utilities
    """
    
    @staticmethod
    def image_to_base64(image: Image.Image, format: str = "PNG") -> str:
        """
        Convert PIL Image to base64 string.
        
        Args:
            image: PIL Image
            format: Image format (PNG, JPEG, etc.)
            
        Returns:
            Base64 encoded string
        """
        buffer = io.BytesIO()
        image.save(buffer, format=format)
        img_bytes = buffer.getvalue()
        return base64.b64encode(img_bytes).decode('utf-8')
    
    @staticmethod
    def base64_to_image(base64_string: str) -> Image.Image:
        """
        Convert base64 string to PIL Image.
        
        Args:
            base64_string: Base64 encoded image string
            
        Returns:
            PIL Image
        """
        img_bytes = base64.b64decode(base64_string)
        return Image.open(io.BytesIO(img_bytes)).convert("RGB")
    
    @staticmethod
    def image_to_bytes(image: Image.Image, format: str = "PNG") -> bytes:
        """
        Convert PIL Image to bytes.
        
        Args:
            image: PIL Image
            format: Image format
            
        Returns:
            Image bytes
        """
        buffer = io.BytesIO()
        image.save(buffer, format=format)
        return buffer.getvalue()
    
    @staticmethod
    def bytes_to_image(image_bytes: bytes) -> Image.Image:
        """
        Convert bytes to PIL Image.
        
        Args:
            image_bytes: Image bytes
            
        Returns:
            PIL Image
        """
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    @staticmethod
    def prepare_batch_request(
        images: List[Union[str, Path, Image.Image]],
        scale_factor: float,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Prepare batch request data.
        
        Args:
            images: List of images
            scale_factor: Scale factor
            **kwargs: Additional parameters
            
        Returns:
            List of request dictionaries
        """
        requests = []
        
        for idx, image in enumerate(images):
            if isinstance(image, (str, Path)):
                image_path = Path(image)
                if image_path.exists():
                    img = Image.open(image_path).convert("RGB")
                else:
                    logger.warning(f"Image not found: {image}")
                    continue
            elif isinstance(image, Image.Image):
                img = image.convert("RGB")
            else:
                logger.warning(f"Invalid image type: {type(image)}")
                continue
            
            request = {
                "image": img,
                "scale_factor": scale_factor,
                "index": idx,
                **kwargs
            }
            requests.append(request)
        
        return requests
    
    @staticmethod
    def format_response(
        result: Dict[str, Any],
        include_image: bool = False,
        image_format: str = "base64"
    ) -> Dict[str, Any]:
        """
        Format response for API or other systems.
        
        Args:
            result: Result dictionary
            include_image: Include image in response
            image_format: Format for image (base64, bytes, path)
            
        Returns:
            Formatted response
        """
        formatted = {
            "success": result.get("success", False),
            "operation_id": result.get("operation_id"),
            "original_size": result.get("original_size"),
            "upscaled_size": result.get("upscaled_size"),
            "scale_factor": result.get("scale_factor"),
            "method_used": result.get("method_used"),
            "processing_time": result.get("processing_time"),
            "quality_score": result.get("quality_score"),
        }
        
        if include_image and "upscaled_image" in result:
            upscaled = result["upscaled_image"]
            
            if image_format == "base64":
                formatted["upscaled_image"] = IntegrationHelper.image_to_base64(upscaled)
            elif image_format == "bytes":
                formatted["upscaled_image_bytes"] = IntegrationHelper.image_to_bytes(upscaled)
            elif image_format == "path":
                # Save to file and return path
                output_path = result.get("output_path", "output.png")
                upscaled.save(output_path)
                formatted["upscaled_image_path"] = output_path
        
        # Include additional data
        if "recommendation" in result:
            formatted["recommendation"] = result["recommendation"]
        
        if "analysis" in result:
            formatted["analysis"] = result["analysis"]
        
        if "quality_report" in result:
            formatted["quality_report"] = result["quality_report"]
        
        return formatted


