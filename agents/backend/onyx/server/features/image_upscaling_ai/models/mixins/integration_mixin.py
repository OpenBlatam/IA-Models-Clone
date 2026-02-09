"""
Integration Mixin

Contains integration functionality for external services and APIs.
"""

import logging
from typing import Union, Dict, Any, Optional, List
from pathlib import Path
from PIL import Image
import base64
import io

logger = logging.getLogger(__name__)


class IntegrationMixin:
    """
    Mixin providing integration functionality.
    
    This mixin contains:
    - API integration
    - Webhook support
    - External service integration
    - Data format conversion
    - Integration utilities
    """
    
    def image_to_base64(
        self,
        image: Image.Image,
        format: str = "PNG"
    ) -> str:
        """
        Convert image to base64 string.
        
        Args:
            image: Input image
            format: Image format
            
        Returns:
            Base64 encoded string
        """
        buffer = io.BytesIO()
        image.save(buffer, format=format)
        img_bytes = buffer.getvalue()
        return base64.b64encode(img_bytes).decode('utf-8')
    
    def base64_to_image(self, base64_string: str) -> Image.Image:
        """
        Convert base64 string to image.
        
        Args:
            base64_string: Base64 encoded image string
            
        Returns:
            PIL Image
        """
        img_bytes = base64.b64decode(base64_string)
        return Image.open(io.BytesIO(img_bytes)).convert("RGB")
    
    def image_to_dict(
        self,
        image: Image.Image,
        include_metadata: bool = True
    ) -> Dict[str, Any]:
        """
        Convert image to dictionary format.
        
        Args:
            image: Input image
            include_metadata: Include metadata
            
        Returns:
            Dictionary with image data
        """
        result = {
            "base64": self.image_to_base64(image),
            "size": image.size,
            "mode": image.mode,
        }
        
        if include_metadata:
            from ..helpers import QualityCalculator
            quality = QualityCalculator.calculate_quality_metrics(image)
            result["metadata"] = {
                "quality": quality.overall_quality,
                "sharpness": quality.sharpness,
                "contrast": quality.contrast,
            }
        
        return result
    
    def prepare_api_request(
        self,
        image: Union[Image.Image, str, Path],
        scale_factor: float,
        method: str = "auto",
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Prepare data for API request.
        
        Args:
            image: Input image
            scale_factor: Scale factor
            method: Upscaling method
            options: Additional options
            
        Returns:
            Dictionary ready for API request
        """
        if isinstance(image, (str, Path)):
            pil_image = Image.open(image).convert("RGB")
        else:
            pil_image = image.convert("RGB")
        
        request_data = {
            "image": self.image_to_base64(pil_image),
            "scale_factor": scale_factor,
            "method": method,
            "options": options or {},
        }
        
        return request_data
    
    def process_api_response(
        self,
        response_data: Dict[str, Any]
    ) -> Image.Image:
        """
        Process API response and extract image.
        
        Args:
            response_data: API response dictionary
            
        Returns:
            PIL Image
        """
        if "image" in response_data:
            return self.base64_to_image(response_data["image"])
        elif "base64" in response_data:
            return self.base64_to_image(response_data["base64"])
        else:
            raise ValueError("No image data found in response")
    
    def create_webhook_payload(
        self,
        operation: str,
        status: str,
        result: Optional[Image.Image] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create webhook payload.
        
        Args:
            operation: Operation name
            status: Status ('success', 'error', 'processing')
            result: Optional result image
            metadata: Optional metadata
            
        Returns:
            Dictionary with webhook payload
        """
        payload = {
            "operation": operation,
            "status": status,
            "timestamp": str(Path().cwd()),  # Placeholder for timestamp
            "metadata": metadata or {},
        }
        
        if result and status == "success":
            payload["result"] = self.image_to_base64(result)
        
        return payload
    
    def batch_to_api_format(
        self,
        images: List[Union[Image.Image, str, Path]],
        scale_factor: float,
        method: str = "auto"
    ) -> Dict[str, Any]:
        """
        Convert batch of images to API format.
        
        Args:
            images: List of images
            scale_factor: Scale factor
            method: Upscaling method
            
        Returns:
            Dictionary with batch data
        """
        image_data = []
        
        for img in images:
            if isinstance(img, (str, Path)):
                pil_image = Image.open(img).convert("RGB")
            else:
                pil_image = img.convert("RGB")
            
            image_data.append(self.image_to_base64(pil_image))
        
        return {
            "images": image_data,
            "scale_factor": scale_factor,
            "method": method,
            "batch_size": len(images),
        }


