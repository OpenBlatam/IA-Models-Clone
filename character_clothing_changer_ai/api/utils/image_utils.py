"""
Image Utilities
==============

Utility functions for image processing and conversion.
"""

from pathlib import Path
from typing import Optional
import base64


def image_to_base64(image_path: Path) -> Optional[str]:
    """
    Convert image file to base64 data URL.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Base64 data URL string or None if file doesn't exist
    """
    if not image_path.exists():
        return None
    
    try:
        with open(image_path, "rb") as img_file:
            img_data = base64.b64encode(img_file.read()).decode()
            return f"data:image/png;base64,{img_data}"
    except Exception:
        return None


def add_image_urls_to_result(result: dict, image_path_key: str = "image_path") -> dict:
    """
    Add base64 and URL fields to result dict if image_path exists.
    
    Args:
        result: Result dictionary
        image_path_key: Key in result dict containing image path
        
    Returns:
        Updated result dictionary
    """
    if image_path_key not in result:
        return result
    
    image_path = Path(result[image_path_key])
    if image_path.exists():
        # Add base64
        base64_data = image_to_base64(image_path)
        if base64_data:
            result["image_base64"] = base64_data
        
        # Add URL
        result["image_url"] = f"/api/v1/image/{image_path.name}"
    
    return result
