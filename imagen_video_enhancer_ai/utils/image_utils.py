"""
Image utilities for Imagen Video Enhancer AI
=============================================

Image processing and analysis utilities.
"""

import logging
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


def get_image_info(image_path: str) -> Dict[str, Any]:
    """
    Get basic image information.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Dictionary with image information
    """
    path = Path(image_path)
    file_size = path.stat().st_size / (1024 * 1024)  # MB
    
    info = {
        "file_path": str(path),
        "file_size_mb": file_size,
        "format": path.suffix.lower(),
    }
    
    # Try to get image dimensions if Pillow is available
    try:
        from PIL import Image
        
        with Image.open(image_path) as img:
            info.update({
                "width": img.width,
                "height": img.height,
                "mode": img.mode,
                "format": img.format or path.suffix.lower(),
                "has_transparency": img.mode in ("RGBA", "LA", "P") and "transparency" in img.info
            })
    except ImportError:
        logger.warning("Pillow not available, using basic info only")
    except Exception as e:
        logger.warning(f"Could not read image info: {e}")
    
    return info


def validate_image_dimensions(
    image_path: str,
    max_width: int = 8192,
    max_height: int = 8192
) -> Tuple[bool, Optional[str]]:
    """
    Validate image dimensions.
    
    Args:
        image_path: Path to image file
        max_width: Maximum width
        max_height: Maximum height
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        from PIL import Image
        
        with Image.open(image_path) as img:
            if img.width > max_width or img.height > max_height:
                return False, f"Image dimensions {img.width}x{img.height} exceed maximum {max_width}x{max_height}"
            
            return True, None
            
    except ImportError:
        logger.warning("Pillow not available, skipping dimension validation")
        return True, None
    except Exception as e:
        return False, f"Error validating dimensions: {e}"


def estimate_enhancement_time(
    file_size_mb: float,
    enhancement_type: str,
    is_video: bool = False
) -> float:
    """
    Estimate enhancement processing time in seconds.
    
    Args:
        file_size_mb: File size in MB
        enhancement_type: Type of enhancement
        is_video: Whether it's a video file
        
    Returns:
        Estimated time in seconds
    """
    # Base time per MB
    base_time_per_mb = 2.0 if is_video else 0.5
    
    # Multipliers by enhancement type
    multipliers = {
        "general": 1.0,
        "sharpness": 1.2,
        "colors": 1.1,
        "denoise": 1.5,
        "upscale": 2.0,
        "restore": 2.5,
    }
    
    multiplier = multipliers.get(enhancement_type, 1.0)
    
    estimated_time = file_size_mb * base_time_per_mb * multiplier
    
    # Minimum time
    return max(estimated_time, 5.0)




