"""
Image Saver Utility
===================

Utility for saving images to various locations.
"""

import logging
from pathlib import Path
from typing import Optional
from PIL import Image
import tempfile
import uuid

logger = logging.getLogger(__name__)


class ImageSaver:
    """Handles saving images to various locations."""
    
    @staticmethod
    def save_temp(
        image: Image.Image,
        prefix: str = "clothing_change",
        suffix: Optional[str] = None,
        extension: str = "png"
    ) -> Path:
        """
        Save image to temporary location.
        
        Args:
            image: Image to save
            prefix: Filename prefix
            suffix: Optional suffix (e.g., character name)
            extension: File extension (default: png)
            
        Returns:
            Path to saved image
        """
        temp_dir = Path(tempfile.gettempdir())
        suffix_str = f"_{suffix}" if suffix else ""
        unique_id = str(uuid.uuid4())[:8]
        filename = f"{prefix}{suffix_str}_{unique_id}.{extension}"
        temp_path = temp_dir / filename
        
        try:
            image.save(temp_path, format=extension.upper())
            logger.debug(f"Saved temporary image to: {temp_path}")
            return temp_path
        except Exception as e:
            logger.error(f"Error saving temporary image: {e}")
            raise RuntimeError(f"Cannot save image to {temp_path}: {e}")
    
    @staticmethod
    def save(
        image: Image.Image,
        output_path: Union[str, Path],
        format: Optional[str] = None
    ) -> Path:
        """
        Save image to specified path.
        
        Args:
            image: Image to save
            output_path: Destination path
            format: Optional format override
            
        Returns:
            Path to saved image
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Determine format
        if format is None:
            format = path.suffix[1:].upper() if path.suffix else "PNG"
        
        try:
            image.save(path, format=format)
            logger.debug(f"Saved image to: {path}")
            return path
        except Exception as e:
            logger.error(f"Error saving image to {path}: {e}")
            raise RuntimeError(f"Cannot save image to {path}: {e}")

