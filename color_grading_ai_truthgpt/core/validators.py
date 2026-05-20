"""
Validators for Color Grading AI
================================

Advanced validation for parameters, files, and configurations.
"""

import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
import mimetypes

from .exceptions import InvalidParametersError, MediaNotFoundError

logger = logging.getLogger(__name__)


class ParameterValidator:
    """Validates color grading parameters."""
    
    @staticmethod
    def validate_color_params(params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate color grading parameters.
        
        Args:
            params: Color parameters dictionary
            
        Returns:
            Validated parameters
            
        Raises:
            InvalidParametersError: If parameters are invalid
        """
        validated = {}
        
        # Brightness: -1.0 to 1.0
        if "brightness" in params:
            brightness = float(params["brightness"])
            if not -1.0 <= brightness <= 1.0:
                raise InvalidParametersError(f"Brightness must be between -1.0 and 1.0, got {brightness}")
            validated["brightness"] = brightness
        
        # Contrast: 0.0 to 3.0
        if "contrast" in params:
            contrast = float(params["contrast"])
            if not 0.0 <= contrast <= 3.0:
                raise InvalidParametersError(f"Contrast must be between 0.0 and 3.0, got {contrast}")
            validated["contrast"] = contrast
        
        # Saturation: 0.0 to 3.0
        if "saturation" in params:
            saturation = float(params["saturation"])
            if not 0.0 <= saturation <= 3.0:
                raise InvalidParametersError(f"Saturation must be between 0.0 and 3.0, got {saturation}")
            validated["saturation"] = saturation
        
        # Color balance: -0.5 to 0.5 for each channel
        if "color_balance" in params:
            balance = params["color_balance"]
            if not isinstance(balance, dict):
                raise InvalidParametersError("color_balance must be a dictionary")
            
            validated_balance = {}
            for channel in ["r", "g", "b"]:
                if channel in balance:
                    value = float(balance[channel])
                    if not -0.5 <= value <= 0.5:
                        raise InvalidParametersError(
                            f"Color balance {channel} must be between -0.5 and 0.5, got {value}"
                        )
                    validated_balance[channel] = value
                else:
                    validated_balance[channel] = 0.0
            
            validated["color_balance"] = validated_balance
        
        return validated
    
    @staticmethod
    def validate_template_name(template_name: str, available_templates: List[str]) -> str:
        """
        Validate template name.
        
        Args:
            template_name: Template name to validate
            available_templates: List of available template names
            
        Returns:
            Validated template name
            
        Raises:
            InvalidParametersError: If template not found
        """
        if template_name not in available_templates:
            raise InvalidParametersError(
                f"Template '{template_name}' not found. Available: {', '.join(available_templates)}"
            )
        return template_name


class MediaValidator:
    """Validates media files."""
    
    SUPPORTED_IMAGE_FORMATS = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]
    SUPPORTED_VIDEO_FORMATS = [".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv"]
    
    @staticmethod
    def validate_file_path(file_path: str, must_exist: bool = True) -> Path:
        """
        Validate file path.
        
        Args:
            file_path: Path to file
            must_exist: Whether file must exist
            
        Returns:
            Path object
            
        Raises:
            MediaNotFoundError: If file not found
        """
        path = Path(file_path)
        
        if must_exist and not path.exists():
            raise MediaNotFoundError(f"File not found: {file_path}")
        
        if must_exist and not path.is_file():
            raise MediaNotFoundError(f"Path is not a file: {file_path}")
        
        return path
    
    @staticmethod
    def validate_image_file(file_path: str) -> Path:
        """
        Validate image file.
        
        Args:
            file_path: Path to image file
            
        Returns:
            Path object
            
        Raises:
            MediaNotFoundError: If file is invalid
        """
        path = MediaValidator.validate_file_path(file_path)
        
        if path.suffix.lower() not in MediaValidator.SUPPORTED_IMAGE_FORMATS:
            raise MediaNotFoundError(
                f"Unsupported image format: {path.suffix}. "
                f"Supported: {', '.join(MediaValidator.SUPPORTED_IMAGE_FORMATS)}"
            )
        
        return path
    
    @staticmethod
    def validate_video_file(file_path: str) -> Path:
        """
        Validate video file.
        
        Args:
            file_path: Path to video file
            
        Returns:
            Path object
            
        Raises:
            MediaNotFoundError: If file is invalid
        """
        path = MediaValidator.validate_file_path(file_path)
        
        if path.suffix.lower() not in MediaValidator.SUPPORTED_VIDEO_FORMATS:
            raise MediaNotFoundError(
                f"Unsupported video format: {path.suffix}. "
                f"Supported: {', '.join(MediaValidator.SUPPORTED_VIDEO_FORMATS)}"
            )
        
        return path
    
    @staticmethod
    def get_file_size_mb(file_path: str) -> float:
        """Get file size in MB."""
        path = Path(file_path)
        if not path.exists():
            return 0.0
        return path.stat().st_size / (1024 * 1024)
    
    @staticmethod
    def validate_file_size(file_path: str, max_size_mb: float = 5000.0) -> bool:
        """
        Validate file size.
        
        Args:
            file_path: Path to file
            max_size_mb: Maximum size in MB
            
        Returns:
            True if valid
            
        Raises:
            InvalidParametersError: If file too large
        """
        size_mb = MediaValidator.get_file_size_mb(file_path)
        if size_mb > max_size_mb:
            raise InvalidParametersError(
                f"File too large: {size_mb:.2f}MB. Maximum: {max_size_mb}MB"
            )
        return True


class ConfigValidator:
    """Validates configuration."""
    
    @staticmethod
    def validate_config(config: Any) -> bool:
        """
        Validate configuration.
        
        Args:
            config: Configuration object
            
        Returns:
            True if valid
            
        Raises:
            InvalidParametersError: If config is invalid
        """
        # Validate OpenRouter API key
        if not config.openrouter.api_key:
            raise InvalidParametersError("OpenRouter API key is required")
        
        # Validate video processing config
        if config.video_processing.max_parallel_jobs < 1:
            raise InvalidParametersError("max_parallel_jobs must be at least 1")
        
        # Validate cache TTL
        if config.cache_ttl < 0:
            raise InvalidParametersError("cache_ttl must be non-negative")
        
        return True




