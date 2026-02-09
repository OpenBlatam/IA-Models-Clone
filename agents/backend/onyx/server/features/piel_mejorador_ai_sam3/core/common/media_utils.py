"""
Media Utilities for Piel Mejorador AI SAM3
=========================================

Unified media file encoding and processing utilities.
"""

import logging
from typing import Dict, Any, Optional, Union
from pathlib import Path
import mimetypes

from .encoding_utils import EncodingUtils

logger = logging.getLogger(__name__)


class MediaUtils:
    """Unified media utilities."""
    
    # MIME type mappings
    MIME_TYPES = {
        # Images
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
        ".svg": "image/svg+xml",
        ".ico": "image/x-icon",
        # Videos
        ".mp4": "video/mp4",
        ".mov": "video/quicktime",
        ".avi": "video/x-msvideo",
        ".webm": "video/webm",
        ".mkv": "video/x-matroska",
        ".flv": "video/x-flv",
        ".wmv": "video/x-ms-wmv",
        # Audio
        ".mp3": "audio/mpeg",
        ".wav": "audio/wav",
        ".ogg": "audio/ogg",
        ".flac": "audio/flac",
        ".aac": "audio/aac",
    }
    
    @staticmethod
    def get_mime_type(file_path: Union[str, Path]) -> str:
        """
        Get MIME type from file path.
        
        Args:
            file_path: Path to file
            
        Returns:
            MIME type string
        """
        path = Path(file_path)
        
        # Try mimetypes first
        mime_type, _ = mimetypes.guess_type(str(path))
        if mime_type:
            return mime_type
        
        # Fallback to mapping
        extension = path.suffix.lower()
        return MediaUtils.MIME_TYPES.get(extension, "application/octet-stream")
    
    @staticmethod
    def is_image(file_path: Union[str, Path]) -> bool:
        """
        Check if file is an image.
        
        Args:
            file_path: Path to file
            
        Returns:
            True if image
        """
        mime_type = MediaUtils.get_mime_type(file_path)
        return mime_type.startswith("image/")
    
    @staticmethod
    def is_video(file_path: Union[str, Path]) -> bool:
        """
        Check if file is a video.
        
        Args:
            file_path: Path to file
            
        Returns:
            True if video
        """
        mime_type = MediaUtils.get_mime_type(file_path)
        return mime_type.startswith("video/")
    
    @staticmethod
    def is_audio(file_path: Union[str, Path]) -> bool:
        """
        Check if file is audio.
        
        Args:
            file_path: Path to file
            
        Returns:
            True if audio
        """
        mime_type = MediaUtils.get_mime_type(file_path)
        return mime_type.startswith("audio/")
    
    @staticmethod
    def encode_image_to_base64(
        image_path: Union[str, Path],
        mime_type: Optional[str] = None
    ) -> str:
        """
        Encode image file to base64 string.
        
        Args:
            image_path: Path to image file
            mime_type: Optional MIME type (auto-detected if not provided)
            
        Returns:
            Base64 encoded string
        """
        return EncodingUtils.encode_base64_file(image_path)
    
    @staticmethod
    def create_data_url(
        file_path: Union[str, Path],
        mime_type: Optional[str] = None
    ) -> str:
        """
        Create data URL from file.
        
        Args:
            file_path: Path to file
            mime_type: Optional MIME type (auto-detected if not provided)
            
        Returns:
            Data URL string (data:mime/type;base64,encoded_data)
        """
        if mime_type is None:
            mime_type = MediaUtils.get_mime_type(file_path)
        
        encoded = EncodingUtils.encode_base64_file(file_path)
        return f"data:{mime_type};base64,{encoded}"
    
    @staticmethod
    def create_image_data_url(
        image_path: Union[str, Path],
        mime_type: Optional[str] = None
    ) -> str:
        """
        Create data URL for image.
        
        Args:
            image_path: Path to image file
            mime_type: Optional MIME type (auto-detected if not provided)
            
        Returns:
            Image data URL
        """
        if mime_type is None:
            mime_type = MediaUtils.get_mime_type(image_path)
        
        return MediaUtils.create_data_url(image_path, mime_type)
    
    @staticmethod
    def create_multimodal_content(
        text: str,
        image_path: Union[str, Path],
        mime_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create multimodal content for vision APIs.
        
        Args:
            text: Text content
            image_path: Path to image file
            mime_type: Optional MIME type (auto-detected if not provided)
            
        Returns:
            Multimodal content dictionary
        """
        if mime_type is None:
            mime_type = MediaUtils.get_mime_type(image_path)
        
        image_url = MediaUtils.create_image_data_url(image_path, mime_type)
        
        return {
            "role": "user",
            "content": [
                {"type": "text", "text": text},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_url
                    }
                }
            ]
        }
    
    @staticmethod
    def create_vision_message(
        text: str,
        image_path: Union[str, Path],
        mime_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create vision API message format.
        
        Args:
            text: Text prompt
            image_path: Path to image file
            mime_type: Optional MIME type (auto-detected if not provided)
            
        Returns:
            Message dictionary for vision APIs
        """
        return MediaUtils.create_multimodal_content(text, image_path, mime_type)
    
    @staticmethod
    def get_file_extension_from_mime(mime_type: str) -> Optional[str]:
        """
        Get file extension from MIME type.
        
        Args:
            mime_type: MIME type string
            
        Returns:
            File extension (with dot) or None
        """
        # Reverse lookup in MIME_TYPES
        for ext, mime in MediaUtils.MIME_TYPES.items():
            if mime == mime_type:
                return ext
        
        # Try mimetypes module
        ext = mimetypes.guess_extension(mime_type)
        return ext
    
    @staticmethod
    def validate_media_file(
        file_path: Union[str, Path],
        allowed_types: Optional[list[str]] = None
    ) -> bool:
        """
        Validate media file type.
        
        Args:
            file_path: Path to file
            allowed_types: Optional list of allowed MIME type prefixes (e.g., ["image/", "video/"])
            
        Returns:
            True if valid
        """
        if allowed_types is None:
            return True
        
        mime_type = MediaUtils.get_mime_type(file_path)
        return any(mime_type.startswith(prefix) for prefix in allowed_types)


# Convenience functions
def get_mime_type(file_path: Union[str, Path]) -> str:
    """Get MIME type."""
    return MediaUtils.get_mime_type(file_path)


def is_image(file_path: Union[str, Path]) -> bool:
    """Check if image."""
    return MediaUtils.is_image(file_path)


def is_video(file_path: Union[str, Path]) -> bool:
    """Check if video."""
    return MediaUtils.is_video(file_path)


def create_data_url(file_path: Union[str, Path], **kwargs) -> str:
    """Create data URL."""
    return MediaUtils.create_data_url(file_path, **kwargs)


def create_image_data_url(image_path: Union[str, Path], **kwargs) -> str:
    """Create image data URL."""
    return MediaUtils.create_image_data_url(image_path, **kwargs)


def create_multimodal_content(text: str, image_path: Union[str, Path], **kwargs) -> Dict[str, Any]:
    """Create multimodal content."""
    return MediaUtils.create_multimodal_content(text, image_path, **kwargs)




