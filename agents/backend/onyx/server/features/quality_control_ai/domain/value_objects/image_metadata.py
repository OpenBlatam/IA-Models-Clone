"""
Image Metadata Value Object

Immutable value object representing image metadata.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
from datetime import datetime


@dataclass(frozen=True)
class ImageMetadata:
    """
    Immutable value object representing image metadata.
    
    Attributes:
        width: Image width in pixels
        height: Image height in pixels
        channels: Number of color channels (1 for grayscale, 3 for RGB)
        format: Image format (e.g., 'jpg', 'png', 'bmp')
        size_bytes: Image size in bytes
        captured_at: Timestamp when image was captured
        source: Source of the image (e.g., 'camera', 'file', 'upload')
    """
    
    width: int
    height: int
    channels: int = 3
    format: Optional[str] = None
    size_bytes: Optional[int] = None
    captured_at: Optional[datetime] = None
    source: Optional[str] = None
    
    def __post_init__(self):
        """Validate image metadata."""
        if self.width <= 0:
            raise ValueError(f"Image width must be positive, got {self.width}")
        if self.height <= 0:
            raise ValueError(f"Image height must be positive, got {self.height}")
        if self.channels not in [1, 3, 4]:
            raise ValueError(f"Image channels must be 1, 3, or 4, got {self.channels}")
    
    @property
    def resolution(self) -> Tuple[int, int]:
        """Get image resolution as (width, height) tuple."""
        return (self.width, self.height)
    
    @property
    def total_pixels(self) -> int:
        """Calculate total number of pixels."""
        return self.width * self.height
    
    @property
    def aspect_ratio(self) -> float:
        """Calculate aspect ratio (width / height)."""
        return self.width / self.height if self.height > 0 else 0.0
    
    def to_dict(self) -> dict:
        """
        Convert image metadata to dictionary.
        
        Returns:
            Dictionary representation of the image metadata
        """
        return {
            "width": self.width,
            "height": self.height,
            "channels": self.channels,
            "format": self.format,
            "size_bytes": self.size_bytes,
            "resolution": self.resolution,
            "total_pixels": self.total_pixels,
            "aspect_ratio": self.aspect_ratio,
            "captured_at": self.captured_at.isoformat() if self.captured_at else None,
            "source": self.source,
        }



