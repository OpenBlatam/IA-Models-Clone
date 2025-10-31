from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

from enum import Enum
from typing import Any, Dict
from dataclasses import dataclass
from ..exceptions.domain_errors import ValueObjectValidationError
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Video Quality Value Object

Represents video quality settings with validation.
"""




class VideoQualityLevel(Enum):
    """Video quality levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    ULTRA = "ultra"


@dataclass(frozen=True)
class VideoQuality:
    """
    Video quality value object.
    
    Business Rules:
    - Quality level must be valid
    - Resolution must match quality level constraints
    - Bitrate must be within acceptable ranges
    """
    
    level: VideoQualityLevel
    width: int
    height: int
    bitrate: int  # kbps
    fps: int = 30
    
    def __post_init__(self) -> None:
        """Validate video quality settings."""
        self._validate()
    
    def _validate(self) -> None:
        """Validate video quality parameters."""
        # Validate level
        if not isinstance(self.level, VideoQualityLevel):
            raise ValueObjectValidationError("Quality level must be a VideoQualityLevel enum")
        
        # Validate dimensions
        if self.width <= 0 or self.height <= 0:
            raise ValueObjectValidationError("Video dimensions must be positive")
        
        if self.width > 4096 or self.height > 4096:
            raise ValueObjectValidationError("Video dimensions cannot exceed 4096 pixels")
        
        # Validate aspect ratio (must be reasonable)
        aspect_ratio = self.width / self.height
        if aspect_ratio < 0.1 or aspect_ratio > 10.0:
            raise ValueObjectValidationError("Video aspect ratio must be between 0.1 and 10.0")
        
        # Validate bitrate
        if self.bitrate <= 0:
            raise ValueObjectValidationError("Bitrate must be positive")
        
        if self.bitrate > 50000:  # 50 Mbps max
            raise ValueObjectValidationError("Bitrate cannot exceed 50,000 kbps")
        
        # Validate FPS
        if self.fps <= 0 or self.fps > 120:
            raise ValueObjectValidationError("FPS must be between 1 and 120")
        
        # Validate quality-specific constraints
        self._validate_quality_constraints()
    
    def _validate_quality_constraints(self) -> None:
        """Validate constraints based on quality level."""
        quality_constraints = {
            VideoQualityLevel.LOW: {
                "max_width": 640,
                "max_height": 480,
                "max_bitrate": 1000,
                "max_fps": 30
            },
            VideoQualityLevel.MEDIUM: {
                "max_width": 1280,
                "max_height": 720,
                "max_bitrate": 3000,
                "max_fps": 60
            },
            VideoQualityLevel.HIGH: {
                "max_width": 1920,
                "max_height": 1080,
                "max_bitrate": 8000,
                "max_fps": 60
            },
            VideoQualityLevel.ULTRA: {
                "max_width": 4096,
                "max_height": 4096,
                "max_bitrate": 50000,
                "max_fps": 120
            }
        }
        
        constraints = quality_constraints[self.level]
        
        if self.width > constraints["max_width"]:
            raise ValueObjectValidationError(
                f"Width {self.width} exceeds maximum {constraints['max_width']} for {self.level.value} quality"
            )
        
        if self.height > constraints["max_height"]:
            raise ValueObjectValidationError(
                f"Height {self.height} exceeds maximum {constraints['max_height']} for {self.level.value} quality"
            )
        
        if self.bitrate > constraints["max_bitrate"]:
            raise ValueObjectValidationError(
                f"Bitrate {self.bitrate} kbps exceeds maximum {constraints['max_bitrate']} kbps for {self.level.value} quality"
            )
        
        if self.fps > constraints["max_fps"]:
            raise ValueObjectValidationError(
                f"FPS {self.fps} exceeds maximum {constraints['max_fps']} for {self.level.value} quality"
            )
    
    @property
    def resolution(self) -> str:
        """Get resolution string."""
        return f"{self.width}x{self.height}"
    
    @property
    def aspect_ratio(self) -> float:
        """Get aspect ratio."""
        return self.width / self.height
    
    @property
    def pixels(self) -> int:
        """Get total pixel count."""
        return self.width * self.height
    
    def is_hd(self) -> bool:
        """Check if quality is HD or higher."""
        return self.level in [VideoQualityLevel.HIGH, VideoQualityLevel.ULTRA]
    
    def is_4k(self) -> bool:
        """Check if quality is 4K."""
        return self.width >= 3840 and self.height >= 2160
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "level": self.level.value,
            "width": self.width,
            "height": self.height,
            "resolution": self.resolution,
            "bitrate": self.bitrate,
            "fps": self.fps,
            "aspect_ratio": round(self.aspect_ratio, 2),
            "pixels": self.pixels,
            "is_hd": self.is_hd(),
            "is_4k": self.is_4k()
        }
    
    @classmethod
    def create_preset(cls, level: VideoQualityLevel) -> 'VideoQuality':
        """Create video quality from preset level."""
        presets = {
            VideoQualityLevel.LOW: {
                "width": 640,
                "height": 480,
                "bitrate": 800,
                "fps": 24
            },
            VideoQualityLevel.MEDIUM: {
                "width": 1280,
                "height": 720,
                "bitrate": 2500,
                "fps": 30
            },
            VideoQualityLevel.HIGH: {
                "width": 1920,
                "height": 1080,
                "bitrate": 5000,
                "fps": 30
            },
            VideoQualityLevel.ULTRA: {
                "width": 3840,
                "height": 2160,
                "bitrate": 15000,
                "fps": 30
            }
        }
        
        preset = presets[level]
        return cls(
            level=level,
            width=preset["width"],
            height=preset["height"],
            bitrate=preset["bitrate"],
            fps=preset["fps"]
        )
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VideoQuality':
        """Create video quality from dictionary."""
        return cls(
            level=VideoQualityLevel(data["level"]),
            width=data["width"],
            height=data["height"],
            bitrate=data["bitrate"],
            fps=data.get("fps", 30)
        ) 