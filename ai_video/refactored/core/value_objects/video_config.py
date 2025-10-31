from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

# Constants
BUFFER_SIZE = 1024

from enum import Enum
from typing import Dict, Optional
from uuid import UUID
from pydantic import BaseModel, Field, field_validator
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Video Configuration Value Object
===============================

Immutable value object representing video configuration settings.
"""




class VideoQuality(str, Enum):
    """Video quality options."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    ULTRA = "ultra"


class VideoFormat(str, Enum):
    """Video format options."""
    MP4 = "mp4"
    MOV = "mov"
    AVI = "avi"
    WEBM = "webm"


class VideoConfig(BaseModel):
    """
    Video configuration value object.
    
    Immutable configuration for video generation including quality,
    format, aspect ratio, and other settings.
    """
    
    # Basic settings
    quality: VideoQuality = Field(default=VideoQuality.HIGH, description="Video quality")
    format: VideoFormat = Field(default=VideoFormat.MP4, description="Video format")
    aspect_ratio: str = Field(default="16:9", description="Video aspect ratio")
    
    # Resolution settings
    width: Optional[int] = Field(None, ge=320, le=7680, description="Video width in pixels")
    height: Optional[int] = Field(None, ge=240, le=4320, description="Video height in pixels")
    fps: int = Field(default=30, ge=1, le=120, description="Frames per second")
    
    # Audio settings
    audio_enabled: bool = Field(default=True, description="Enable audio")
    audio_bitrate: int = Field(default=128, ge=32, le=320, description="Audio bitrate in kbps")
    audio_channels: int = Field(default=2, ge=1, le=8, description="Audio channels")
    audio_sample_rate: int = Field(default=44100, ge=8000, le=192000, description="Audio sample rate")
    
    # Video encoding settings
    video_bitrate: Optional[int] = Field(None, ge=100, le=50000, description="Video bitrate in kbps")
    codec: str = Field(default="h264", description="Video codec")
    preset: str = Field(default="medium", description="Encoding preset")
    
    # Advanced settings
    enable_hardware_acceleration: bool = Field(default=False, description="Enable hardware acceleration")
    enable_two_pass: bool = Field(default=False, description="Enable two-pass encoding")
    enable_optimization: bool = Field(default=True, description="Enable optimization")
    
    # Metadata
    title: Optional[str] = Field(None, max_length=100, description="Video title")
    description: Optional[str] = Field(None, max_length=500, description="Video description")
    tags: list[str] = Field(default=[], description="Video tags")
    
    # Custom settings
    custom_settings: Dict = Field(
        default={},
        description="Custom encoding settings"
    )
    
    @field_validator("aspect_ratio")
    @classmethod
    def validate_aspect_ratio(cls, v: str) -> str:
        """Validate aspect ratio."""
        valid_ratios = ["16:9", "9:16", "1:1", "4:3", "3:4", "21:9"]
        if v not in valid_ratios:
            raise ValueError(f"Invalid aspect ratio: {v}")
        return v
    
    @field_validator("width", "height")
    @classmethod
    def validate_resolution(cls, v: Optional[int]) -> Optional[int]:
        """Validate resolution dimensions."""
        if v is not None and v % 2 != 0:
            raise ValueError("Resolution dimensions must be even numbers")
        return v
    
    @field_validator("fps")
    @classmethod
    def validate_fps(cls, v: int) -> int:
        """Validate frames per second."""
        common_fps = [24, 25, 30, 50, 60, 120]
        if v not in common_fps:
            raise ValueError(f"FPS must be one of: {common_fps}")
        return v
    
    def get_resolution(self) -> tuple[int, int]:
        """Get video resolution based on aspect ratio and quality."""
        if self.width and self.height:
            return self.width, self.height
        
        # Default resolutions based on quality and aspect ratio
        resolutions = {
            VideoQuality.LOW: {
                "16:9": (1280, 720),
                "9:16": (720, 1280),
                "1:1": (1080, 1080),
                "4:3": (1024, 768),
                "3:4": (768, 1024),
                "21:9": (1920, 823),
            },
            VideoQuality.MEDIUM: {
                "16:9": (1920, 1080),
                "9:16": (1080, 1920),
                "1:1": (1440, 1440),
                "4:3": (1440, 1080),
                "3:4": (1080, 1440),
                "21:9": (2560, 1097),
            },
            VideoQuality.HIGH: {
                "16:9": (2560, 1440),
                "9:16": (1440, 2560),
                "1:1": (1920, 1920),
                "4:3": (1920, 1440),
                "3:4": (1440, 1920),
                "21:9": (3440, 1474),
            },
            VideoQuality.ULTRA: {
                "16:9": (3840, 2160),
                "9:16": (2160, 3840),
                "1:1": (2880, 2880),
                "4:3": (2880, 2160),
                "3:4": (2160, 2880),
                "21:9": (5120, 2194),
            },
        }
        
        return resolutions[self.quality][self.aspect_ratio]
    
    def get_video_bitrate(self) -> int:
        """Get video bitrate based on quality and resolution."""
        if self.video_bitrate:
            return self.video_bitrate
        
        width, height = self.get_resolution()
        pixels = width * height
        
        # Bitrate calculation based on quality and resolution
        bitrate_multipliers = {
            VideoQuality.LOW: 0.5,
            VideoQuality.MEDIUM: 1.0,
            VideoQuality.HIGH: 2.0,
            VideoQuality.ULTRA: 4.0,
        }
        
        base_bitrate = pixels * 0.1  # Base bitrate per pixel
        return int(base_bitrate * bitrate_multipliers[self.quality])
    
    def get_encoding_settings(self) -> Dict:
        """Get encoding settings for video processing."""
        width, height = self.get_resolution()
        
        return {
            "width": width,
            "height": height,
            "fps": self.fps,
            "video_bitrate": self.get_video_bitrate(),
            "audio_bitrate": self.audio_bitrate,
            "audio_channels": self.audio_channels,
            "audio_sample_rate": self.audio_sample_rate,
            "codec": self.codec,
            "preset": self.preset,
            "enable_hardware_acceleration": self.enable_hardware_acceleration,
            "enable_two_pass": self.enable_two_pass,
            "enable_optimization": self.enable_optimization,
            "custom_settings": self.custom_settings,
        }
    
    def get_file_extension(self) -> str:
        """Get file extension based on format."""
        return f".{self.format.value}"
    
    def get_mime_type(self) -> str:
        """Get MIME type based on format."""
        mime_types = {
            VideoFormat.MP4: "video/mp4",
            VideoFormat.MOV: "video/quicktime",
            VideoFormat.AVI: "video/x-msvideo",
            VideoFormat.WEBM: "video/webm",
        }
        return mime_types[self.format]
    
    def estimate_file_size(self, duration_seconds: float) -> int:
        """Estimate file size in bytes."""
        video_bitrate = self.get_video_bitrate()
        audio_bitrate = self.audio_bitrate if self.audio_enabled else 0
        total_bitrate = video_bitrate + audio_bitrate
        
        # Convert to bytes per second
        bytes_per_second = (total_bitrate * 1000) / 8
        
        return int(bytes_per_second * duration_seconds)
    
    def is_mobile_optimized(self) -> bool:
        """Check if configuration is optimized for mobile."""
        width, height = self.get_resolution()
        return width <= 1920 and height <= 1920 and self.fps <= 60
    
    def is_web_optimized(self) -> bool:
        """Check if configuration is optimized for web."""
        return self.format in [VideoFormat.MP4, VideoFormat.WEBM] and self.fps <= 60
    
    def clone_with_updates(self, **updates) -> "VideoConfig":
        """Create a new instance with updated values."""
        return self.model_copy(update=updates)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return self.model_dump()
    
    def __eq__(self, other: object) -> bool:
        """Compare configurations."""
        if not isinstance(other, VideoConfig):
            return False
        return self.model_dump() == other.model_dump()
    
    def __hash__(self) -> int:
        """Hash configuration."""
        return hash(tuple(sorted(self.model_dump().items()))) 