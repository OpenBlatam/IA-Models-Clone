"""
Video Types
===========

Data types for video rendering.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

from shared.enums import VideoQuality, VideoFormat, VideoCodec


@dataclass
class VideoConfig:
    """Configuration for video rendering.
    
    Attributes:
        resolution: Output resolution (720p, 1080p, 4k)
        fps: Frames per second
        quality: Video quality level
        format: Output video format
        codec: Video codec
        bitrate: Bitrate in kbps (None for auto)
        enable_effects: Enable video effects
        enable_optimization: Enable quality optimization
    """
    resolution: str = "1080p"
    fps: int = 30
    quality: VideoQuality = VideoQuality.HIGH
    format: VideoFormat = VideoFormat.MP4
    codec: VideoCodec = VideoCodec.H264
    bitrate: int | None = None
    enable_effects: bool = True
    enable_optimization: bool = True

    def get_resolution_tuple(self) -> Tuple[int, int]:
        """Get resolution as (width, height) tuple."""
        resolution_map = {
            "720p": (1280, 720),
            "1080p": (1920, 1080),
            "4k": (3840, 2160),
        }
        return resolution_map.get(self.resolution, (1920, 1080))

    def get_bitrate(self) -> int:
        """Get bitrate based on quality."""
        if self.bitrate:
            return self.bitrate
        
        quality_bitrates = {
            VideoQuality.LOW: 1000,
            VideoQuality.MEDIUM: 2500,
            VideoQuality.HIGH: 5000,
            VideoQuality.ULTRA: 10000,
        }
        return quality_bitrates.get(self.quality, 5000)


@dataclass
class VideoEffect:
    """Video effect configuration.
    
    Attributes:
        name: Effect name
        parameters: Effect parameters dictionary
        start_time: Effect start time in seconds
        duration: Effect duration in seconds
        enabled: Whether effect is enabled
    """
    name: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    start_time: float = 0.0
    duration: float = 0.0
    enabled: bool = True



