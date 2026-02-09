"""
Video Enums
===========

Enumerations for video rendering.
"""

from enum import Enum


class VideoQuality(str, Enum):
    """Video quality levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    ULTRA = "ultra"


class VideoCodec(str, Enum):
    """Video codec options."""
    H264 = "h264"
    H265 = "h265"
    VP9 = "vp9"
    PRORES = "prores"


class VideoFormat(str, Enum):
    """Video format options."""
    MP4 = "mp4"
    MOV = "mov"
    AVI = "avi"
    WEBM = "webm"



