"""
Shared Module
=============

Shared types, enums, configurations, and interfaces used across the application.
"""

from .enums import (
    AvatarStyle,
    AvatarQuality,
    VoiceQuality,
    VideoQuality,
    ScriptStyle,
    Resolution,
    AudioFormat,
    VideoFormat,
    VideoCodec,
)
from .types import (
    AvatarGenerationConfig,
    VoiceGenerationConfig,
    VideoConfig,
    ScriptGenerationConfig,
    AvatarModel,
    VoiceModel,
    VideoEffect,
)

__all__ = [
    # Enums
    "AvatarStyle",
    "AvatarQuality",
    "VoiceQuality",
    "VideoQuality",
    "ScriptStyle",
    "Resolution",
    "AudioFormat",
    "VideoFormat",
    "VideoCodec",
    # Types
    "AvatarGenerationConfig",
    "VoiceGenerationConfig",
    "VideoConfig",
    "ScriptGenerationConfig",
    "AvatarModel",
    "VoiceModel",
    "VideoEffect",
]



