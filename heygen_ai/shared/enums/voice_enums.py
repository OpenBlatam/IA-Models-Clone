"""
Voice Enums
===========

Enumerations for voice generation.
"""

from enum import Enum


class VoiceQuality(str, Enum):
    """Voice quality levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    ULTRA = "ultra"


class AudioFormat(str, Enum):
    """Audio format options."""
    WAV = "wav"
    MP3 = "mp3"
    OGG = "ogg"
    FLAC = "flac"



