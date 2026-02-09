"""
Avatar Enums
============

Enumerations for avatar generation.
"""

from enum import Enum


class AvatarStyle(str, Enum):
    """Avatar style enumeration."""
    REALISTIC = "realistic"
    CARTOON = "cartoon"
    ANIME = "anime"
    ARTISTIC = "artistic"


class AvatarQuality(str, Enum):
    """Avatar quality levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    ULTRA = "ultra"


class Resolution(str, Enum):
    """Video resolution options."""
    P720 = "720p"
    P1080 = "1080p"
    P4K = "4k"



