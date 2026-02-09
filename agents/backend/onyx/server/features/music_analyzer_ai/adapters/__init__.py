"""
Adapter Pattern - Adapt interfaces
"""

from .adapter import IAdapter, BaseAdapter
from .spotify_adapter import SpotifyAdapter
from .audio_adapter import AudioAdapter

__all__ = [
    "IAdapter",
    "BaseAdapter",
    "SpotifyAdapter",
    "AudioAdapter"
]








