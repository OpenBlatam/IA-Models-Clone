"""
Embeddings Submodule
Aggregates various embedding components.
"""

from .base import FeatureEmbedding
from .audio import AudioFeatureEmbedding
from .music import MusicFeatureEmbedding

__all__ = [
    "FeatureEmbedding",
    "AudioFeatureEmbedding",
    "MusicFeatureEmbedding",
]



