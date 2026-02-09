"""
Feature Extraction Module

Provides:
- Audio feature extraction
- Text feature extraction
- Feature engineering
- Feature selection
"""

from .audio_features import (
    AudioFeatureExtractor,
    extract_mfcc,
    extract_mel_spectrogram,
    extract_chroma,
    extract_spectral_features
)

from .text_features import (
    TextFeatureExtractor,
    extract_embeddings,
    extract_tfidf,
    extract_bow
)

__all__ = [
    # Audio features
    "AudioFeatureExtractor",
    "extract_mfcc",
    "extract_mel_spectrogram",
    "extract_chroma",
    "extract_spectral_features",
    # Text features
    "TextFeatureExtractor",
    "extract_embeddings",
    "extract_tfidf",
    "extract_bow"
]



