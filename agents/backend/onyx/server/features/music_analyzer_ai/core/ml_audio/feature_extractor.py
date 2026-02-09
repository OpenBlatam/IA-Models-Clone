"""
Audio Feature Extractor Module

Extract advanced audio features using librosa.
"""

import logging
import time
import numpy as np

logger = logging.getLogger(__name__)

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    logger.warning("Librosa not available, audio analysis will be limited")

from .dataclasses import AudioFeatures


class AudioFeatureExtractor:
    """
    Extract advanced audio features using librosa.
    
    Args:
        sr: Sample rate for audio processing.
    """
    
    def __init__(self, sr: int = 22050):
        self.sr = sr
        if not LIBROSA_AVAILABLE:
            raise ImportError("Librosa is required for audio feature extraction")
    
    def extract_features(self, audio_path: str) -> AudioFeatures:
        """
        Extract comprehensive audio features from file.
        
        Args:
            audio_path: Path to audio file.
        
        Returns:
            AudioFeatures object.
        """
        start_time = time.time()
        
        # Load audio
        y, sr = librosa.load(audio_path, sr=self.sr)
        duration = librosa.get_duration(y=y, sr=sr)
        
        # Extract features
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
        
        # Tempo and beats
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        
        processing_time = time.time() - start_time
        logger.info(f"Extracted audio features in {processing_time:.2f}s")
        
        return AudioFeatures(
            mfcc=mfcc,
            chroma=chroma,
            spectral_contrast=spectral_contrast,
            tonnetz=tonnetz,
            tempo=tempo,
            beats=beats,
            duration=duration
        )
    
    def extract_from_array(self, y: np.ndarray, sr: int) -> AudioFeatures:
        """
        Extract features from audio array.
        
        Args:
            y: Audio signal array.
            sr: Sample rate.
        
        Returns:
            AudioFeatures object.
        """
        duration = librosa.get_duration(y=y, sr=sr)
        
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        
        return AudioFeatures(
            mfcc=mfcc,
            chroma=chroma,
            spectral_contrast=spectral_contrast,
            tonnetz=tonnetz,
            tempo=tempo,
            beats=beats,
            duration=duration
        )



