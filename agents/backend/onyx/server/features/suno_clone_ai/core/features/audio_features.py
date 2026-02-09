"""
Audio Feature Extraction

Utilities for extracting audio features.
"""

import logging
import numpy as np
from typing import Optional, Dict

logger = logging.getLogger(__name__)

# Try to import librosa
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    logger.warning("Librosa not available for feature extraction")


class AudioFeatureExtractor:
    """Extract audio features."""
    
    def __init__(self, sample_rate: int = 32000):
        """
        Initialize feature extractor.
        
        Args:
            sample_rate: Sample rate
        """
        if not LIBROSA_AVAILABLE:
            raise ImportError("Librosa required for feature extraction")
        
        self.sample_rate = sample_rate
    
    def extract_mfcc(
        self,
        audio: np.ndarray,
        n_mfcc: int = 13,
        n_fft: int = 2048,
        hop_length: int = 512
    ) -> np.ndarray:
        """
        Extract MFCC features.
        
        Args:
            audio: Audio array
            n_mfcc: Number of MFCC coefficients
            n_fft: FFT window size
            hop_length: Hop length
            
        Returns:
            MFCC features
        """
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_mfcc=n_mfcc,
            n_fft=n_fft,
            hop_length=hop_length
        )
        return mfcc
    
    def extract_mel_spectrogram(
        self,
        audio: np.ndarray,
        n_mels: int = 128,
        n_fft: int = 2048,
        hop_length: int = 512
    ) -> np.ndarray:
        """
        Extract mel spectrogram.
        
        Args:
            audio: Audio array
            n_mels: Number of mel bands
            n_fft: FFT window size
            hop_length: Hop length
            
        Returns:
            Mel spectrogram
        """
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length
        )
        return librosa.power_to_db(mel_spec)
    
    def extract_chroma(
        self,
        audio: np.ndarray,
        n_fft: int = 2048,
        hop_length: int = 512
    ) -> np.ndarray:
        """
        Extract chroma features.
        
        Args:
            audio: Audio array
            n_fft: FFT window size
            hop_length: Hop length
            
        Returns:
            Chroma features
        """
        chroma = librosa.feature.chroma(
            y=audio,
            sr=self.sample_rate,
            n_fft=n_fft,
            hop_length=hop_length
        )
        return chroma
    
    def extract_all(
        self,
        audio: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Extract all features.
        
        Args:
            audio: Audio array
            
        Returns:
            Dictionary of features
        """
        return {
            'mfcc': self.extract_mfcc(audio),
            'mel_spectrogram': self.extract_mel_spectrogram(audio),
            'chroma': self.extract_chroma(audio)
        }


def extract_mfcc(
    audio: np.ndarray,
    sample_rate: int = 32000,
    **kwargs
) -> np.ndarray:
    """Convenience function to extract MFCC."""
    extractor = AudioFeatureExtractor(sample_rate)
    return extractor.extract_mfcc(audio, **kwargs)


def extract_mel_spectrogram(
    audio: np.ndarray,
    sample_rate: int = 32000,
    **kwargs
) -> np.ndarray:
    """Convenience function to extract mel spectrogram."""
    extractor = AudioFeatureExtractor(sample_rate)
    return extractor.extract_mel_spectrogram(audio, **kwargs)


def extract_chroma(
    audio: np.ndarray,
    sample_rate: int = 32000,
    **kwargs
) -> np.ndarray:
    """Convenience function to extract chroma."""
    extractor = AudioFeatureExtractor(sample_rate)
    return extractor.extract_chroma(audio, **kwargs)


def extract_spectral_features(
    audio: np.ndarray,
    sample_rate: int = 32000
) -> Dict[str, np.ndarray]:
    """Convenience function to extract all features."""
    extractor = AudioFeatureExtractor(sample_rate)
    return extractor.extract_all(audio)



