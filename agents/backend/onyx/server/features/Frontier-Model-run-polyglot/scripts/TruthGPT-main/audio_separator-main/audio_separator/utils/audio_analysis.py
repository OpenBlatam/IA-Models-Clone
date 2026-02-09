"""
Audio analysis utilities.

Refactored to consolidate functions into AudioAnalyzer class.
"""

from typing import Dict, Tuple, Optional
import numpy as np

from ..core.base_component import BaseComponent
from ..logger import logger
from ..separator.base_separator import DEFAULT_SAMPLE_RATE
from .audio_helpers import (
    calculate_rms,
    calculate_peak,
    amplitude_to_db
)


class AudioAnalyzer(BaseComponent):
    """
    Audio analysis utilities.
    
    Responsibilities:
    - Analyze audio statistics
    - Detect silence regions
    - Calculate loudness metrics
    - Detect beats
    - Extract audio features
    
    Single Responsibility: Handle all audio analysis operations.
    """
    
    def __init__(self, sample_rate: int = DEFAULT_SAMPLE_RATE, name: Optional[str] = None):
        """
        Initialize audio analyzer.
        
        Args:
            sample_rate: Sample rate for audio operations
            name: Component name (defaults to class name)
        """
        super().__init__(name=name or "AudioAnalyzer")
        self.sample_rate = sample_rate
        self.initialize()
    
    def _do_initialize(self, **kwargs):
        """No initialization needed."""
        pass
    
    def analyze(self, audio: np.ndarray) -> Dict[str, float]:
        """
        Analyze audio and return statistics.
        
        Args:
            audio: Audio array
            
        Returns:
            Dictionary with audio statistics
        """
        duration = len(audio) / self.sample_rate
        
        # Basic statistics using helpers
        peak = calculate_peak(audio)
        rms = calculate_rms(audio)
        
        stats = {
            "duration": duration,
            "sample_rate": self.sample_rate,
            "samples": len(audio),
            "channels": 1 if audio.ndim == 1 else audio.shape[0],
            "max_amplitude": peak,
            "min_amplitude": float(np.abs(audio).min()),
            "mean_amplitude": float(np.abs(audio).mean()),
            "rms": rms,
            "peak_db": amplitude_to_db(peak),
        }
        
        # Zero crossing rate
        if audio.ndim == 1:
            zero_crossings = np.sum(np.diff(np.signbit(audio)))
            stats["zero_crossing_rate"] = zero_crossings / len(audio)
        else:
            zcr = [np.sum(np.diff(np.signbit(ch))) / len(ch) for ch in audio]
            stats["zero_crossing_rate"] = float(np.mean(zcr))
        
        # Spectral features
        try:
            import librosa
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)[0]
            stats["spectral_centroid_mean"] = float(np.mean(spectral_centroid))
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sample_rate)[0]
            stats["spectral_rolloff_mean"] = float(np.mean(spectral_rolloff))
            zcr = librosa.feature.zero_crossing_rate(audio)[0]
            stats["zcr_mean"] = float(np.mean(zcr))
            tempo, _ = librosa.beat.beat_track(y=audio, sr=self.sample_rate)
            stats["tempo"] = float(tempo)
        except ImportError:
            logger.warning("librosa not available, skipping advanced analysis")
        
        return stats
    
    def detect_silence(
        self,
        audio: np.ndarray,
        threshold: float = 0.01,
        min_duration: float = 0.1
    ) -> list:
        """
        Detect silence regions in audio.
        
        Args:
            audio: Audio array
            threshold: Silence threshold
            min_duration: Minimum duration to consider as silence (seconds)
            
        Returns:
            List of (start, end) tuples in seconds
        """
        # Find samples below threshold
        if audio.ndim > 1:
            audio = np.mean(audio, axis=0)
        
        below_threshold = np.abs(audio) < threshold
        
        # Find silence regions
        silence_regions = []
        in_silence = False
        silence_start = 0
        
        for i, is_silent in enumerate(below_threshold):
            if is_silent and not in_silence:
                silence_start = i
                in_silence = True
            elif not is_silent and in_silence:
                silence_end = i
                duration = (silence_end - silence_start) / self.sample_rate
                if duration >= min_duration:
                    silence_regions.append((
                        silence_start / self.sample_rate,
                        silence_end / self.sample_rate
                    ))
                in_silence = False
        
        # Handle case where audio ends in silence
        if in_silence:
            silence_end = len(audio)
            duration = (silence_end - silence_start) / self.sample_rate
            if duration >= min_duration:
                silence_regions.append((
                    silence_start / self.sample_rate,
                    silence_end / self.sample_rate
                ))
        
        return silence_regions
    
    def calculate_loudness(self, audio: np.ndarray) -> Dict[str, float]:
        """
        Calculate loudness metrics (LUFS approximation).
        
        Args:
            audio: Audio array
            
        Returns:
            Dictionary with loudness metrics
        """
        # Simple loudness approximation
        # For accurate LUFS, use specialized libraries like pyloudnorm
        
        # Calculate RMS and peak using helpers
        rms = calculate_rms(audio)
        peak = calculate_peak(audio)
        
        # Convert to dB using helper
        rms_db = amplitude_to_db(rms)
        peak_db = amplitude_to_db(peak)
        
        # Dynamic range
        dynamic_range = peak_db - rms_db
        
        return {
            "rms_db": float(rms_db),
            "peak_db": float(peak_db),
            "dynamic_range_db": float(dynamic_range),
            "lufs_approx": float(rms_db - 23.0)  # Rough approximation
        }
    
    def detect_beats(self, audio: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Detect beats in audio.
        
        Args:
            audio: Audio array
            
        Returns:
            Tuple of (beat_times, tempo)
        """
        try:
            import librosa
            tempo, beats = librosa.beat.beat_track(y=audio, sr=self.sample_rate)
            beat_times = librosa.frames_to_time(beats, sr=self.sample_rate)
            return beat_times, float(tempo)
        except ImportError:
            logger.warning("librosa not available for beat detection")
            return np.array([]), 0.0
    
    def extract_features(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract audio features.
        
        Args:
            audio: Audio array
            
        Returns:
            Dictionary of feature arrays
        """
        features = {}
        try:
            import librosa
            features["mfcc"] = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=13)
            features["chroma"] = librosa.feature.chroma_stft(y=audio, sr=self.sample_rate)
            features["mel_spectrogram"] = librosa.feature.melspectrogram(y=audio, sr=self.sample_rate)
            features["spectral_contrast"] = librosa.feature.spectral_contrast(y=audio, sr=self.sample_rate)
        except ImportError:
            logger.warning("librosa not available for feature extraction")
        return features


# ════════════════════════════════════════════════════════════════════════════
# BACKWARD COMPATIBILITY FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════

def analyze_audio(
    audio: np.ndarray,
    sample_rate: int = DEFAULT_SAMPLE_RATE
) -> Dict[str, float]:
    """
    Analyze audio (backward compatibility).
    
    Args:
        audio: Audio array
        sample_rate: Sample rate
        
    Returns:
        Dictionary with audio statistics
    """
    analyzer = AudioAnalyzer(sample_rate=sample_rate)
    return analyzer.analyze(audio)


def detect_silence(
    audio: np.ndarray,
    threshold: float = 0.01,
    min_duration: float = 0.1,
    sample_rate: int = DEFAULT_SAMPLE_RATE
) -> list:
    """
    Detect silence (backward compatibility).
    
    Args:
        audio: Audio array
        threshold: Silence threshold
        min_duration: Minimum duration to consider as silence (seconds)
        sample_rate: Sample rate
        
    Returns:
        List of (start, end) tuples in seconds
    """
    analyzer = AudioAnalyzer(sample_rate=sample_rate)
    return analyzer.detect_silence(audio, threshold, min_duration)


def calculate_loudness(
    audio: np.ndarray,
    sample_rate: int = DEFAULT_SAMPLE_RATE
) -> Dict[str, float]:
    """
    Calculate loudness (backward compatibility).
    
    Args:
        audio: Audio array
        sample_rate: Sample rate
        
    Returns:
        Dictionary with loudness metrics
    """
    analyzer = AudioAnalyzer(sample_rate=sample_rate)
    return analyzer.calculate_loudness(audio)


def detect_beats(
    audio: np.ndarray,
    sample_rate: int = DEFAULT_SAMPLE_RATE
) -> Tuple[np.ndarray, float]:
    """
    Detect beats (backward compatibility).
    
    Args:
        audio: Audio array
        sample_rate: Sample rate
        
    Returns:
        Tuple of (beat_times, tempo)
    """
    analyzer = AudioAnalyzer(sample_rate=sample_rate)
    return analyzer.detect_beats(audio)


def extract_features(
    audio: np.ndarray,
    sample_rate: int = DEFAULT_SAMPLE_RATE
) -> Dict[str, np.ndarray]:
    """
    Extract features (backward compatibility).
    
    Args:
        audio: Audio array
        sample_rate: Sample rate
        
    Returns:
        Dictionary of feature arrays
    """
    analyzer = AudioAnalyzer(sample_rate=sample_rate)
    return analyzer.extract_features(audio)
