"""
Mixing and Mastering Module

Professional mixing and mastering utilities.
"""

from typing import Optional, List, Dict, Any
import numpy as np
import logging

logger = logging.getLogger(__name__)


class AudioMixer:
    """
    Professional audio mixing utilities.
    """
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
    
    def mix_tracks(
        self,
        tracks: List[np.ndarray],
        volumes: Optional[List[float]] = None,
        panning: Optional[List[float]] = None
    ) -> np.ndarray:
        """
        Mix multiple audio tracks.
        
        Args:
            tracks: List of audio arrays
            volumes: Optional volume levels (0.0 to 1.0) for each track
            panning: Optional panning (-1.0 left to 1.0 right) for each track
            
        Returns:
            Mixed audio
        """
        if not tracks:
            raise ValueError("No tracks provided")
        
        # Normalize lengths
        max_length = max(len(track) for track in tracks)
        normalized_tracks = []
        
        for i, track in enumerate(tracks):
            # Convert to mono if stereo
            if len(track.shape) > 1:
                track = np.mean(track, axis=0)
            
            # Pad or trim to max length
            if len(track) < max_length:
                track = np.pad(track, (0, max_length - len(track)))
            else:
                track = track[:max_length]
            
            # Apply volume
            volume = volumes[i] if volumes and i < len(volumes) else 1.0
            track = track * volume
            
            # Apply panning (convert to stereo if needed)
            if panning and i < len(panning):
                pan = panning[i]
                # Simple panning: -1.0 = left, 0.0 = center, 1.0 = right
                left_gain = np.sqrt((1 - pan) / 2)
                right_gain = np.sqrt((1 + pan) / 2)
                track = np.array([track * left_gain, track * right_gain])
            else:
                # Keep mono or convert to stereo
                if len(track.shape) == 1:
                    track = np.array([track, track])
            
            normalized_tracks.append(track)
        
        # Mix all tracks
        mixed = np.sum(normalized_tracks, axis=0)
        
        # Normalize to prevent clipping
        max_val = np.max(np.abs(mixed))
        if max_val > 1.0:
            mixed = mixed / max_val
        
        return mixed
    
    def apply_eq(
        self,
        audio: np.ndarray,
        low_gain: float = 0.0,
        mid_gain: float = 0.0,
        high_gain: float = 0.0
    ) -> np.ndarray:
        """
        Apply simple 3-band EQ.
        
        Args:
            audio: Audio array
            low_gain: Gain for low frequencies (dB)
            mid_gain: Gain for mid frequencies (dB)
            high_gain: Gain for high frequencies (dB)
            
        Returns:
            EQ'd audio
        """
        try:
            import scipy.signal
            
            # Convert dB to linear
            low_linear = 10 ** (low_gain / 20)
            mid_linear = 10 ** (mid_gain / 20)
            high_linear = 10 ** (high_gain / 20)
            
            # Simple filter implementation
            # Low shelf
            if low_gain != 0:
                b, a = scipy.signal.iirfilter(
                    2, 200 / (self.sample_rate / 2),
                    btype='low', ftype='butter'
                )
                audio = scipy.signal.filtfilt(b, a, audio) * low_linear
            
            # High shelf
            if high_gain != 0:
                b, a = scipy.signal.iirfilter(
                    2, 5000 / (self.sample_rate / 2),
                    btype='high', ftype='butter'
                )
                audio = scipy.signal.filtfilt(b, a, audio) * high_linear
            
            # Mid (bandpass)
            if mid_gain != 0:
                b, a = scipy.signal.iirfilter(
                    2, [200 / (self.sample_rate / 2), 5000 / (self.sample_rate / 2)],
                    btype='band', ftype='butter'
                )
                audio = scipy.signal.filtfilt(b, a, audio) * mid_linear
            
            return audio
        except ImportError:
            logger.warning("scipy not available, skipping EQ")
            return audio
    
    def apply_limiter(
        self,
        audio: np.ndarray,
        threshold: float = 0.95,
        release: float = 0.01
    ) -> np.ndarray:
        """
        Apply limiter to prevent clipping.
        
        Args:
            audio: Audio array
            threshold: Limiter threshold (0.0 to 1.0)
            release: Release time in seconds
            
        Returns:
            Limited audio
        """
        limited = audio.copy()
        
        # Simple limiter: hard clip above threshold
        above_threshold = np.abs(limited) > threshold
        if np.any(above_threshold):
            # Soft limiting
            limited[above_threshold] = np.sign(limited[above_threshold]) * (
                threshold + (np.abs(limited[above_threshold]) - threshold) * 0.1
            )
        
        return limited


class StemSeparator:
    """
    Separate audio into stems (vocals, drums, bass, other).
    """
    
    def __init__(self, model_name: str = "htdemucs"):
        self.model_name = model_name
        self.model = None
        self._initialized = False
    
    def initialize(self) -> None:
        """Initialize stem separation model."""
        try:
            import torch
            from demucs import pretrained
            
            logger.info(f"Loading stem separation model: {self.model_name}")
            self.model = pretrained.get_model(self.model_name)
            self.model.eval()
            self._initialized = True
            logger.info("Stem separation model loaded")
        except ImportError:
            raise ImportError(
                "demucs not installed. Install with: pip install demucs"
            )
        except Exception as e:
            logger.error(f"Error loading stem separation model: {e}")
            raise
    
    def separate(
        self,
        audio: np.ndarray,
        sample_rate: int = 44100
    ) -> Dict[str, np.ndarray]:
        """
        Separate audio into stems.
        
        Args:
            audio: Audio array
            sample_rate: Sample rate
            
        Returns:
            Dictionary with separated stems
        """
        if not self._initialized:
            self.initialize()
        
        try:
            import torch
            
            # Prepare audio
            if len(audio.shape) == 1:
                audio = np.array([audio, audio])  # Convert to stereo
            
            # Convert to tensor
            audio_tensor = torch.from_numpy(audio).float()
            
            # Separate
            with torch.no_grad():
                stems = self.model.separate(audio_tensor)
            
            # Convert back to numpy
            result = {}
            stem_names = ["drums", "bass", "other", "vocals"]
            
            for i, name in enumerate(stem_names):
                if i < len(stems):
                    stem_audio = stems[i].cpu().numpy()
                    # Convert to mono if needed
                    if len(stem_audio.shape) > 1:
                        stem_audio = np.mean(stem_audio, axis=0)
                    result[name] = stem_audio
            
            return result
        except Exception as e:
            logger.error(f"Error separating stems: {e}")
            raise















