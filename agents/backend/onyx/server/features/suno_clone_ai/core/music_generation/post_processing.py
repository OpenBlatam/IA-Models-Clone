"""
Post-Processing Module

Professional audio post-processing utilities.
"""

from typing import Optional, List, Union
import numpy as np
import logging

logger = logging.getLogger(__name__)


class AudioPostProcessor:
    """
    Professional audio post-processing pipeline.
    """
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self._pedalboard = None
        self._noise_reducer = None
    
    def _init_pedalboard(self):
        """Initialize pedalboard effects."""
        try:
            import pedalboard
            from pedalboard import Reverb, Compressor, Gain, Delay
            
            if self._pedalboard is None:
                self._pedalboard = pedalboard.Pedalboard([
                    Compressor(threshold_db=-20, ratio=4),
                    Gain(gain_db=2),
                    Reverb(room_size=0.5, wet_level=0.2),
                    Delay(delay_seconds=0.1, feedback=0.3)
                ])
            return self._pedalboard
        except ImportError:
            logger.warning("pedalboard not available")
            return None
    
    def apply_reverb(
        self,
        audio: np.ndarray,
        room_size: float = 0.5,
        wet_level: float = 0.2
    ) -> np.ndarray:
        """Apply reverb effect."""
        try:
            import pedalboard
            from pedalboard import Reverb
            
            board = pedalboard.Pedalboard([
                Reverb(room_size=room_size, wet_level=wet_level)
            ])
            return board(audio, sample_rate=self.sample_rate)
        except ImportError:
            logger.warning("pedalboard not available, skipping reverb")
            return audio
    
    def apply_compression(
        self,
        audio: np.ndarray,
        threshold_db: float = -20,
        ratio: float = 4
    ) -> np.ndarray:
        """Apply compression."""
        try:
            import pedalboard
            from pedalboard import Compressor
            
            board = pedalboard.Pedalboard([
                Compressor(threshold_db=threshold_db, ratio=ratio)
            ])
            return board(audio, sample_rate=self.sample_rate)
        except ImportError:
            logger.warning("pedalboard not available, skipping compression")
            return audio
    
    def reduce_noise(
        self,
        audio: np.ndarray,
        stationary: bool = False,
        prop_decrease: float = 0.8
    ) -> np.ndarray:
        """Reduce noise in audio."""
        try:
            import noisereduce as nr
            
            return nr.reduce_noise(
                y=audio,
                sr=self.sample_rate,
                stationary=stationary,
                prop_decrease=prop_decrease,
                n_jobs=2
            )
        except ImportError:
            logger.warning("noisereduce not available, skipping noise reduction")
            return audio
    
    def enhance_with_deepfilter(
        self,
        audio: np.ndarray
    ) -> np.ndarray:
        """Enhance audio with DeepFilterNet."""
        try:
            from deepfilternet import DeepFilterNet
            
            model = DeepFilterNet()
            return model.process(audio, sample_rate=self.sample_rate)
        except ImportError:
            logger.warning("deepfilternet not available, skipping enhancement")
            return audio
    
    def normalize(
        self,
        audio: np.ndarray,
        norm_type: str = "inf"
    ) -> np.ndarray:
        """Normalize audio."""
        try:
            import librosa
            
            if norm_type == "inf":
                return librosa.util.normalize(audio, norm=np.inf)
            elif norm_type == "l2":
                return librosa.util.normalize(audio, norm=2)
            else:
                return librosa.util.normalize(audio)
        except ImportError:
            # Fallback normalization
            max_val = np.max(np.abs(audio))
            if max_val > 0:
                return audio / max_val
            return audio
    
    def process_full_pipeline(
        self,
        audio: np.ndarray,
        apply_noise_reduction: bool = True,
        apply_reverb: bool = True,
        apply_compression: bool = True,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Apply full post-processing pipeline.
        
        Args:
            audio: Input audio array
            apply_noise_reduction: Apply noise reduction
            apply_reverb: Apply reverb
            apply_compression: Apply compression
            normalize: Normalize audio
            
        Returns:
            Processed audio
        """
        processed = audio.copy()
        
        # Convert to mono if stereo
        if len(processed.shape) > 1:
            processed = np.mean(processed, axis=0, dtype=np.float32)
        
        # Noise reduction
        if apply_noise_reduction:
            processed = self.reduce_noise(processed)
        
        # Compression
        if apply_compression:
            processed = self.apply_compression(processed)
        
        # Reverb
        if apply_reverb:
            processed = self.apply_reverb(processed)
        
        # Normalize
        if normalize:
            processed = self.normalize(processed)
        
        return processed


class TimeStretchProcessor:
    """Time stretching and pitch shifting utilities."""
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
    
    def time_stretch(
        self,
        audio: np.ndarray,
        time_ratio: float
    ) -> np.ndarray:
        """
        Stretch or compress audio in time.
        
        Args:
            audio: Input audio
            time_ratio: Ratio > 1.0 stretches, < 1.0 compresses
            
        Returns:
            Time-stretched audio
        """
        try:
            import pyRubberBand as rubberband
            
            return rubberband.time_stretch(
                audio,
                sample_rate=self.sample_rate,
                time_ratio=time_ratio
            )
        except ImportError:
            # Fallback using librosa
            try:
                import librosa
                return librosa.effects.time_stretch(audio, rate=time_ratio)
            except ImportError:
                logger.warning("No time stretching library available")
                return audio
    
    def pitch_shift(
        self,
        audio: np.ndarray,
        semitones: float
    ) -> np.ndarray:
        """
        Shift pitch of audio.
        
        Args:
            audio: Input audio
            semitones: Number of semitones to shift (positive = higher)
            
        Returns:
            Pitch-shifted audio
        """
        try:
            import pyRubberBand as rubberband
            
            return rubberband.pitch_shift(
                audio,
                sample_rate=self.sample_rate,
                semitones=semitones
            )
        except ImportError:
            # Fallback using librosa
            try:
                import librosa
                return librosa.effects.pitch_shift(
                    audio,
                    sr=self.sample_rate,
                    n_steps=semitones
                )
            except ImportError:
                logger.warning("No pitch shifting library available")
                return audio















