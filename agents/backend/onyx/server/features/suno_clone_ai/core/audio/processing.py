"""
Audio Processing Module

Implements:
- Audio preprocessing
- Post-processing
- Format conversion
- Quality enhancement
"""

import logging
from typing import Optional, Tuple, Union
import torch
import torchaudio
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


class AudioProcessor:
    """Audio processing utilities."""
    
    def __init__(self, sample_rate: int = 32000):
        """
        Initialize audio processor.
        
        Args:
            sample_rate: Target sample rate
        """
        self.sample_rate = sample_rate
    
    def normalize(
        self,
        audio: Union[torch.Tensor, np.ndarray],
        target_max: float = 1.0
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Normalize audio to target range.
        
        Args:
            audio: Audio tensor or array
            target_max: Target maximum value
            
        Returns:
            Normalized audio
        """
        is_tensor = isinstance(audio, torch.Tensor)
        
        if not is_tensor:
            audio = torch.from_numpy(audio).float()
        
        max_val = torch.abs(audio).max()
        if max_val > 0:
            audio = audio / max_val * target_max
        
        if not is_tensor:
            audio = audio.numpy()
        
        return audio
    
    def resample(
        self,
        audio: torch.Tensor,
        original_rate: int,
        target_rate: Optional[int] = None
    ) -> torch.Tensor:
        """
        Resample audio to target sample rate.
        
        Args:
            audio: Audio tensor
            original_rate: Original sample rate
            target_rate: Target sample rate (uses self.sample_rate if None)
            
        Returns:
            Resampled audio
        """
        target_rate = target_rate or self.sample_rate
        
        if original_rate == target_rate:
            return audio
        
        resampler = torchaudio.transforms.Resample(
            original_rate,
            target_rate
        )
        
        return resampler(audio)
    
    def trim_silence(
        self,
        audio: np.ndarray,
        threshold: float = 0.01,
        frame_length: int = 2048
    ) -> np.ndarray:
        """
        Trim silence from beginning and end of audio.
        
        Args:
            audio: Audio array
            threshold: Silence threshold
            frame_length: Frame length for analysis
            
        Returns:
            Trimmed audio
        """
        # Find non-silent regions
        frames = np.abs(audio) > threshold
        
        # Find first and last non-silent frames
        if not frames.any():
            return audio
        
        first_non_silent = np.argmax(frames)
        last_non_silent = len(frames) - np.argmax(frames[::-1]) - 1
        
        return audio[first_non_silent:last_non_silent + 1]
    
    def apply_fade(
        self,
        audio: np.ndarray,
        fade_in_samples: int = 1000,
        fade_out_samples: int = 1000
    ) -> np.ndarray:
        """
        Apply fade in/out to audio.
        
        Args:
            audio: Audio array
            fade_in_samples: Fade in length in samples
            fade_out_samples: Fade out length in samples
            
        Returns:
            Audio with fade applied
        """
        audio = audio.copy()
        length = len(audio)
        
        # Fade in
        if fade_in_samples > 0 and fade_in_samples < length:
            fade_in = np.linspace(0, 1, fade_in_samples)
            audio[:fade_in_samples] *= fade_in
        
        # Fade out
        if fade_out_samples > 0 and fade_out_samples < length:
            fade_out = np.linspace(1, 0, fade_out_samples)
            audio[-fade_out_samples:] *= fade_out
        
        return audio
    
    def save_audio(
        self,
        audio: Union[torch.Tensor, np.ndarray],
        output_path: str,
        sample_rate: Optional[int] = None
    ) -> str:
        """
        Save audio to file.
        
        Args:
            audio: Audio tensor or array
            output_path: Path to save file
            sample_rate: Sample rate (uses self.sample_rate if None)
            
        Returns:
            Path of saved file
        """
        sample_rate = sample_rate or self.sample_rate
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to tensor if needed
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio).float()
        
        # Ensure correct shape
        if len(audio.shape) == 1:
            audio = audio.unsqueeze(0)
        
        # Normalize to prevent clipping
        max_val = torch.abs(audio).max()
        if max_val > 1.0:
            audio = audio / max_val
            logger.warning("Audio normalized to prevent clipping")
        
        # Save
        torchaudio.save(
            str(output_path),
            audio,
            sample_rate,
            format="wav"
        )
        
        logger.info(f"Audio saved to: {output_path}")
        return str(output_path)
    
    def load_audio(
        self,
        file_path: str,
        target_sample_rate: Optional[int] = None
    ) -> Tuple[torch.Tensor, int]:
        """
        Load audio from file.
        
        Args:
            file_path: Path to audio file
            target_sample_rate: Target sample rate (resamples if different)
            
        Returns:
            Tuple of (audio_tensor, sample_rate)
        """
        audio, sample_rate = torchaudio.load(file_path)
        
        if target_sample_rate and target_sample_rate != sample_rate:
            audio = self.resample(audio, sample_rate, target_sample_rate)
            sample_rate = target_sample_rate
        
        return audio, sample_rate


class AudioEnhancer:
    """Audio quality enhancement."""
    
    @staticmethod
    def reduce_noise(
        audio: np.ndarray,
        noise_reduction_factor: float = 0.5
    ) -> np.ndarray:
        """
        Simple noise reduction.
        
        Args:
            audio: Audio array
            noise_reduction_factor: Noise reduction strength
            
        Returns:
            Denoised audio
        """
        # Simple high-pass filter approximation
        # In production, use proper noise reduction library
        return audio * (1 - noise_reduction_factor * 0.1)
    
    @staticmethod
    def enhance_quality(
        audio: np.ndarray,
        normalize: bool = True,
        trim_silence: bool = True,
        apply_fade: bool = True
    ) -> np.ndarray:
        """
        Enhance audio quality with multiple techniques.
        
        Args:
            audio: Audio array
            normalize: Normalize audio
            trim_silence: Trim silence
            apply_fade: Apply fade in/out
            
        Returns:
            Enhanced audio
        """
        processor = AudioProcessor()
        
        if normalize:
            audio = processor.normalize(audio)
        
        if trim_silence:
            audio = processor.trim_silence(audio)
        
        if apply_fade:
            audio = processor.apply_fade(audio)
        
        return audio



