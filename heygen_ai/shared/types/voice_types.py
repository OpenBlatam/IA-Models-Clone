"""
Voice Types
===========

Data types for voice generation.
"""

from dataclasses import dataclass, field
from typing import Any, Dict

from shared.enums import VoiceQuality, AudioFormat


@dataclass
class VoiceGenerationConfig:
    """Configuration for voice generation.
    
    Attributes:
        sample_rate: Audio sample rate in Hz
        bit_depth: Audio bit depth (16, 24, 32)
        channels: Number of audio channels (1=mono, 2=stereo)
        format: Output audio format
        quality: Generation quality level
        normalize: Normalize audio levels
        remove_silence: Remove leading/trailing silence
        speed: Speech speed multiplier (0.5-2.0)
        pitch: Pitch shift in semitones (-12 to 12)
        volume: Volume multiplier (0.0-2.0)
    """
    sample_rate: int = 22050
    bit_depth: int = 16
    channels: int = 1
    format: AudioFormat = AudioFormat.WAV
    quality: VoiceQuality = VoiceQuality.HIGH
    normalize: bool = True
    remove_silence: bool = True
    speed: float = 1.0
    pitch: float = 0.0
    volume: float = 1.0

    def get_sample_rate(self) -> int:
        """Get sample rate based on quality."""
        quality_rates = {
            VoiceQuality.LOW: 16000,
            VoiceQuality.MEDIUM: 22050,
            VoiceQuality.HIGH: 22050,
            VoiceQuality.ULTRA: 44100,
        }
        return quality_rates.get(self.quality, self.sample_rate)

    def validate(self) -> None:
        """Validate configuration parameters."""
        if not 0.5 <= self.speed <= 2.0:
            raise ValueError("Speed must be between 0.5 and 2.0")
        if not -12 <= self.pitch <= 12:
            raise ValueError("Pitch must be between -12 and 12 semitones")
        if not 0.0 <= self.volume <= 2.0:
            raise ValueError("Volume must be between 0.0 and 2.0")


@dataclass
class VoiceModel:
    """Voice model configuration.
    
    Attributes:
        id: Unique identifier
        name: Display name
        language: Language code (ISO 639-1)
        model_path: Path to model or HuggingFace model ID
        supports_cloning: Whether model supports voice cloning
        supports_emotion: Whether model supports emotion control
    """
    id: str
    name: str
    language: str
    model_path: str
    supports_cloning: bool = False
    supports_emotion: bool = False
    characteristics: Dict[str, Any] = field(default_factory=dict)



