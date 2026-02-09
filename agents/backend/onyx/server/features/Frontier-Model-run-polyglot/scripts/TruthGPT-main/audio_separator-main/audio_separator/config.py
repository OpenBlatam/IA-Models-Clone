"""
Configuration management for audio separator.
Refactored to use constants.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from pathlib import Path

from .separator.base_separator import DEFAULT_SAMPLE_RATE
from .processor.constants import DEFAULT_NORMALIZE, DEFAULT_TRIM_SILENCE, DEFAULT_SILENCE_THRESHOLD
from .model.constants import DEFAULT_NUM_SOURCES, DEFAULT_MODEL_TYPE
from .utils.constants import DEFAULT_DEVICE, DEFAULT_TARGET_PEAK


@dataclass
class AudioConfig:
    """Base configuration for audio processing."""
    sample_rate: int = DEFAULT_SAMPLE_RATE
    channels: int = 2  # 1 for mono, 2 for stereo
    normalize: bool = DEFAULT_NORMALIZE
    trim_silence: bool = DEFAULT_TRIM_SILENCE
    silence_threshold: float = DEFAULT_SILENCE_THRESHOLD


@dataclass
class SeparationConfig(AudioConfig):
    """Configuration for audio separation."""
    num_sources: int = DEFAULT_NUM_SOURCES
    overlap: float = 0.25  # Overlap for chunked processing
    chunk_size: Optional[int] = None  # None for no chunking
    device: str = DEFAULT_DEVICE  # "auto", "cpu", "cuda"
    batch_size: int = 1


@dataclass
class ModelConfig:
    """Configuration for separation models."""
    model_type: str = DEFAULT_MODEL_TYPE
    model_kwargs: Dict[str, Any] = field(default_factory=dict)
    cache_dir: Optional[Path] = None
    download_models: bool = True


@dataclass
class ProcessingConfig:
    """Configuration for audio processing."""
    preprocess: bool = True
    postprocess: bool = True
    denoise: bool = False
    normalize_output: bool = DEFAULT_NORMALIZE


@dataclass
class OutputConfig:
    """Configuration for output files."""
    output_format: str = "wav"  # wav, mp3, flac
    output_dir: Optional[Path] = None
    naming_pattern: str = "{original_name}_{source_name}.{ext}"
    create_subdirs: bool = True
    overwrite: bool = False


@dataclass
class AudioSeparatorConfig:
    """Complete configuration for AudioSeparator."""
    audio: AudioConfig = field(default_factory=AudioConfig)
    separation: SeparationConfig = field(default_factory=SeparationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "AudioSeparatorConfig":
        """Create config from dictionary."""
        audio_config = AudioConfig(**config_dict.get("audio", {}))
        separation_config = SeparationConfig(
            **config_dict.get("separation", {}),
            **{k: v for k, v in audio_config.__dict__.items() 
               if k in SeparationConfig.__annotations__}
        )
        model_config = ModelConfig(**config_dict.get("model", {}))
        processing_config = ProcessingConfig(**config_dict.get("processing", {}))
        output_config = OutputConfig(**config_dict.get("output", {}))
        
        return cls(
            audio=audio_config,
            separation=separation_config,
            model=model_config,
            processing=processing_config,
            output=output_config
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "audio": self.audio.__dict__,
            "separation": self.separation.__dict__,
            "model": self.model.__dict__,
            "processing": self.processing.__dict__,
            "output": self.output.__dict__
        }

