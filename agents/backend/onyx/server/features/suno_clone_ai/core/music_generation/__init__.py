"""
Music Generation Module

Unified interface for music generation, post-processing, and voice synthesis.
"""

from .generators import (
    BaseMusicGenerator,
    AudiocraftGenerator,
    MusicGenHuggingFaceGenerator,
    StableAudioGenerator,
    create_generator
)

from .post_processing import (
    AudioPostProcessor,
    TimeStretchProcessor
)

from .voice_synthesis import (
    VoiceSynthesizer,
    VoiceCloner
)

from .analysis import (
    AudioAnalyzer,
    BeatTracker
)

from .mixing import (
    AudioMixer,
    StemSeparator
)

from .optimization import (
    GPUOptimizer,
    MemoryOptimizer,
    BatchOptimizer,
    ModelCache,
    get_model_cache
)

from .pipeline import (
    MusicGenerationPipeline,
    VoiceMusicPipeline
)

from .async_generator import (
    AsyncMusicGenerator,
    AsyncPipeline
)

from .export import (
    AudioExporter
)

__all__ = [
    # Generators
    "BaseMusicGenerator",
    "AudiocraftGenerator",
    "MusicGenHuggingFaceGenerator",
    "StableAudioGenerator",
    "create_generator",
    # Post-processing
    "AudioPostProcessor",
    "TimeStretchProcessor",
    # Voice synthesis
    "VoiceSynthesizer",
    "VoiceCloner",
    # Analysis
    "AudioAnalyzer",
    "BeatTracker",
    # Mixing
    "AudioMixer",
    "StemSeparator",
    # Optimization
    "GPUOptimizer",
    "MemoryOptimizer",
    "BatchOptimizer",
    "ModelCache",
    "get_model_cache",
    # Pipeline
    "MusicGenerationPipeline",
    "VoiceMusicPipeline",
    # Async
    "AsyncMusicGenerator",
    "AsyncPipeline",
    # Export
    "AudioExporter",
]

