"""
Data Augmentation Module

Provides:
- Audio augmentation
- Text augmentation
- Augmentation pipelines
- Augmentation strategies
"""

from .audio_augmentation import (
    AudioAugmenter,
    TimeStretch,
    PitchShift,
    AddNoise,
    TimeMasking,
    FrequencyMasking,
    create_audio_augmentation_pipeline
)

from .text_augmentation import (
    TextAugmenter,
    SynonymReplacement,
    RandomInsertion,
    RandomDeletion,
    create_text_augmentation_pipeline
)

__all__ = [
    # Audio augmentation
    "AudioAugmenter",
    "TimeStretch",
    "PitchShift",
    "AddNoise",
    "TimeMasking",
    "FrequencyMasking",
    "create_audio_augmentation_pipeline",
    # Text augmentation
    "TextAugmenter",
    "SynonymReplacement",
    "RandomInsertion",
    "RandomDeletion",
    "create_text_augmentation_pipeline"
]



