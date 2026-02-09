"""
Core Module - Foundation for Audio Separation Core.

Este módulo proporciona acceso organizado a los componentes core:
- Interfaces: Clases base y ABCs para arquitectura modular
- Exceptions: Jerarquía de excepciones
- Config: Gestión de configuración
- Factories: Implementaciones del patrón Factory
"""

from __future__ import annotations

# ════════════════════════════════════════════════════════════════════════════════
# BASE COMPONENT (Eager Import)
# ════════════════════════════════════════════════════════════════════════════════

from .base_component import BaseComponent

# ════════════════════════════════════════════════════════════════════════════════
# INTERFACES (Eager Import)
# ════════════════════════════════════════════════════════════════════════════════

from .interfaces import (
    IAudioComponent,
    IAudioSeparator,
    IAudioMixer,
    IAudioProcessor,
)

# ════════════════════════════════════════════════════════════════════════════════
# EXCEPTIONS (Eager Import)
# ════════════════════════════════════════════════════════════════════════════════

from .exceptions import (
    AudioSeparationError,
    AudioProcessingError,
    AudioFormatError,
    AudioModelError,
    AudioValidationError,
    AudioIOError,
)

# ════════════════════════════════════════════════════════════════════════════════
# CONFIG (Eager Import)
# ════════════════════════════════════════════════════════════════════════════════

from .config import (
    AudioConfig,
    SeparationConfig,
    MixingConfig,
    ProcessorConfig,
)

# ════════════════════════════════════════════════════════════════════════════════
# FACTORY HELPERS (Eager Import)
# ════════════════════════════════════════════════════════════════════════════════

from .registry import ComponentRegistry
from .loader import ComponentLoader
from .detector import SeparatorDetector

# ════════════════════════════════════════════════════════════════════════════════
# FACTORIES (Eager Import)
# ════════════════════════════════════════════════════════════════════════════════

from .factories import (
    create_audio_separator,
    create_audio_mixer,
    create_audio_processor,
    AudioSeparatorFactory,
    AudioMixerFactory,
    AudioProcessorFactory,
)

__all__ = [
    # Base Component
    "BaseComponent",
    # Interfaces
    "IAudioComponent",
    "IAudioSeparator",
    "IAudioMixer",
    "IAudioProcessor",
    # Exceptions
    "AudioSeparationError",
    "AudioProcessingError",
    "AudioFormatError",
    "AudioModelError",
    "AudioValidationError",
    "AudioIOError",
    # Config
    "AudioConfig",
    "SeparationConfig",
    "MixingConfig",
    "ProcessorConfig",
    # Factory Helpers (Refactored)
    "ComponentRegistry",
    "ComponentLoader",
    "SeparatorDetector",
    # Factories
    "create_audio_separator",
    "create_audio_mixer",
    "create_audio_processor",
    "AudioSeparatorFactory",
    "AudioMixerFactory",
    "AudioProcessorFactory",
]




