"""
Audio Separator - Advanced Audio Source Separation Framework

Refactored to provide a clean, organized API with:
- Centralized constants
- Base classes for extensibility
- Factory patterns
- Comprehensive error handling
"""

# ════════════════════════════════════════════════════════════════════════════
# VERSION
# ════════════════════════════════════════════════════════════════════════════

__version__ = "0.1.0"

# ════════════════════════════════════════════════════════════════════════════
# MODEL BUILDER
# ════════════════════════════════════════════════════════════════════════════

from .model_builder import (
    build_audio_separator_model,
    register_model,
    get_registered_models
)

# ════════════════════════════════════════════════════════════════════════════
# SEPARATORS
# ════════════════════════════════════════════════════════════════════════════

from .separator.audio_separator import AudioSeparator
from .separator.batch_separator import BatchSeparator

# ════════════════════════════════════════════════════════════════════════════
# FACTORIES
# ════════════════════════════════════════════════════════════════════════════

from .factories.separator_factory import SeparatorFactory

# ════════════════════════════════════════════════════════════════════════════
# CORE COMPONENTS
# ════════════════════════════════════════════════════════════════════════════

from .core.base_component import BaseComponent
from .core.resource_manager import ResourceManager

# ════════════════════════════════════════════════════════════════════════════
# EXCEPTIONS
# ════════════════════════════════════════════════════════════════════════════

from .exceptions import (
    AudioSeparatorError,
    AudioProcessingError,
    AudioFormatError,
    AudioModelError,
    AudioValidationError,
    AudioIOError,
    AudioInitializationError,
    AudioConfigurationError
)

# ════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ════════════════════════════════════════════════════════════════════════════

__all__ = [
    # Version
    "__version__",
    # Model builder
    "build_audio_separator_model",
    "register_model",
    "get_registered_models",
    # Separators
    "AudioSeparator",
    "BatchSeparator",
    # Factory
    "SeparatorFactory",
    # Core components
    "BaseComponent",
    "ResourceManager",
    # Exceptions
    "AudioSeparatorError",
    "AudioProcessingError",
    "AudioFormatError",
    "AudioModelError",
    "AudioValidationError",
    "AudioIOError",
    "AudioInitializationError",
    "AudioConfigurationError",
]

