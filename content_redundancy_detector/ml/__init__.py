"""
ML Framework
Complete deep learning framework for MobileNet and related models
"""

from .core import (
    setup_logging,
    get_logger,
    MLFrameworkError,
    ModelError,
    TrainingError,
    InferenceError,
    ConfigurationError,
    __version__,
)

__version__ = __version__

__all__ = [
    "setup_logging",
    "get_logger",
    "MLFrameworkError",
    "ModelError",
    "TrainingError",
    "InferenceError",
    "ConfigurationError",
    "__version__",
]
