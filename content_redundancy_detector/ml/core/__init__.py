"""
Core Module
Core utilities and base classes
"""

from .exceptions import (
    MLFrameworkError,
    ModelError,
    TrainingError,
    InferenceError,
    ConfigurationError,
)
from .logging_setup import setup_logging, get_logger
from .version import __version__

__all__ = [
    "MLFrameworkError",
    "ModelError",
    "TrainingError",
    "InferenceError",
    "ConfigurationError",
    "setup_logging",
    "get_logger",
    "__version__",
]



