"""
Structured Logging Module

Provides:
- Structured logging
- Log formatting
- Log handlers
- Training loggers
"""

from .structured_logger import (
    StructuredLogger,
    setup_logging,
    get_logger
)

from .training_logger import (
    TrainingLogger,
    log_training_step,
    log_epoch_summary
)

__all__ = [
    # Structured logging
    "StructuredLogger",
    "setup_logging",
    "get_logger",
    # Training logging
    "TrainingLogger",
    "log_training_step",
    "log_epoch_summary"
]



