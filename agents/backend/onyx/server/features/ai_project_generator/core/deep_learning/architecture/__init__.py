"""
Architecture Module - Design Patterns and Architectural Components
==================================================================

Provides architectural patterns and components:
- Builder pattern
- Strategy pattern
- Observer pattern
- Factory pattern enhancements
"""

from typing import Optional, Dict, Any

from .builder import ModelBuilder, TrainingBuilder
from .strategy import (
    TrainingStrategy, StandardTrainingStrategy, FastTrainingStrategy,
    DataStrategy, StandardDataStrategy, CrossValidationDataStrategy
)
from .observer import Observer, EventPublisher, TrainingObserver, LoggingObserver

__all__ = [
    "ModelBuilder",
    "TrainingBuilder",
    "TrainingStrategy",
    "StandardTrainingStrategy",
    "FastTrainingStrategy",
    "DataStrategy",
    "StandardDataStrategy",
    "CrossValidationDataStrategy",
    "Observer",
    "EventPublisher",
    "TrainingObserver",
    "LoggingObserver",
]

