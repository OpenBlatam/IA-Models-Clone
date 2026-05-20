"""
Core Components Module
======================

Core interfaces and base classes for the LLM trainer system.
Provides abstract base classes for extensibility.

Author: BUL System
Date: 2024
"""

from .base_trainer import BaseLLMTrainer
from .trainer_factory import TrainerFactory
from .config_builder import ConfigBuilder

__all__ = [
    "BaseLLMTrainer",
    "TrainerFactory",
    "ConfigBuilder",
]

