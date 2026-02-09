"""
Presets Module - Pre-configured Settings and Templates
======================================================

Provides pre-configured presets for common use cases:
- Model presets
- Training presets
- Data presets
- Optimizer presets
"""

from typing import Dict, Any

from .presets import (
    get_model_preset,
    get_training_preset,
    get_optimizer_preset,
    get_data_preset,
    list_presets
)

__all__ = [
    "get_model_preset",
    "get_training_preset",
    "get_optimizer_preset",
    "get_data_preset",
    "list_presets",
]

