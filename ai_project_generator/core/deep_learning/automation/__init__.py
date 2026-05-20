"""
Automation Module - Automation Utilities
=========================================

Automation utilities:
- AutoML capabilities
- Automated hyperparameter tuning
- Automated model selection
- Automated pipeline generation
"""

from typing import Optional, Dict, Any, List

from .automation_utils import (
    AutoML,
    AutoTrainer,
    AutoPipeline,
    automated_model_selection
)

__all__ = [
    "AutoML",
    "AutoTrainer",
    "AutoPipeline",
    "automated_model_selection",
]

