"""
Helpers Module - Utility Functions and Helpers
==============================================

Provides helper functions for common tasks:
- Model utilities
- Data utilities
- Training utilities
- Visualization utilities
"""

from typing import Optional, Dict, Any, List
import torch

from .model_helpers import (
    count_parameters,
    get_model_summary,
    freeze_layers,
    unfreeze_layers,
    save_model_onnx,
    load_model_onnx
)

# Try to import visualization
try:
    from .visualization import (
        plot_training_curves,
        plot_confusion_matrix
    )
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    plot_training_curves = None
    plot_confusion_matrix = None

__all__ = [
    "count_parameters",
    "get_model_summary",
    "freeze_layers",
    "unfreeze_layers",
    "save_model_onnx",
    "load_model_onnx",
]

if VISUALIZATION_AVAILABLE:
    __all__.extend(["plot_training_curves", "plot_confusion_matrix"])

