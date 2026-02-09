"""
Debugging Module

Provides:
- Gradient debugging
- NaN/Inf detection
- Model inspection
- Debugging utilities
"""

from .gradient_debug import (
    GradientDebugger,
    check_gradients,
    log_gradient_norms
)

from .nan_detector import (
    NaNDetector,
    check_for_nan_inf,
    detect_nan_in_model
)

from .model_inspector import (
    ModelInspector,
    inspect_model,
    count_parameters,
    get_model_summary
)

__all__ = [
    # Gradient debugging
    "GradientDebugger",
    "check_gradients",
    "log_gradient_norms",
    # NaN detection
    "NaNDetector",
    "check_for_nan_inf",
    "detect_nan_in_model",
    # Model inspection
    "ModelInspector",
    "inspect_model",
    "count_parameters",
    "get_model_summary"
]



