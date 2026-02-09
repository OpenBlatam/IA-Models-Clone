"""
Debugging Utilities Module

NaN/Inf detection, gradient checking, anomaly detection.
"""

import sys
from pathlib import Path

# Import from parent directory
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from debugging_utils import (
    NaNInfDetector,
    GradientChecker,
    detect_anomaly,
    enable_debug_mode,
    disable_debug_mode
)

__all__ = [
    "NaNInfDetector",
    "GradientChecker",
    "detect_anomaly",
    "enable_debug_mode",
    "disable_debug_mode",
]

