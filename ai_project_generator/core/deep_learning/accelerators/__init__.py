"""
Accelerators Module - Hardware Acceleration Utilities
====================================================

Utilities for hardware acceleration:
- GPU optimization
- TPU support
- Multi-GPU strategies
- Mixed precision utilities
"""

from typing import Optional, Dict, Any

from .accelerator_utils import (
    setup_accelerator,
    optimize_for_gpu,
    setup_multi_gpu,
    get_accelerator_info,
    enable_mixed_precision,
    optimize_batch_size
)

__all__ = [
    "setup_accelerator",
    "optimize_for_gpu",
    "setup_multi_gpu",
    "get_accelerator_info",
    "enable_mixed_precision",
    "optimize_batch_size",
]

