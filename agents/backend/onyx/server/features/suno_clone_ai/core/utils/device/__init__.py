"""
Device Management Module

Provides:
- Device selection and management
- GPU optimization
- Device utilities
"""

from .device_manager import (
    get_device,
    setup_gpu_optimizations,
    clear_gpu_cache,
    get_device_info
)

__all__ = [
    "get_device",
    "setup_gpu_optimizations",
    "clear_gpu_cache",
    "get_device_info"
]



