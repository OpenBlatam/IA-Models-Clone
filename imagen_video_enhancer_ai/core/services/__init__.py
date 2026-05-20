"""
Service Handlers
================

Service handlers for different enhancement types.
"""

from .base_handler import BaseServiceHandler
from .enhance_image_handler import EnhanceImageHandler
from .enhance_video_handler import EnhanceVideoHandler
from .upscale_handler import UpscaleHandler
from .denoise_handler import DenoiseHandler
from .restore_handler import RestoreHandler
from .color_correction_handler import ColorCorrectionHandler

__all__ = [
    "BaseServiceHandler",
    "EnhanceImageHandler",
    "EnhanceVideoHandler",
    "UpscaleHandler",
    "DenoiseHandler",
    "RestoreHandler",
    "ColorCorrectionHandler",
]




