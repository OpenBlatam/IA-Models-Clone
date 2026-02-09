"""
Brand Voice Feature Module
===========================

AI-powered brand voice analysis and generation system.
"""

__version__ = "1.0.0"
__author__ = "Blatam Academy"
__license__ = "Proprietary"
__description__ = "AI-powered brand voice analysis and generation system"

# Try to import main components with error handling
try:
    from .api import router as brand_voice_router
except ImportError:
    brand_voice_router = None

__all__ = [
    "brand_voice_router",
]
