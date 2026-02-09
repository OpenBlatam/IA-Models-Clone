"""
Copywriting Service Module
===========================

AI-powered copywriting and content creation system.
"""

__version__ = "1.0.0"
__author__ = "Blatam Academy"
__license__ = "Proprietary"
__description__ = "AI-powered copywriting and content creation system"

# Try to import main components with error handling
try:
    from .api import router as copywriting_router
except ImportError:
    copywriting_router = None

__all__ = [
    "copywriting_router",
]