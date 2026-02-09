"""
Markdown to Professional Documents AI
======================================

Convert Markdown to professional formats (PDF, DOCX, HTML, etc.)
"""

__version__ = "1.0.0"
__author__ = "Blatam Academy"
__license__ = "Proprietary"
__description__ = "AI-powered Markdown to professional document converter"

# Try to import components with error handling
try:
    from .services.converter_service import ConverterService
except ImportError:
    ConverterService = None

try:
    from .api import (
        root_router,
        health_router,
        conversion_router,
        formats_router,
    )
except ImportError:
    root_router = None
    health_router = None
    conversion_router = None
    formats_router = None

__all__ = [
    "ConverterService",
    "root_router",
    "health_router",
    "conversion_router",
    "formats_router",
]