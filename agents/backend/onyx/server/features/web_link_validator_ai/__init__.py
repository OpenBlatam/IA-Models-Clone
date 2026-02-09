"""
Web Link Validator AI
=====================

Validates web links using AI and HTTP checks.
"""

__version__ = "1.0.0"
__author__ = "Blatam Academy"
__license__ = "Proprietary"
__description__ = "AI-powered web link validator with HTTP checks"

# Try to import components with error handling
try:
    from .services.link_validator import LinkValidator
except ImportError:
    LinkValidator = None

__all__ = [
    "LinkValidator",
]