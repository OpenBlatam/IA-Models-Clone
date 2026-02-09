"""
Web Content Extractor AI
========================

Extrae información completa de páginas web usando OpenRouter.
"""

__version__ = "1.0.0"
__author__ = "Blatam Academy"
__license__ = "Proprietary"
__description__ = "AI-powered web content extraction using OpenRouter"

# Try to import components with error handling
try:
    from .core.app_factory import create_app
except ImportError:
    create_app = None

try:
    from .api.v1.routes import router
except ImportError:
    router = None

__all__ = [
    "create_app",
    "router",
]
