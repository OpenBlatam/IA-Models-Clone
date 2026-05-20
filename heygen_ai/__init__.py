"""
HeyGen AI - Sistema de IA para generación de videos con avatares
=================================================================

Sistema completo para generar videos con avatares usando IA,
inspirado en HeyGen con capacidades avanzadas de síntesis de video.
"""

__version__ = "1.0.0"
__author__ = "Blatam Academy"
__license__ = "Proprietary"
__description__ = "AI system for generating videos with avatars, inspired by HeyGen with advanced video synthesis capabilities"

# Main entry point - try multiple locations
try:
    from .main import app
except ImportError:
    try:
        from .heygen_ai_main import app
    except ImportError:
        app = None

__all__ = ["app"]


