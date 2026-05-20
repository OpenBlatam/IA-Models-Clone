"""
AI Video - Sistema de generación y procesamiento de video con IA
================================================================

Sistema completo para generación, procesamiento y análisis de videos usando IA.
"""

__version__ = "1.0.0"
__author__ = "Blatam Academy"
__license__ = "Proprietary"
__description__ = "Complete AI-powered video generation, processing and analysis system"

# Try to import components with error handling
try:
    from .api import app
except ImportError:
    app = None

__all__ = ["app"]

