"""
Suno Clone AI - Sistema de generación de música con IA
======================================================

Permite a los usuarios crear canciones mediante chat, similar a Suno AI.
"""

__version__ = "1.0.0"
__author__ = "Blatam Academy"
__license__ = "Proprietary"
__description__ = "AI-powered music generation system similar to Suno AI"

# Try to import components with error handling
try:
    from .server import create_app, get_app, initialize_server
except ImportError:
    create_app = None
    get_app = None
    initialize_server = None

__all__ = [
    "create_app",
    "get_app",
    "initialize_server",
]
