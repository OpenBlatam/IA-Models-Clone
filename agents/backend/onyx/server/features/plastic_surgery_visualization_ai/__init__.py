"""
Plastic Surgery Visualization AI
=================================

AI system that shows how you'll look after plastic surgery procedures.
"""

__version__ = "1.0.0"
__author__ = "Blatam Academy"
__license__ = "Proprietary"
__description__ = "AI system for visualizing plastic surgery results"

# Try to import components with error handling
try:
    from .core.app_factory import create_app
except ImportError:
    create_app = None

__all__ = [
    "create_app",
]