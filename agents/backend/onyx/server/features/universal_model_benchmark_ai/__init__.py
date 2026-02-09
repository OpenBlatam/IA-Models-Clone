"""
Universal Model Benchmark AI - Sistema de Benchmarking de Modelos de IA
========================================================================

Sistema completo para evaluar y comparar el rendimiento de diferentes modelos de IA.
"""

__version__ = "1.0.0"
__author__ = "Blatam Academy"
__license__ = "Proprietary"
__description__ = "AI system for benchmarking and comparing AI model performance"

# Main entry point with error handling
try:
    from .python.api.rest_api import create_app, app
except ImportError:
    create_app = None
    app = None

__all__ = ["create_app", "app"]

