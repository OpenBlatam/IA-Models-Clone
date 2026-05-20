"""
Logistics AI Platform - A comprehensive freight forwarding system
==================================================================

Sistema completo de gestión de logística y transporte de carga.
"""

__version__ = "1.0.0"
__author__ = "Blatam Academy"
__license__ = "Proprietary"
__description__ = "Comprehensive AI-powered freight forwarding and logistics management system"

# Try to import components with error handling
try:
    from .core.app_factory import create_app
except ImportError:
    create_app = None

__all__ = ["create_app"]

