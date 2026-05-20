"""
API endpoints for dermatology AI

This module exports the modular router which is the primary API interface.
The old monolithic dermatology_api.py is deprecated and kept only for backward compatibility.
"""

from .dermatology_api_modular import modular_router as router

__all__ = ["router"]






