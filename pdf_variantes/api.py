"""
PDF Variantes API - Legacy Compatibility Module
===============================================

This module provides backward compatibility by re-exporting the main FastAPI application.
The proper entry point is api.main.app.

Note: This file previously contained Flask Blueprint code which has been removed
as part of the refactoring to use FastAPI exclusively.
"""

from api.main import app, create_application

__all__ = ["app", "create_application"]