"""
API Endpoints for AI History Comparison System

This module contains all API endpoint definitions organized by functionality.
"""

from .analysis import router as analysis_endpoints
from .comparison import router as comparison_endpoints
from .trends import router as trend_endpoints
from .content import router as content_endpoints
from .system import router as system_endpoints

__all__ = [
    'analysis_endpoints',
    'comparison_endpoints',
    'trend_endpoints',
    'content_endpoints',
    'system_endpoints'
]





















