"""
API module for AI History Comparison System

This module provides a unified API structure that consolidates all
API endpoints into a single, well-organized system.
"""

from .router import create_api_router
from .endpoints import (
    analysis_endpoints,
    comparison_endpoints,
    trend_endpoints,
    content_endpoints,
    system_endpoints
)

__all__ = [
    'create_api_router',
    'analysis_endpoints',
    'comparison_endpoints', 
    'trend_endpoints',
    'content_endpoints',
    'system_endpoints'
]





















