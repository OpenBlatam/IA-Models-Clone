"""
Ads Feature Module
===================

AI-powered advertisement generation and management system.

Refactored to expose the unified API router under `ads_router`.
Legacy routers are no longer exported from this package.
"""

__version__ = "1.0.0"
__author__ = "Blatam Academy"
__license__ = "Proprietary"
__description__ = "AI-powered advertisement generation and management system"

from .api import main_router as ads_router

__all__ = [
    "ads_router",
] 