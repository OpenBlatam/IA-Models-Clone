"""
Ads feature module initialization.

Refactored to expose the unified API router under `ads_router`.
Legacy routers are no longer exported from this package.
"""

from .api import main_router as ads_router

__all__ = [
    "ads_router",
] 