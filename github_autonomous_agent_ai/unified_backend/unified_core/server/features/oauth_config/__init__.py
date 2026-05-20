"""OAuth configuration feature module."""

from unified_core.server.features.oauth_config.api import admin_router
from unified_core.server.features.oauth_config.api import router

__all__ = ["admin_router", "router"]
