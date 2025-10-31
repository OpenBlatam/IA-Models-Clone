"""
Example plugin - demonstrates how to create a custom router plugin.

To use:
1. Place this file (or rename it) in the plugins/ directory
2. Name it with prefix 'plugin_' or suffix '_plugin' (e.g., plugin_myfeature.py or myfeature_plugin.py)
3. Export a 'router' variable of type APIRouter
4. Optionally set ROUTER_PREFIX to customize the URL prefix
"""

from fastapi import APIRouter

# Optional: customize the URL prefix (defaults to /api/v1)
ROUTER_PREFIX = "/api/v1"

router = APIRouter(prefix="/example", tags=["Example Plugin"])


@router.get("/hello")
async def hello():
    """Example endpoint from a plugin."""
    return {"message": "Hello from plugin!", "plugin": "example_plugin"}


@router.get("/info")
async def plugin_info():
    """Get plugin information."""
    return {
        "name": "example_plugin",
        "version": "1.0.0",
        "description": "Example plugin demonstrating auto-discovery"
    }


