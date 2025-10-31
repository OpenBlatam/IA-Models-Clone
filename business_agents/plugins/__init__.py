"""Plugin system for auto-discovery of routers."""
import os
from importlib import import_module
from pathlib import Path
from typing import List, Tuple, Optional
from fastapi import APIRouter

PLUGINS_DIR = Path(__file__).parent


def discover_plugin_routers() -> List[Tuple[APIRouter, str]]:
    """
    Auto-discover routers from plugin modules.
    
    Looks for files matching plugin_*.py or *_plugin.py in the plugins directory.
    Each plugin should export a 'router' variable of type APIRouter.
    
    Returns:
        List of (router, prefix) tuples
    """
    routers = []
    
    if not PLUGINS_DIR.exists():
        return routers
    
    for file_path in PLUGINS_DIR.glob("*.py"):
        if file_path.name == "__init__.py":
            continue
        
        # Check if it looks like a plugin file
        name = file_path.stem
        if not (name.startswith("plugin_") or name.endswith("_plugin")):
            continue
        
        try:
            # Import the module
            module_name = f".plugins.{name}"
            mod = import_module(module_name, package="agents.backend.onyx.server.features.business_agents")
            
            # Look for router attribute
            if hasattr(mod, "router"):
                router = getattr(mod, "router")
                if isinstance(router, APIRouter):
                    # Get prefix if defined, default to /api/v1
                    prefix = getattr(mod, "ROUTER_PREFIX", "/api/v1")
                    routers.append((router, prefix))
        except Exception as e:
            # Silently skip plugins that fail to load
            import logging
            logging.getLogger(__name__).warning(f"Failed to load plugin {name}: {e}")
    
    return routers


