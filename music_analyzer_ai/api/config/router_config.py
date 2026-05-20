"""
Configuration for routers
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class RouterConfig:
    """Configuration for a router"""
    prefix: str
    tags: list
    include_in_schema: bool = True
    dependencies: Optional[list] = None
    responses: Optional[Dict[int, Dict[str, Any]]] = None


class RouterConfigManager:
    """Manager for router configurations"""
    
    def __init__(self):
        self._configs: Dict[str, RouterConfig] = {}
    
    def register(self, router_name: str, config: RouterConfig):
        """Register a router configuration"""
        self._configs[router_name] = config
    
    def get(self, router_name: str) -> Optional[RouterConfig]:
        """Get router configuration"""
        return self._configs.get(router_name)
    
    def get_all(self) -> Dict[str, RouterConfig]:
        """Get all configurations"""
        return self._configs.copy()


router_config_manager = RouterConfigManager()

