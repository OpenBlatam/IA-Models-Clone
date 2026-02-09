"""
Example Plugin - Template for creating new plugins
"""

from core.plugin_system import BasePlugin, PluginMetadata, PluginType
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class ExamplePlugin(BasePlugin):
    """Example plugin demonstrating plugin structure"""
    
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="example",
            version="1.0.0",
            plugin_type=PluginType.SERVICE,
            description="Example plugin for demonstration",
            author="Dermatology AI Team",
            dependencies=[],
            config_schema={
                "required": [],
                "properties": {
                    "enabled": {"type": "boolean", "default": True}
                }
            }
        )
    
    async def initialize(self, config: Optional[Dict[str, Any]] = None):
        """Initialize plugin"""
        config = config or {}
        self.enabled = config.get("enabled", True)
        logger.info(f"Example plugin initialized (enabled: {self.enabled})")
    
    async def shutdown(self):
        """Shutdown plugin"""
        logger.info("Example plugin shutdown")
    
    async def do_something(self):
        """Plugin functionality"""
        if self.enabled:
            logger.info("Example plugin doing something")
            return {"status": "success"}
        return {"status": "disabled"}















