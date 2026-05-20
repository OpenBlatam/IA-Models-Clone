"""
Tool Factory
============
Factory pattern for creating tools with dependency injection.
"""

from typing import Dict, Any, Optional, Type, Callable
from .base import BaseAPITool
from .config import ToolConfig, get_config
from .registry import get_registry


class ToolFactory:
    """Factory for creating tool instances."""
    
    def __init__(self, config: Optional[ToolConfig] = None):
        self.config = config or get_config()
        self.registry = get_registry()
        self._custom_creators: Dict[str, Callable] = {}
    
    def register_creator(self, name: str, creator: Callable):
        """Register custom creator for a tool."""
        self._custom_creators[name] = creator
    
    def create(
        self,
        name: str,
        base_url: Optional[str] = None,
        **kwargs
    ) -> Optional[BaseAPITool]:
        """Create tool instance."""
        # Check for custom creator
        if name in self._custom_creators:
            return self._custom_creators[name](**kwargs)
        
        # Get tool class from registry
        tool_class = self.registry.get(name)
        if not tool_class:
            return None
        
        # Merge config with kwargs
        tool_kwargs = {
            "base_url": base_url or self.config.base_url,
            "timeout": self.config.timeout
        }
        tool_kwargs.update(kwargs)
        
        # Create instance
        tool = tool_class(**tool_kwargs)
        
        # Set auth token if configured
        if self.config.auth_token:
            tool.set_auth_token(self.config.auth_token)
        
        return tool
    
    def create_with_config(
        self,
        name: str,
        config: ToolConfig,
        **kwargs
    ) -> Optional[BaseAPITool]:
        """Create tool with specific config."""
        original_config = self.config
        self.config = config
        result = self.create(name, **kwargs)
        self.config = original_config
        return result


# Global factory
_factory: Optional[ToolFactory] = None


def get_factory() -> ToolFactory:
    """Get global factory instance."""
    global _factory
    if _factory is None:
        _factory = ToolFactory()
    return _factory



