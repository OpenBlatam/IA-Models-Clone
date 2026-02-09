"""
Tool Registry
=============
Registry for managing and discovering tools.
"""

from typing import Dict, Type, List, Optional
from .base import BaseAPITool


class ToolRegistry:
    """Registry for API tools."""
    
    def __init__(self):
        self.tools: Dict[str, Type[BaseAPITool]] = {}
    
    def register(self, name: str, tool_class: Type[BaseAPITool]):
        """Register a tool."""
        self.tools[name] = tool_class
    
    def get(self, name: str) -> Optional[Type[BaseAPITool]]:
        """Get tool by name."""
        return self.tools.get(name)
    
    def list_tools(self) -> List[str]:
        """List all registered tools."""
        return list(self.tools.keys())
    
    def create_tool(self, name: str, **kwargs) -> Optional[BaseAPITool]:
        """Create tool instance."""
        tool_class = self.get(name)
        if tool_class:
            return tool_class(**kwargs)
        return None


# Global registry
_registry = ToolRegistry()


def register_tool(name: str):
    """Decorator to register a tool."""
    def decorator(tool_class: Type[BaseAPITool]):
        _registry.register(name, tool_class)
        return tool_class
    return decorator


def get_registry() -> ToolRegistry:
    """Get global registry."""
    return _registry



