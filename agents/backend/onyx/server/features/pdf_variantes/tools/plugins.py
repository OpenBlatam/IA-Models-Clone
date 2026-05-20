"""
Tool Plugins
============
Plugin system for extending tool functionality.
"""

from typing import Dict, Any, List, Optional, Callable
from abc import ABC, abstractmethod
from .base import BaseAPITool, ToolResult


class ToolPlugin(ABC):
    """Base class for tool plugins."""
    
    @abstractmethod
    def before_run(self, tool: BaseAPITool, **kwargs) -> Dict[str, Any]:
        """Called before tool execution."""
        pass
    
    @abstractmethod
    def after_run(self, tool: BaseAPITool, result: ToolResult, **kwargs) -> ToolResult:
        """Called after tool execution."""
        pass


class LoggingPlugin(ToolPlugin):
    """Plugin for logging tool execution."""
    
    def before_run(self, tool: BaseAPITool, **kwargs) -> Dict[str, Any]:
        """Log before execution."""
        print(f"🔧 Running {tool.__class__.__name__}...")
        return kwargs
    
    def after_run(self, tool: BaseAPITool, result: ToolResult, **kwargs) -> ToolResult:
        """Log after execution."""
        status = "✅" if result.success else "❌"
        print(f"{status} {tool.__class__.__name__}: {result.message}")
        return result


class MetricsPlugin(ToolPlugin):
    """Plugin for collecting metrics."""
    
    def __init__(self):
        self.metrics: List[Dict[str, Any]] = []
    
    def before_run(self, tool: BaseAPITool, **kwargs) -> Dict[str, Any]:
        """Record start time."""
        import time
        kwargs["_start_time"] = time.time()
        return kwargs
    
    def after_run(self, tool: BaseAPITool, result: ToolResult, **kwargs) -> ToolResult:
        """Record metrics."""
        import time
        start_time = kwargs.get("_start_time", time.time())
        duration = time.time() - start_time
        
        self.metrics.append({
            "tool": tool.__class__.__name__,
            "success": result.success,
            "duration": duration,
            "timestamp": result.timestamp
        })
        
        return result
    
    def get_metrics(self) -> List[Dict[str, Any]]:
        """Get collected metrics."""
        return self.metrics


class CachingPlugin(ToolPlugin):
    """Plugin for caching results."""
    
    def __init__(self):
        self.cache: Dict[str, ToolResult] = {}
    
    def before_run(self, tool: BaseAPITool, **kwargs) -> Dict[str, Any]:
        """Check cache."""
        cache_key = self._get_cache_key(tool, kwargs)
        if cache_key in self.cache:
            print("📦 Using cached result")
            return {**kwargs, "_use_cache": True, "_cache_key": cache_key}
        return kwargs
    
    def after_run(self, tool: BaseAPITool, result: ToolResult, **kwargs) -> ToolResult:
        """Store in cache."""
        if "_use_cache" in kwargs:
            return self.cache[kwargs["_cache_key"]]
        
        cache_key = self._get_cache_key(tool, kwargs)
        self.cache[cache_key] = result
        return result
    
    def _get_cache_key(self, tool: BaseAPITool, kwargs: Dict[str, Any]) -> str:
        """Generate cache key."""
        import hashlib
        import json
        
        key_data = {
            "tool": tool.__class__.__name__,
            "base_url": tool.base_url,
            "kwargs": kwargs
        }
        
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()


class PluginManager:
    """Manager for tool plugins."""
    
    def __init__(self):
        self.plugins: List[ToolPlugin] = []
    
    def register(self, plugin: ToolPlugin):
        """Register a plugin."""
        self.plugins.append(plugin)
    
    def apply_before(self, tool: BaseAPITool, **kwargs) -> Dict[str, Any]:
        """Apply all before hooks."""
        for plugin in self.plugins:
            kwargs = plugin.before_run(tool, **kwargs)
        return kwargs
    
    def apply_after(self, tool: BaseAPITool, result: ToolResult, **kwargs) -> ToolResult:
        """Apply all after hooks."""
        for plugin in self.plugins:
            result = plugin.after_run(tool, result, **kwargs)
        return result


# Global plugin manager
_plugin_manager: Optional[PluginManager] = None


def get_plugin_manager() -> PluginManager:
    """Get global plugin manager."""
    global _plugin_manager
    if _plugin_manager is None:
        _plugin_manager = PluginManager()
    return _plugin_manager



