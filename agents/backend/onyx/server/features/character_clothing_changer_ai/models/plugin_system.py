"""
Plugin System for Flux2 Clothing Changer
=========================================

Extensible plugin system for custom processing hooks.
"""

import logging
from typing import Dict, Any, Optional, Callable, List
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import inspect

logger = logging.getLogger(__name__)


class HookType(Enum):
    """Plugin hook types."""
    PRE_VALIDATION = "pre_validation"
    POST_VALIDATION = "post_validation"
    PRE_ENHANCEMENT = "pre_enhancement"
    POST_ENHANCEMENT = "post_enhancement"
    PRE_PROCESSING = "pre_processing"
    POST_PROCESSING = "post_processing"
    PRE_INPAINTING = "pre_inpainting"
    POST_INPAINTING = "post_inpainting"
    ERROR_HANDLING = "error_handling"


@dataclass
class ProcessingContext:
    """Context passed to plugins."""
    image: Any
    clothing_description: str
    mask: Optional[Any] = None
    prompt: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class Plugin(ABC):
    """Base class for plugins."""
    
    def __init__(self, name: str, priority: int = 100):
        """
        Initialize plugin.
        
        Args:
            name: Plugin name
            priority: Execution priority (lower = earlier)
        """
        self.name = name
        self.priority = priority
        self.enabled = True
    
    @abstractmethod
    def execute(self, context: ProcessingContext, **kwargs) -> Optional[ProcessingContext]:
        """
        Execute plugin logic.
        
        Args:
            context: Processing context
            **kwargs: Additional arguments
            
        Returns:
            Modified context or None to stop processing
        """
        pass
    
    def on_error(self, error: Exception, context: ProcessingContext) -> bool:
        """
        Handle errors.
        
        Args:
            error: Exception that occurred
            context: Processing context
            
        Returns:
            True to continue, False to stop
        """
        return True


class PluginManager:
    """Manages plugins and hooks."""
    
    def __init__(self):
        """Initialize plugin manager."""
        self.plugins: Dict[HookType, List[Plugin]] = {
            hook_type: [] for hook_type in HookType
        }
    
    def register_plugin(
        self,
        plugin: Plugin,
        hook_types: List[HookType],
    ) -> None:
        """
        Register a plugin for specific hooks.
        
        Args:
            plugin: Plugin instance
            hook_types: List of hook types to register for
        """
        for hook_type in hook_types:
            self.plugins[hook_type].append(plugin)
            # Sort by priority
            self.plugins[hook_type].sort(key=lambda p: p.priority)
        
        logger.info(f"Plugin '{plugin.name}' registered for {len(hook_types)} hooks")
    
    def execute_hooks(
        self,
        hook_type: HookType,
        context: ProcessingContext,
        **kwargs
    ) -> Optional[ProcessingContext]:
        """
        Execute all plugins for a hook type.
        
        Args:
            hook_type: Hook type to execute
            context: Processing context
            **kwargs: Additional arguments
            
        Returns:
            Modified context or None if processing should stop
        """
        plugins = self.plugins[hook_type]
        
        for plugin in plugins:
            if not plugin.enabled:
                continue
            
            try:
                result = plugin.execute(context, **kwargs)
                if result is None:
                    logger.info(f"Plugin '{plugin.name}' stopped processing")
                    return None
                context = result
            except Exception as e:
                logger.error(f"Plugin '{plugin.name}' error: {e}")
                if not plugin.on_error(e, context):
                    return None
        
        return context
    
    def unregister_plugin(self, plugin_name: str) -> None:
        """
        Unregister a plugin.
        
        Args:
            plugin_name: Name of plugin to unregister
        """
        for hook_type in HookType:
            self.plugins[hook_type] = [
                p for p in self.plugins[hook_type]
                if p.name != plugin_name
            ]
        logger.info(f"Plugin '{plugin_name}' unregistered")
    
    def list_plugins(self) -> Dict[HookType, List[str]]:
        """List all registered plugins by hook type."""
        return {
            hook_type: [p.name for p in plugins]
            for hook_type, plugins in self.plugins.items()
            if plugins
        }


# Example plugins
class LoggingPlugin(Plugin):
    """Plugin for logging processing steps."""
    
    def execute(self, context: ProcessingContext, **kwargs) -> ProcessingContext:
        """Log processing step."""
        logger.debug(f"Processing: {context.clothing_description}")
        return context


class MetadataPlugin(Plugin):
    """Plugin for adding metadata."""
    
    def execute(self, context: ProcessingContext, **kwargs) -> ProcessingContext:
        """Add processing metadata."""
        context.metadata["processed_by"] = "Flux2ClothingChanger"
        context.metadata["timestamp"] = time.time()
        return context


import time


