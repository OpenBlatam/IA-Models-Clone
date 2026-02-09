"""
Plugin Loader
=============

Plugin loader for dynamic plugin loading.
"""

import logging
import os
import importlib.util
from typing import List, Dict, Any
from pathlib import Path
from aws.modules.plugins.plugin_manager import PluginManager, IPlugin

logger = logging.getLogger(__name__)


class PluginLoader:
    """Plugin loader for dynamic loading."""
    
    def __init__(self, plugin_manager: PluginManager):
        self.plugin_manager = plugin_manager
    
    def load_from_directory(self, directory: str, pattern: str = "*.py"):
        """Load plugins from directory."""
        plugin_dir = Path(directory)
        if not plugin_dir.exists():
            logger.warning(f"Plugin directory does not exist: {directory}")
            return
        
        plugin_files = list(plugin_dir.glob(pattern))
        for plugin_file in plugin_files:
            if plugin_file.name.startswith("__"):
                continue
            
            try:
                self._load_plugin_file(plugin_file)
            except Exception as e:
                logger.error(f"Failed to load plugin from {plugin_file}: {e}")
    
    def _load_plugin_file(self, plugin_file: Path):
        """Load plugin from file."""
        spec = importlib.util.spec_from_file_location(
            plugin_file.stem,
            plugin_file
        )
        
        if spec is None or spec.loader is None:
            raise ValueError(f"Could not load spec from {plugin_file}")
        
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Find plugin classes
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if (isinstance(attr, type) and
                issubclass(attr, IPlugin) and
                attr != IPlugin):
                plugin = attr()
                self.plugin_manager.register(plugin)
                logger.info(f"Loaded plugin: {plugin.get_metadata().name}")















