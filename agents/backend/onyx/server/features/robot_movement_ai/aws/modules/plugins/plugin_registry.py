"""
Plugin Registry
===============

Plugin registry for discovery.
"""

import logging
from typing import Dict, List, Optional
from aws.modules.plugins.plugin_manager import PluginManager, PluginMetadata

logger = logging.getLogger(__name__)


class PluginRegistry:
    """Plugin registry for discovery and management."""
    
    def __init__(self, plugin_manager: PluginManager):
        self.plugin_manager = plugin_manager
        self._categories: Dict[str, List[str]] = {}
        self._tags: Dict[str, List[str]] = {}
    
    def categorize(self, plugin_name: str, category: str):
        """Categorize plugin."""
        if category not in self._categories:
            self._categories[category] = []
        if plugin_name not in self._categories[category]:
            self._categories[category].append(plugin_name)
    
    def tag(self, plugin_name: str, tags: List[str]):
        """Tag plugin."""
        self._tags[plugin_name] = tags
    
    def find_by_category(self, category: str) -> List[PluginMetadata]:
        """Find plugins by category."""
        plugin_names = self._categories.get(category, [])
        return [
            self.plugin_manager._metadata[name]
            for name in plugin_names
            if name in self.plugin_manager._metadata
        ]
    
    def find_by_tag(self, tag: str) -> List[PluginMetadata]:
        """Find plugins by tag."""
        matching = []
        for plugin_name, tags in self._tags.items():
            if tag in tags:
                metadata = self.plugin_manager._metadata.get(plugin_name)
                if metadata:
                    matching.append(metadata)
        return matching
    
    def get_all_categories(self) -> List[str]:
        """Get all categories."""
        return list(self._categories.keys())
    
    def get_plugin_info(self, plugin_name: str) -> Optional[Dict]:
        """Get plugin information."""
        metadata = self.plugin_manager._metadata.get(plugin_name)
        if not metadata:
            return None
        
        return {
            "metadata": metadata,
            "category": self._get_category(plugin_name),
            "tags": self._tags.get(plugin_name, [])
        }
    
    def _get_category(self, plugin_name: str) -> Optional[str]:
        """Get category for plugin."""
        for category, plugins in self._categories.items():
            if plugin_name in plugins:
                return category
        return None















