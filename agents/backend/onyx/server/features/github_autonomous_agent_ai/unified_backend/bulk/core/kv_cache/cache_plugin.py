"""
Cache plugin system.

Provides plugin architecture for extensibility.
"""
from __future__ import annotations

import logging
from typing import Dict, Any, List, Optional, Callable, Protocol
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class CachePlugin(Protocol):
    """Protocol for cache plugins."""
    
    def on_cache_hit(self, position: int, cache: Any) -> None:
        """Called when cache hit occurs."""
        ...
    
    def on_cache_miss(self, position: int, cache: Any) -> None:
        """Called when cache miss occurs."""
        ...
    
    def on_cache_put(self, position: int, cache: Any) -> None:
        """Called when entry is put in cache."""
        ...
    
    def on_cache_evict(self, positions: List[int], cache: Any) -> None:
        """Called when entries are evicted."""
        ...
    
    def get_name(self) -> str:
        """Get plugin name."""
        ...


class PluginManager:
    """
    Plugin manager for cache.
    
    Manages plugins and their lifecycle.
    """
    
    def __init__(self, cache: Any):
        """
        Initialize plugin manager.
        
        Args:
            cache: Cache instance
        """
        self.cache = cache
        self.plugins: List[CachePlugin] = []
        self.plugin_configs: Dict[str, Dict[str, Any]] = {}
    
    def register_plugin(
        self,
        plugin: CachePlugin,
        config: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Register a plugin.
        
        Args:
            plugin: Plugin instance
            config: Optional plugin configuration
        """
        self.plugins.append(plugin)
        self.plugin_configs[plugin.get_name()] = config or {}
        logger.info(f"Registered plugin: {plugin.get_name()}")
    
    def unregister_plugin(self, plugin_name: str) -> bool:
        """
        Unregister a plugin.
        
        Args:
            plugin_name: Name of plugin to unregister
            
        Returns:
            True if unregistered
        """
        for plugin in self.plugins:
            if plugin.get_name() == plugin_name:
                self.plugins.remove(plugin)
                if plugin_name in self.plugin_configs:
                    del self.plugin_configs[plugin_name]
                logger.info(f"Unregistered plugin: {plugin_name}")
                return True
        return False
    
    def notify_hit(self, position: int) -> None:
        """Notify plugins of cache hit."""
        for plugin in self.plugins:
            try:
                plugin.on_cache_hit(position, self.cache)
            except Exception as e:
                logger.warning(f"Plugin {plugin.get_name()} failed on hit: {e}")
    
    def notify_miss(self, position: int) -> None:
        """Notify plugins of cache miss."""
        for plugin in self.plugins:
            try:
                plugin.on_cache_miss(position, self.cache)
            except Exception as e:
                logger.warning(f"Plugin {plugin.get_name()} failed on miss: {e}")
    
    def notify_put(self, position: int) -> None:
        """Notify plugins of cache put."""
        for plugin in self.plugins:
            try:
                plugin.on_cache_put(position, self.cache)
            except Exception as e:
                logger.warning(f"Plugin {plugin.get_name()} failed on put: {e}")
    
    def notify_evict(self, positions: List[int]) -> None:
        """Notify plugins of cache eviction."""
        for plugin in self.plugins:
            try:
                plugin.on_cache_evict(positions, self.cache)
            except Exception as e:
                logger.warning(f"Plugin {plugin.get_name()} failed on evict: {e}")
    
    def list_plugins(self) -> List[str]:
        """
        List registered plugins.
        
        Returns:
            List of plugin names
        """
        return [plugin.get_name() for plugin in self.plugins]


class LoggingPlugin:
    """Example plugin that logs cache operations."""
    
    def __init__(self, log_level: str = "INFO"):
        """
        Initialize logging plugin.
        
        Args:
            log_level: Log level
        """
        self.log_level = log_level
    
    def on_cache_hit(self, position: int, cache: Any) -> None:
        """Log cache hit."""
        logger.log(
            getattr(logging, self.log_level),
            f"Cache hit at position {position}"
        )
    
    def on_cache_miss(self, position: int, cache: Any) -> None:
        """Log cache miss."""
        logger.log(
            getattr(logging, self.log_level),
            f"Cache miss at position {position}"
        )
    
    def on_cache_put(self, position: int, cache: Any) -> None:
        """Log cache put."""
        logger.debug(f"Cache put at position {position}")
    
    def on_cache_evict(self, positions: List[int], cache: Any) -> None:
        """Log cache eviction."""
        logger.info(f"Cache evicted {len(positions)} entries")
    
    def get_name(self) -> str:
        """Get plugin name."""
        return "LoggingPlugin"


class MetricsPlugin:
    """Example plugin that collects metrics."""
    
    def __init__(self):
        """Initialize metrics plugin."""
        self.hits = 0
        self.misses = 0
        self.puts = 0
        self.evictions = 0
    
    def on_cache_hit(self, position: int, cache: Any) -> None:
        """Record hit."""
        self.hits += 1
    
    def on_cache_miss(self, position: int, cache: Any) -> None:
        """Record miss."""
        self.misses += 1
    
    def on_cache_put(self, position: int, cache: Any) -> None:
        """Record put."""
        self.puts += 1
    
    def on_cache_evict(self, positions: List[int], cache: Any) -> None:
        """Record eviction."""
        self.evictions += len(positions)
    
    def get_stats(self) -> Dict[str, int]:
        """
        Get plugin statistics.
        
        Returns:
            Dictionary with stats
        """
        return {
            "hits": self.hits,
            "misses": self.misses,
            "puts": self.puts,
            "evictions": self.evictions
        }
    
    def get_name(self) -> str:
        """Get plugin name."""
        return "MetricsPlugin"

