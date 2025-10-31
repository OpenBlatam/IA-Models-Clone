from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Type, TypeVar
from dataclasses import dataclass, field
from datetime import datetime
from ..core.interfaces import (
from ..core.exceptions import PluginError, ValidationError
from ..core.types import PluginInfo, ComponentConfig
from typing import Any, List, Dict, Optional
"""
Base Plugin Classes and Registry

This module provides the base classes and registry system for all plugins
in the modular AI video workflow system.
"""


    ExtractorInterface,
    SuggestionInterface,
    GeneratorInterface,
    StateRepositoryInterface,
    MetricsInterface,
    PluginInterface
)

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class PluginMetadata:
    """Metadata for plugin registration."""
    name: str
    version: str
    description: str
    author: str
    category: str
    dependencies: List[str] = field(default_factory=list)
    config_schema: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


class BasePlugin(ABC):
    """
    Base class for all plugins in the AI video workflow system.
    
    This class provides:
    - Standard plugin interface
    - Lifecycle management
    - Configuration handling
    - Component registration
    - Error handling
    """
    
    # Plugin metadata - must be overridden by subclasses
    name: str = "base_plugin"
    version: str = "1.0.0"
    description: str = "Base plugin class"
    author: str = "Unknown"
    category: str = "general"
    dependencies: List[str] = field(default_factory=list)
    config_schema: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        
    """__init__ function."""
self.config = config or {}
        self.is_initialized = False
        self.is_enabled = True
        self.logger = logging.getLogger(f"plugin.{self.name}")
        self.metadata = self._create_metadata()
        
        # Component registries
        self._extractors: Dict[str, ExtractorInterface] = {}
        self._suggestion_engines: Dict[str, SuggestionInterface] = {}
        self._generators: Dict[str, GeneratorInterface] = {}
        self._repositories: Dict[str, StateRepositoryInterface] = {}
        self._metrics_collectors: Dict[str, MetricsInterface] = {}
        
        self.logger.debug(f"Plugin {self.name} initialized")
    
    def _create_metadata(self) -> PluginMetadata:
        """Create plugin metadata."""
        return PluginMetadata(
            name=self.name,
            version=self.version,
            description=self.description,
            author=self.author,
            category=self.category,
            dependencies=self.dependencies,
            config_schema=self.config_schema,
            tags=self.tags
        )
    
    async def initialize(self, context: Any) -> None:
        """
        Initialize the plugin with the given context.
        
        Args:
            context: Plugin context containing dependencies and resources
        """
        try:
            if self.is_initialized:
                self.logger.warning(f"Plugin {self.name} is already initialized")
                return
            
            # Validate configuration
            if not self.validate_config(self.config):
                raise ValidationError(f"Invalid configuration for plugin {self.name}")
            
            # Initialize components
            await self._initialize_components()
            
            # Register components
            await self._register_components()
            
            self.is_initialized = True
            self.logger.info(f"Plugin {self.name} initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize plugin {self.name}: {e}")
            raise PluginError(f"Initialization failed for plugin {self.name}", plugin_name=self.name) from e
    
    async def cleanup(self) -> None:
        """Cleanup plugin resources."""
        try:
            if not self.is_initialized:
                return
            
            # Cleanup components
            await self._cleanup_components()
            
            # Clear registries
            self._extractors.clear()
            self._suggestion_engines.clear()
            self._generators.clear()
            self._repositories.clear()
            self._metrics_collectors.clear()
            
            self.is_initialized = False
            self.logger.info(f"Plugin {self.name} cleaned up successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup plugin {self.name}: {e}")
            raise PluginError(f"Cleanup failed for plugin {self.name}", plugin_name=self.name) from e
    
    @abstractmethod
    async def _initialize_components(self) -> None:
        """Initialize plugin-specific components. Must be implemented by subclasses."""
        pass
    
    async def _register_components(self) -> None:
        """Register components provided by this plugin."""
        # This method can be overridden by subclasses to register components
        pass
    
    async def _cleanup_components(self) -> None:
        """Cleanup plugin-specific components."""
        # This method can be overridden by subclasses to cleanup components
        pass
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate plugin configuration.
        
        Args:
            config: Configuration to validate
            
        Returns:
            True if configuration is valid
        """
        try:
            # Basic validation - subclasses can override for more specific validation
            if not isinstance(config, dict):
                return False
            
            # Check required fields if config_schema is defined
            if self.config_schema:
                required_fields = self.config_schema.get('required', [])
                for field in required_fields:
                    if field not in config:
                        self.logger.error(f"Required field '{field}' missing in configuration")
                        return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Configuration validation failed: {e}")
            return False
    
    def get_config_schema(self) -> Dict[str, Any]:
        """Get configuration schema for this plugin."""
        return self.config_schema.copy()
    
    def get_metadata(self) -> PluginMetadata:
        """Get plugin metadata."""
        return self.metadata
    
    # Component registration methods
    def register_extractor(self, name: str, extractor: ExtractorInterface) -> None:
        """Register an extractor component."""
        self._extractors[name] = extractor
        self.logger.debug(f"Registered extractor: {name}")
    
    def register_suggestion_engine(self, name: str, engine: SuggestionInterface) -> None:
        """Register a suggestion engine component."""
        self._suggestion_engines[name] = engine
        self.logger.debug(f"Registered suggestion engine: {name}")
    
    def register_generator(self, name: str, generator: GeneratorInterface) -> None:
        """Register a video generator component."""
        self._generators[name] = generator
        self.logger.debug(f"Registered generator: {name}")
    
    def register_repository(self, name: str, repository: StateRepositoryInterface) -> None:
        """Register a state repository component."""
        self._repositories[name] = repository
        self.logger.debug(f"Registered repository: {name}")
    
    def register_metrics_collector(self, name: str, collector: MetricsInterface) -> None:
        """Register a metrics collector component."""
        self._metrics_collectors[name] = collector
        self.logger.debug(f"Registered metrics collector: {name}")
    
    # Component retrieval methods
    async def get_extractors(self) -> Dict[str, ExtractorInterface]:
        """Get all extractors registered by this plugin."""
        return self._extractors.copy()
    
    async def get_suggestion_engines(self) -> Dict[str, SuggestionInterface]:
        """Get all suggestion engines registered by this plugin."""
        return self._suggestion_engines.copy()
    
    async def get_generators(self) -> Dict[str, GeneratorInterface]:
        """Get all generators registered by this plugin."""
        return self._generators.copy()
    
    async def get_repositories(self) -> Dict[str, StateRepositoryInterface]:
        """Get all repositories registered by this plugin."""
        return self._repositories.copy()
    
    async def get_metrics_collectors(self) -> Dict[str, MetricsInterface]:
        """Get all metrics collectors registered by this plugin."""
        return self._metrics_collectors.copy()
    
    def is_available(self) -> bool:
        """Check if the plugin is available for use."""
        return self.is_initialized and self.is_enabled
    
    def enable(self) -> None:
        """Enable the plugin."""
        self.is_enabled = True
        self.logger.info(f"Plugin {self.name} enabled")
    
    def disable(self) -> None:
        """Disable the plugin."""
        self.is_enabled = False
        self.logger.info(f"Plugin {self.name} disabled")


class PluginRegistry:
    """
    Registry for managing plugin registration and discovery.
    
    This class provides:
    - Plugin registration and lookup
    - Category-based organization
    - Dependency tracking
    - Version management
    """
    
    def __init__(self) -> Any:
        self._plugins: Dict[str, Type[BasePlugin]] = {}
        self._metadata: Dict[str, PluginMetadata] = {}
        self._categories: Dict[str, List[str]] = {}
        self._tags: Dict[str, List[str]] = {}
        
        logger.info("PluginRegistry initialized")
    
    def register(self, plugin_class: Type[BasePlugin]) -> None:
        """
        Register a plugin class.
        
        Args:
            plugin_class: Plugin class to register
        """
        if not issubclass(plugin_class, BasePlugin):
            raise ValueError(f"Plugin class must inherit from BasePlugin")
        
        # Create instance to get metadata
        instance = plugin_class()
        metadata = instance.get_metadata()
        
        # Register plugin
        self._plugins[metadata.name] = plugin_class
        self._metadata[metadata.name] = metadata
        
        # Organize by category
        if metadata.category not in self._categories:
            self._categories[metadata.category] = []
        self._categories[metadata.category].append(metadata.name)
        
        # Organize by tags
        for tag in metadata.tags:
            if tag not in self._tags:
                self._tags[tag] = []
            self._tags[tag].append(metadata.name)
        
        logger.info(f"Registered plugin: {metadata.name} v{metadata.version}")
    
    def unregister(self, plugin_name: str) -> None:
        """
        Unregister a plugin.
        
        Args:
            plugin_name: Name of the plugin to unregister
        """
        if plugin_name not in self._plugins:
            logger.warning(f"Plugin {plugin_name} not found in registry")
            return
        
        metadata = self._metadata[plugin_name]
        
        # Remove from registries
        del self._plugins[plugin_name]
        del self._metadata[plugin_name]
        
        # Remove from categories
        if metadata.category in self._categories:
            self._categories[metadata.category].remove(plugin_name)
            if not self._categories[metadata.category]:
                del self._categories[metadata.category]
        
        # Remove from tags
        for tag in metadata.tags:
            if tag in self._tags:
                self._tags[tag].remove(plugin_name)
                if not self._tags[tag]:
                    del self._tags[tag]
        
        logger.info(f"Unregistered plugin: {plugin_name}")
    
    def get_plugin_class(self, plugin_name: str) -> Optional[Type[BasePlugin]]:
        """Get plugin class by name."""
        return self._plugins.get(plugin_name)
    
    def get_metadata(self, plugin_name: str) -> Optional[PluginMetadata]:
        """Get plugin metadata by name."""
        return self._metadata.get(plugin_name)
    
    def list_plugins(self, category: Optional[str] = None, tag: Optional[str] = None) -> List[str]:
        """
        List registered plugins, optionally filtered by category or tag.
        
        Args:
            category: Filter by category
            tag: Filter by tag
            
        Returns:
            List of plugin names
        """
        if category:
            return self._categories.get(category, [])
        elif tag:
            return self._tags.get(tag, [])
        else:
            return list(self._plugins.keys())
    
    def get_categories(self) -> List[str]:
        """Get list of all categories."""
        return list(self._categories.keys())
    
    def get_tags(self) -> List[str]:
        """Get list of all tags."""
        return list(self._tags.keys())
    
    def search_plugins(self, query: str) -> List[str]:
        """
        Search plugins by name, description, or tags.
        
        Args:
            query: Search query
            
        Returns:
            List of matching plugin names
        """
        query = query.lower()
        results = []
        
        for name, metadata in self._metadata.items():
            if (query in name.lower() or 
                query in metadata.description.lower() or
                any(query in tag.lower() for tag in metadata.tags)):
                results.append(name)
        
        return results
    
    def get_dependencies(self, plugin_name: str) -> List[str]:
        """Get dependencies for a plugin."""
        metadata = self._metadata.get(plugin_name)
        if metadata:
            return metadata.dependencies.copy()
        return []
    
    def check_dependencies(self, plugin_name: str) -> Dict[str, bool]:
        """
        Check if all dependencies for a plugin are available.
        
        Args:
            plugin_name: Name of the plugin
            
        Returns:
            Dictionary mapping dependency names to availability status
        """
        dependencies = self.get_dependencies(plugin_name)
        status = {}
        
        for dep in dependencies:
            status[dep] = dep in self._plugins
        
        return status


# Decorator for easy plugin registration
def register_plugin(registry: Optional[PluginRegistry] = None):
    """
    Decorator to automatically register a plugin class.
    
    Args:
        registry: Plugin registry to use (uses global registry if None)
    """
    def decorator(plugin_class: Type[BasePlugin]) -> Type[BasePlugin]:
        if registry:
            registry.register(plugin_class)
        else:
            # Use global registry if available
            global _global_registry
            if '_global_registry' in globals():
                _global_registry.register(plugin_class)
        return plugin_class
    return decorator


# Global registry instance
_global_registry = PluginRegistry()


def get_global_registry() -> PluginRegistry:
    """Get the global plugin registry."""
    return _global_registry 