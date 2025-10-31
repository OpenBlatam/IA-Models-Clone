#!/usr/bin/env python3
"""
HeyGen AI Plugin System

This module provides a comprehensive plugin system for dynamically
loading and managing AI models, optimizations, and advanced features.
"""

import importlib
import inspect
import json
import os
import sys
import time
import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Callable, Union, Protocol
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# Plugin Interfaces and Protocols
# =============================================================================

class HeyGenAIPlugin(Protocol):
    """Protocol for HeyGen AI plugins."""
    
    def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize the plugin with configuration."""
        ...
    
    def get_capabilities(self) -> List[str]:
        """Get list of capabilities provided by the plugin."""
        ...
    
    def is_compatible(self, system_version: str) -> bool:
        """Check if plugin is compatible with current system version."""
        ...
    
    def cleanup(self) -> None:
        """Cleanup plugin resources."""
        ...


class ModelPlugin(HeyGenAIPlugin, Protocol):
    """Protocol for model plugins."""
    
    def load_model(self, model_config: Dict[str, Any]) -> Any:
        """Load a model with the given configuration."""
        ...
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        ...
    
    def unload_model(self) -> None:
        """Unload the current model."""
        ...


class OptimizationPlugin(HeyGenAIPlugin, Protocol):
    """Protocol for optimization plugins."""
    
    def apply_optimization(self, model: Any, config: Dict[str, Any]) -> Any:
        """Apply optimization to a model."""
        ...
    
    def get_optimization_info(self) -> Dict[str, Any]:
        """Get information about the optimization."""
        ...
    
    def benchmark_optimization(self, model: Any, test_data: Any) -> Dict[str, Any]:
        """Benchmark the optimization performance."""
        ...


class FeaturePlugin(HeyGenAIPlugin, Protocol):
    """Protocol for feature plugins."""
    
    def enable_feature(self, feature_name: str, config: Dict[str, Any]) -> bool:
        """Enable a specific feature."""
        ...
    
    def get_feature_status(self, feature_name: str) -> Dict[str, Any]:
        """Get status of a specific feature."""
        ...
    
    def disable_feature(self, feature_name: str) -> bool:
        """Disable a specific feature."""
        ...


# =============================================================================
# Plugin Configuration and Metadata
# =============================================================================

@dataclass
class PluginConfig:
    """Configuration for the plugin system."""
    plugin_directories: List[str] = None
    enable_hot_reload: bool = False
    hot_reload_interval: float = 30.0
    enable_plugin_validation: bool = True
    allow_unsafe_plugins: bool = False
    plugin_cache_size: int = 100
    enable_plugin_metrics: bool = True
    plugin_timeout: float = 30.0
    auto_load_plugins: bool = True
    plugin_priority_order: List[str] = None
    
    def __post_init__(self):
        if self.plugin_directories is None:
            self.plugin_directories = [
                "plugins", 
                "extensions", 
                "custom_models",
                "optimizations",
                "features"
            ]
        if self.plugin_priority_order is None:
            self.plugin_priority_order = [
                "core", "optimization", "model", "feature", "experimental"
            ]


@dataclass
class PluginMetadata:
    """Metadata for a plugin."""
    name: str
    version: str
    description: str = ""
    author: str = ""
    license: str = ""
    homepage: str = ""
    repository: str = ""
    tags: List[str] = None
    dependencies: List[str] = None
    requirements: Dict[str, Any] = None
    plugin_type: str = "general"  # model, optimization, feature, core
    priority: str = "normal"  # low, normal, high, critical
    system_version: str = "2.0.0"
    created_at: float = 0.0
    updated_at: float = 0.0
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.dependencies is None:
            self.dependencies = []
        if self.requirements is None:
            self.requirements = {}
        if self.created_at == 0.0:
            self.created_at = time.time()
        self.updated_at = self.created_at
    
    def is_compatible(self, system_version: str) -> bool:
        """Check if plugin is compatible with current system version."""
        # Simple version compatibility check
        # For now, assume all plugins are compatible
        # In a real system, you might want more sophisticated version checking
        return True


@dataclass
class PluginInfo:
    """Complete information about a plugin."""
    metadata: PluginMetadata
    plugin_path: Path
    plugin_class: Optional[Type] = None
    plugin_instance: Optional[Any] = None
    is_loaded: bool = False
    load_time: float = 0.0
    error_count: int = 0
    last_error: Optional[str] = None
    last_used: float = 0.0
    usage_count: int = 0
    performance_metrics: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Base Plugin Classes
# =============================================================================

class BasePlugin(ABC):
    """Base class for all HeyGen AI plugins."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.metadata = self._get_metadata()
        self.initialized = False
        self.logger = logging.getLogger(f"plugin.{self.__class__.__name__}")
    
    @abstractmethod
    def _get_metadata(self) -> PluginMetadata:
        """Get plugin metadata."""
        pass
    
    def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize the plugin with configuration."""
        try:
            self.config.update(config)
            self._initialize_impl()
            self.initialized = True
            self.logger.info(f"Plugin {self.metadata.name} initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize plugin {self.metadata.name}: {e}")
            return False
    
    @abstractmethod
    def _initialize_impl(self) -> None:
        """Implementation of initialization."""
        pass
    
    def get_capabilities(self) -> List[str]:
        """Get list of capabilities provided by the plugin."""
        return []
    
    def is_compatible(self, system_version: str) -> bool:
        """Check if plugin is compatible with current system version."""
        # Simple version compatibility check
        try:
            from packaging import version
            return version.parse(system_version) >= version.parse(self.metadata.system_version)
        except ImportError:
            # Fallback to string comparison
            return system_version >= self.metadata.system_version
    
    def cleanup(self) -> None:
        """Cleanup plugin resources."""
        try:
            self._cleanup_impl()
            self.initialized = False
            self.logger.info(f"Plugin {self.metadata.name} cleaned up successfully")
        except Exception as e:
            self.logger.error(f"Failed to cleanup plugin {self.metadata.name}: {e}")
    
    def _cleanup_impl(self) -> None:
        """Implementation of cleanup."""
        pass
    
    def get_status(self) -> Dict[str, Any]:
        """Get current plugin status."""
        return {
            "name": self.metadata.name,
            "version": self.metadata.version,
            "initialized": self.initialized,
            "capabilities": self.get_capabilities(),
            "config": self.config
        }


class BaseModelPlugin(BasePlugin):
    """Base class for model plugins."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.model = None
        self.model_config = {}
    
    def load_model(self, model_config: Dict[str, Any]) -> Any:
        """Load a model with the given configuration."""
        try:
            self.model_config = model_config
            self.model = self._load_model_impl(model_config)
            self.logger.info(f"Model loaded successfully in {self.metadata.name}")
            return self.model
        except Exception as e:
            self.logger.error(f"Failed to load model in {self.metadata.name}: {e}")
            raise
    
    @abstractmethod
    def _load_model_impl(self, model_config: Dict[str, Any]) -> Any:
        """Implementation of model loading."""
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if self.model is None:
            return {"error": "No model loaded"}
        
        try:
            return self._get_model_info_impl()
        except Exception as e:
            self.logger.error(f"Failed to get model info: {e}")
            return {"error": str(e)}
    
    @abstractmethod
    def _get_model_info_impl(self) -> Dict[str, Any]:
        """Implementation of getting model info."""
        pass
    
    def unload_model(self) -> None:
        """Unload the current model."""
        try:
            if self.model is not None:
                self._unload_model_impl()
                self.model = None
                self.model_config = {}
                self.logger.info(f"Model unloaded from {self.metadata.name}")
        except Exception as e:
            self.logger.error(f"Failed to unload model from {self.metadata.name}: {e}")
    
    def _unload_model_impl(self) -> None:
        """Implementation of model unloading."""
        pass


class BaseOptimizationPlugin(BasePlugin):
    """Base class for optimization plugins."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.optimization_applied = False
    
    def apply_optimization(self, model: Any, config: Dict[str, Any]) -> Any:
        """Apply optimization to a model."""
        try:
            optimized_model = self._apply_optimization_impl(model, config)
            self.optimization_applied = True
            self.logger.info(f"Optimization applied successfully by {self.metadata.name}")
            return optimized_model
        except Exception as e:
            self.logger.error(f"Failed to apply optimization with {self.metadata.name}: {e}")
            raise
    
    @abstractmethod
    def _apply_optimization_impl(self, model: Any, config: Dict[str, Any]) -> Any:
        """Implementation of optimization application."""
        pass
    
    def get_optimization_info(self) -> Dict[str, Any]:
        """Get information about the optimization."""
        try:
            return self._get_optimization_info_impl()
        except Exception as e:
            self.logger.error(f"Failed to get optimization info: {e}")
            return {"error": str(e)}
    
    @abstractmethod
    def _get_optimization_info_impl(self) -> Dict[str, Any]:
        """Implementation of getting optimization info."""
        pass
    
    def benchmark_optimization(self, model: Any, test_data: Any) -> Dict[str, Any]:
        """Benchmark the optimization performance."""
        try:
            return self._benchmark_optimization_impl(model, test_data)
        except Exception as e:
            self.logger.error(f"Failed to benchmark optimization: {e}")
            return {"error": str(e)}
    
    @abstractmethod
    def _benchmark_optimization_impl(self, model: Any, test_data: Any) -> Dict[str, Any]:
        """Implementation of optimization benchmarking."""
        pass


class BaseFeaturePlugin(BasePlugin):
    """Base class for feature plugins."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.enabled_features = set()
    
    def enable_feature(self, feature_name: str, config: Dict[str, Any]) -> bool:
        """Enable a specific feature."""
        try:
            success = self._enable_feature_impl(feature_name, config)
            if success:
                self.enabled_features.add(feature_name)
                self.logger.info(f"Feature {feature_name} enabled in {self.metadata.name}")
            return success
        except Exception as e:
            self.logger.error(f"Failed to enable feature {feature_name} in {self.metadata.name}: {e}")
            return False
    
    @abstractmethod
    def _enable_feature_impl(self, feature_name: str, config: Dict[str, Any]) -> bool:
        """Implementation of feature enabling."""
        pass
    
    def get_feature_status(self, feature_name: str) -> Dict[str, Any]:
        """Get status of a specific feature."""
        try:
            status = self._get_feature_status_impl(feature_name)
            status["enabled"] = feature_name in self.enabled_features
            return status
        except Exception as e:
            self.logger.error(f"Failed to get feature status: {e}")
            return {"error": str(e)}
    
    @abstractmethod
    def _get_feature_status_impl(self, feature_name: str) -> Dict[str, Any]:
        """Implementation of getting feature status."""
        pass
    
    def disable_feature(self, feature_name: str) -> bool:
        """Disable a specific feature."""
        try:
            success = self._disable_feature_impl(feature_name)
            if success:
                self.enabled_features.discard(feature_name)
                self.logger.info(f"Feature {feature_name} disabled in {self.metadata.name}")
            return success
        except Exception as e:
            self.logger.error(f"Failed to disable feature {feature_name} in {self.metadata.name}: {e}")
            return False
    
    @abstractmethod
    def _disable_feature_impl(self, feature_name: str) -> bool:
        """Implementation of feature disabling."""
        pass


# =============================================================================
# Plugin Manager Implementation
# =============================================================================

class PluginManager:
    """Manager for HeyGen AI plugins."""
    
    def __init__(self, config: Optional[PluginConfig] = None):
        self.config = config or PluginConfig()
        self.logger = logging.getLogger("plugin_manager")
        self.plugins: Dict[str, PluginInfo] = {}
        self.plugin_cache: Dict[str, Any] = {}
        self.plugin_metrics: Dict[str, Dict[str, Any]] = {}
        
        # Initialize plugin directories
        self._setup_plugin_directories()
        
        # Load plugins if auto-load is enabled
        if self.config.auto_load_plugins:
            self.load_all_plugins()
    
    def _setup_plugin_directories(self):
        """Setup plugin directories."""
        for plugin_dir in self.config.plugin_directories:
            plugin_path = Path(plugin_dir)
            if not plugin_path.exists():
                plugin_path.mkdir(parents=True, exist_ok=True)
                self.logger.info(f"Created plugin directory: {plugin_path}")
    
    def discover_plugins(self) -> List[Path]:
        """Discover available plugins in plugin directories."""
        plugin_files = []
        
        for plugin_dir in self.config.plugin_directories:
            plugin_path = Path(plugin_dir)
            if plugin_path.exists():
                # Look for Python files and directories
                for item in plugin_path.iterdir():
                    if item.is_file() and item.suffix == ".py":
                        plugin_files.append(item)
                    elif item.is_dir() and (item / "__init__.py").exists():
                        plugin_files.append(item)
        
        self.logger.info(f"Discovered {len(plugin_files)} potential plugins")
        return plugin_files
    
    def load_plugin(self, plugin_path: Path) -> Optional[PluginInfo]:
        """Load a single plugin from path."""
        try:
            # Load plugin metadata
            metadata = self._load_plugin_metadata(plugin_path)
            if not metadata:
                return None
            
            # Check compatibility
            if not self._check_plugin_compatibility(metadata):
                self.logger.warning(f"Plugin {metadata.name} is not compatible")
                return None
            
            # Load plugin class
            plugin_class = self._load_plugin_class(plugin_path, metadata)
            if not plugin_class:
                return None
            
            # Create plugin info
            plugin_info = PluginInfo(
                metadata=metadata,
                plugin_path=plugin_path,
                plugin_class=plugin_class
            )
            
            # Initialize plugin if auto-initialization is enabled
            if self.config.auto_load_plugins:
                self._initialize_plugin(plugin_info)
            
            # Store plugin
            self.plugins[metadata.name] = plugin_info
            self.logger.info(f"Plugin {metadata.name} loaded successfully")
            
            return plugin_info
            
        except Exception as e:
            self.logger.error(f"Failed to load plugin from {plugin_path}: {e}")
            return None
    
    def _load_plugin_metadata(self, plugin_path: Path) -> Optional[PluginMetadata]:
        """Load plugin metadata from file or directory."""
        try:
            # Try to load from metadata file
            metadata_file = plugin_path / "metadata.json"
            if metadata_file.exists():
                with open(metadata_file, "r") as f:
                    metadata_dict = json.load(f)
                    return PluginMetadata(**metadata_dict)
            
            # Try to load from __init__.py
            if plugin_path.is_dir():
                init_file = plugin_path / "__init__.py"
                if init_file.exists():
                    # This is a simplified approach - in practice, you'd want more robust parsing
                    return self._extract_metadata_from_init(init_file)
            
            # Try to load from single Python file
            if plugin_path.is_file() and plugin_path.suffix == ".py":
                return self._extract_metadata_from_file(plugin_path)
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to load plugin metadata from {plugin_path}: {e}")
            return None
    
    def _extract_metadata_from_init(self, init_file: Path) -> Optional[PluginMetadata]:
        """Extract metadata from __init__.py file."""
        try:
            # This is a simplified approach - in practice, you'd want more robust parsing
            return PluginMetadata(
                name=init_file.parent.name,
                version="1.0.0",
                description=f"Plugin from {init_file.parent.name}",
                plugin_type="general"
            )
        except Exception:
            return None
    
    def _extract_metadata_from_file(self, plugin_file: Path) -> Optional[PluginMetadata]:
        """Extract metadata from Python file."""
        try:
            # This is a simplified approach - in practice, you'd want more robust parsing
            return PluginMetadata(
                name=plugin_file.stem,
                version="1.0.0",
                description=f"Plugin from {plugin_file.name}",
                plugin_type="general"
            )
        except Exception:
            return None
    
    def _check_plugin_compatibility(self, metadata: PluginMetadata) -> bool:
        """Check if plugin is compatible with current system."""
        # Check system version compatibility
        if not metadata.is_compatible("2.0.0"):  # Current system version
            return False
        
        # Check dependencies
        for dependency in metadata.dependencies:
            try:
                importlib.import_module(dependency)
            except ImportError:
                self.logger.warning(f"Plugin {metadata.name} dependency {dependency} not available")
                return False
        
        return True
    
    def _load_plugin_class(self, plugin_path: Path, metadata: PluginMetadata) -> Optional[Type]:
        """Load plugin class from file or directory."""
        try:
            if plugin_path.is_dir():
                # Load from directory
                module_name = plugin_path.name
                sys.path.insert(0, str(plugin_path.parent))
                module = importlib.import_module(module_name)
            else:
                # Load from file
                module_name = plugin_path.stem
                sys.path.insert(0, str(plugin_path.parent))
                module = importlib.import_module(module_name)
            
            # Find plugin class
            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and 
                    issubclass(obj, BasePlugin) and 
                    obj != BasePlugin):
                    return obj
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to load plugin class from {plugin_path}: {e}")
            return None
        finally:
            # Clean up sys.path
            if plugin_path.parent in sys.path:
                sys.path.remove(str(plugin_path.parent))
    
    def _initialize_plugin(self, plugin_info: PluginInfo) -> bool:
        """Initialize a plugin."""
        try:
            if plugin_info.plugin_class:
                plugin_instance = plugin_info.plugin_class()
                if plugin_instance.initialize({}):
                    plugin_info.plugin_instance = plugin_instance
                    plugin_info.is_loaded = True
                    plugin_info.load_time = time.time()
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to initialize plugin {plugin_info.metadata.name}: {e}")
            plugin_info.last_error = str(e)
            plugin_info.error_count += 1
            return False
    
    def load_all_plugins(self) -> List[PluginInfo]:
        """Load all available plugins."""
        plugin_files = self.discover_plugins()
        loaded_plugins = []
        
        for plugin_file in plugin_files:
            plugin_info = self.load_plugin(plugin_file)
            if plugin_info:
                loaded_plugins.append(plugin_info)
        
        self.logger.info(f"Loaded {len(loaded_plugins)} plugins")
        return loaded_plugins
    
    def get_plugin(self, plugin_name: str) -> Optional[PluginInfo]:
        """Get a specific plugin by name."""
        return self.plugins.get(plugin_name)
    
    def get_plugins_by_type(self, plugin_type: str) -> List[PluginInfo]:
        """Get plugins by type."""
        return [
            plugin for plugin in self.plugins.values()
            if plugin.metadata.plugin_type == plugin_type
        ]
    
    def get_plugins_by_capability(self, capability: str) -> List[PluginInfo]:
        """Get plugins that provide a specific capability."""
        matching_plugins = []
        
        for plugin_info in self.plugins.values():
            if plugin_info.plugin_instance:
                capabilities = plugin_info.plugin_instance.get_capabilities()
                if capability in capabilities:
                    matching_plugins.append(plugin_info)
        
        return matching_plugins
    
    def reload_plugin(self, plugin_name: str) -> bool:
        """Reload a specific plugin."""
        plugin_info = self.plugins.get(plugin_name)
        if not plugin_info:
            return False
        
        try:
            # Cleanup current instance
            if plugin_info.plugin_instance:
                plugin_info.plugin_instance.cleanup()
            
            # Reload plugin
            plugin_info.plugin_instance = None
            plugin_info.is_loaded = False
            
            # Reinitialize
            if self._initialize_plugin(plugin_info):
                self.logger.info(f"Plugin {plugin_name} reloaded successfully")
                return True
            else:
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to reload plugin {plugin_name}: {e}")
            return False
    
    def unload_plugin(self, plugin_name: str) -> bool:
        """Unload a specific plugin."""
        plugin_info = self.plugins.get(plugin_name)
        if not plugin_info:
            return False
        
        try:
            # Cleanup plugin instance
            if plugin_info.plugin_instance:
                plugin_info.plugin_instance.cleanup()
            
            # Remove from plugins
            del self.plugins[plugin_name]
            
            self.logger.info(f"Plugin {plugin_name} unloaded successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to unload plugin {plugin_name}: {e}")
            return False
    
    def get_plugin_status(self) -> Dict[str, Any]:
        """Get status of all plugins."""
        status = {
            "total_plugins": len(self.plugins),
            "loaded_plugins": len([p for p in self.plugins.values() if p.is_loaded]),
            "plugins": {}
        }
        
        for name, plugin_info in self.plugins.items():
            status["plugins"][name] = {
                "version": plugin_info.metadata.version,
                "type": plugin_info.metadata.plugin_type,
                "loaded": plugin_info.is_loaded,
                "error_count": plugin_info.error_count,
                "last_error": plugin_info.last_error,
                "capabilities": plugin_info.plugin_instance.get_capabilities() if plugin_info.plugin_instance else []
            }
        
        return status
    
    def cleanup(self):
        """Cleanup all plugins."""
        for plugin_info in self.plugins.values():
            try:
                if plugin_info.plugin_instance:
                    plugin_info.plugin_instance.cleanup()
            except Exception as e:
                self.logger.error(f"Failed to cleanup plugin {plugin_info.metadata.name}: {e}")
        
        self.plugins.clear()
        self.logger.info("All plugins cleaned up")


# =============================================================================
# Example Plugin Implementations
# =============================================================================

class ExampleModelPlugin(BaseModelPlugin):
    """Example model plugin implementation."""
    
    def _get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="example_model",
            version="1.0.0",
            description="Example model plugin for demonstration",
            author="HeyGen AI Team",
            plugin_type="model",
            priority="normal"
        )
    
    def _initialize_impl(self) -> None:
        self.logger.info("Example model plugin initialized")
    
    def _load_model_impl(self, model_config: Dict[str, Any]) -> Any:
        # This would load an actual model
        return {"type": "example_model", "config": model_config}
    
    def _get_model_info_impl(self) -> Dict[str, Any]:
        return {
            "type": "example_model",
            "parameters": 1000000,
            "architecture": "transformer"
        }
    
    def _unload_model_impl(self) -> None:
        pass


class ExampleOptimizationPlugin(BaseOptimizationPlugin):
    """Example optimization plugin implementation."""
    
    def _get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="example_optimization",
            version="1.0.0",
            description="Example optimization plugin for demonstration",
            author="HeyGen AI Team",
            plugin_type="optimization",
            priority="normal"
        )
    
    def _initialize_impl(self) -> None:
        self.logger.info("Example optimization plugin initialized")
    
    def _apply_optimization_impl(self, model: Any, config: Dict[str, Any]) -> Any:
        # This would apply actual optimization
        return {"optimized_model": model, "optimization": "example"}
    
    def _get_optimization_info_impl(self) -> Dict[str, Any]:
        return {
            "type": "example_optimization",
            "applied": self.optimization_applied
        }
    
    def _benchmark_optimization_impl(self, model: Any, test_data: Any) -> Dict[str, Any]:
        return {
            "speedup": 1.5,
            "memory_reduction": 0.2,
            "accuracy_maintained": True
        }


# =============================================================================
# Plugin System Factory
# =============================================================================

def create_plugin_manager(config: Optional[PluginConfig] = None) -> PluginManager:
    """Create a plugin manager instance."""
    return PluginManager(config)


def get_plugin_manager() -> PluginManager:
    """Get the global plugin manager instance."""
    if not hasattr(get_plugin_manager, "_instance"):
        get_plugin_manager._instance = create_plugin_manager()
    return get_plugin_manager._instance


# =============================================================================
# Utility Functions
# =============================================================================

def register_plugin(plugin_class: Type[BasePlugin]) -> bool:
    """Register a plugin class with the global plugin manager."""
    try:
        manager = get_plugin_manager()
        # This is a simplified registration - in practice, you'd want more robust handling
        return True
    except Exception as e:
        logger.error(f"Failed to register plugin {plugin_class.__name__}: {e}")
        return False


def get_plugin_by_name(plugin_name: str) -> Optional[PluginInfo]:
    """Get a plugin by name from the global plugin manager."""
    try:
        manager = get_plugin_manager()
        return manager.get_plugin(plugin_name)
    except Exception as e:
        logger.error(f"Failed to get plugin {plugin_name}: {e}")
        return None


def get_plugins_by_type(plugin_type: str) -> List[PluginInfo]:
    """Get plugins by type from the global plugin manager."""
    try:
        manager = get_plugin_manager()
        return manager.get_plugins_by_type(plugin_type)
    except Exception as e:
        logger.error(f"Failed to get plugins by type {plugin_type}: {e}")
        return []


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create plugin manager
    plugin_config = PluginConfig(
        enable_hot_reload=True,
        auto_load_plugins=True
    )
    
    manager = create_plugin_manager(plugin_config)
    
    try:
        # Load all plugins
        plugins = manager.load_all_plugins()
        print(f"Loaded {len(plugins)} plugins")
        
        # Get plugin status
        status = manager.get_plugin_status()
        print(f"Plugin status: {status}")
        
        # Example: Get model plugins
        model_plugins = manager.get_plugins_by_type("model")
        print(f"Found {len(model_plugins)} model plugins")
        
        # Example: Get optimization plugins
        opt_plugins = manager.get_plugins_by_type("optimization")
        print(f"Found {len(opt_plugins)} optimization plugins")
        
    finally:
        # Cleanup
        manager.cleanup()
