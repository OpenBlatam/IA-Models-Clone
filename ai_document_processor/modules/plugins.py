"""
Plugin System - Ultra-Modular Plugin Architecture
================================================

Ultra-modular plugin system for dynamic component loading and extensibility.
"""

import asyncio
import importlib
import inspect
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type, TypeVar, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import json
import yaml

logger = logging.getLogger(__name__)

T = TypeVar('T')


class PluginStatus(str, Enum):
    """Plugin status enumeration."""
    LOADED = "loaded"
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    DEPRECATED = "deprecated"


class PluginType(str, Enum):
    """Plugin type enumeration."""
    DOCUMENT_PROCESSOR = "document_processor"
    AI_SERVICE = "ai_service"
    TRANSFORM_SERVICE = "transform_service"
    VALIDATION_SERVICE = "validation_service"
    CACHE_SERVICE = "cache_service"
    FILE_SERVICE = "file_service"
    NOTIFICATION_SERVICE = "notification_service"
    METRICS_SERVICE = "metrics_service"
    MIDDLEWARE = "middleware"
    CUSTOM = "custom"


@dataclass
class PluginMetadata:
    """Plugin metadata."""
    name: str
    version: str
    description: str
    author: str
    plugin_type: PluginType
    dependencies: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    configuration_schema: Optional[Dict[str, Any]] = None
    api_version: str = "1.0"
    min_system_version: str = "4.0.0"
    max_system_version: Optional[str] = None
    license: str = "MIT"
    homepage: Optional[str] = None
    repository: Optional[str] = None
    documentation: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class PluginInfo:
    """Plugin information."""
    id: str
    name: str
    plugin_type: PluginType
    metadata: PluginMetadata
    module_path: str
    class_name: str
    instance: Optional[Any] = None
    status: PluginStatus = PluginStatus.LOADED
    configuration: Dict[str, Any] = field(default_factory=dict)
    error_count: int = 0
    last_error: Optional[str] = None
    usage_count: int = 0
    last_used: Optional[datetime] = None


class PluginInterface(ABC):
    """Base plugin interface."""
    
    @abstractmethod
    async def initialize(self, configuration: Dict[str, Any]) -> bool:
        """Initialize the plugin."""
        pass
    
    @abstractmethod
    async def start(self) -> bool:
        """Start the plugin."""
        pass
    
    @abstractmethod
    async def stop(self) -> bool:
        """Stop the plugin."""
        pass
    
    @abstractmethod
    async def health_check(self) -> float:
        """Perform health check."""
        pass
    
    @abstractmethod
    def get_capabilities(self) -> List[str]:
        """Get plugin capabilities."""
        pass
    
    @abstractmethod
    def get_configuration_schema(self) -> Dict[str, Any]:
        """Get configuration schema."""
        pass


class DocumentProcessorPlugin(PluginInterface):
    """Base class for document processor plugins."""
    
    @abstractmethod
    async def process_document(self, document: Any, configuration: Dict[str, Any]) -> Any:
        """Process a document."""
        pass
    
    @abstractmethod
    def get_supported_formats(self) -> List[str]:
        """Get supported document formats."""
        pass


class AIServicePlugin(PluginInterface):
    """Base class for AI service plugins."""
    
    @abstractmethod
    async def classify_document(self, content: str, configuration: Dict[str, Any]) -> str:
        """Classify document content."""
        pass
    
    @abstractmethod
    async def transform_content(self, content: str, configuration: Dict[str, Any]) -> str:
        """Transform document content."""
        pass
    
    @abstractmethod
    def get_available_models(self) -> List[str]:
        """Get available AI models."""
        pass


class TransformServicePlugin(PluginInterface):
    """Base class for transform service plugins."""
    
    @abstractmethod
    async def transform_document(self, document: Any, target_format: str, configuration: Dict[str, Any]) -> Any:
        """Transform document to target format."""
        pass
    
    @abstractmethod
    def get_supported_formats(self) -> List[str]:
        """Get supported output formats."""
        pass


class ValidationServicePlugin(PluginInterface):
    """Base class for validation service plugins."""
    
    @abstractmethod
    async def validate_document(self, document: Any, configuration: Dict[str, Any]) -> bool:
        """Validate document."""
        pass
    
    @abstractmethod
    def get_validation_rules(self) -> List[str]:
        """Get validation rules."""
        pass


class CacheServicePlugin(PluginInterface):
    """Base class for cache service plugins."""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        pass
    
    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache."""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete value from cache."""
        pass
    
    @abstractmethod
    async def clear(self) -> bool:
        """Clear all cache."""
        pass


class FileServicePlugin(PluginInterface):
    """Base class for file service plugins."""
    
    @abstractmethod
    async def read_file(self, file_path: str) -> bytes:
        """Read file content."""
        pass
    
    @abstractmethod
    async def write_file(self, file_path: str, content: bytes) -> bool:
        """Write file content."""
        pass
    
    @abstractmethod
    async def extract_text(self, file_path: str, file_type: str) -> str:
        """Extract text from file."""
        pass
    
    @abstractmethod
    def get_supported_formats(self) -> List[str]:
        """Get supported file formats."""
        pass


class NotificationServicePlugin(PluginInterface):
    """Base class for notification service plugins."""
    
    @abstractmethod
    async def send_notification(self, message: str, recipients: List[str], configuration: Dict[str, Any]) -> bool:
        """Send notification."""
        pass
    
    @abstractmethod
    def get_supported_channels(self) -> List[str]:
        """Get supported notification channels."""
        pass


class MetricsServicePlugin(PluginInterface):
    """Base class for metrics service plugins."""
    
    @abstractmethod
    async def record_metric(self, name: str, value: float, tags: Dict[str, str]) -> bool:
        """Record a metric."""
        pass
    
    @abstractmethod
    async def get_metrics(self, query: str) -> Dict[str, Any]:
        """Get metrics data."""
        pass
    
    @abstractmethod
    def get_available_metrics(self) -> List[str]:
        """Get available metrics."""
        pass


class MiddlewarePlugin(PluginInterface):
    """Base class for middleware plugins."""
    
    @abstractmethod
    async def process_request(self, request: Any) -> Any:
        """Process incoming request."""
        pass
    
    @abstractmethod
    async def process_response(self, response: Any) -> Any:
        """Process outgoing response."""
        pass


class PluginManager:
    """Ultra-modular plugin manager."""
    
    def __init__(self, plugin_directories: List[str] = None):
        self.plugin_directories = plugin_directories or ["plugins"]
        self._plugins: Dict[str, PluginInfo] = {}
        self._plugin_instances: Dict[str, Any] = {}
        self._plugin_configurations: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()
    
    async def discover_plugins(self) -> List[PluginInfo]:
        """Discover plugins in plugin directories."""
        discovered_plugins = []
        
        for plugin_dir in self.plugin_directories:
            plugin_path = Path(plugin_dir)
            if not plugin_path.exists():
                continue
            
            # Look for plugin manifest files
            for manifest_file in plugin_path.glob("**/plugin.yaml"):
                try:
                    plugin_info = await self._load_plugin_manifest(manifest_file)
                    if plugin_info:
                        discovered_plugins.append(plugin_info)
                except Exception as e:
                    logger.error(f"Failed to load plugin manifest {manifest_file}: {e}")
            
            # Look for Python plugin files
            for plugin_file in plugin_path.glob("**/*.py"):
                if plugin_file.name.startswith("plugin_") or plugin_file.name.endswith("_plugin.py"):
                    try:
                        plugin_info = await self._load_python_plugin(plugin_file)
                        if plugin_info:
                            discovered_plugins.append(plugin_info)
                    except Exception as e:
                        logger.error(f"Failed to load Python plugin {plugin_file}: {e}")
        
        return discovered_plugins
    
    async def _load_plugin_manifest(self, manifest_file: Path) -> Optional[PluginInfo]:
        """Load plugin from manifest file."""
        try:
            with open(manifest_file, 'r') as f:
                manifest_data = yaml.safe_load(f)
            
            # Extract plugin information
            plugin_id = manifest_data.get('id', str(uuid.uuid4()))
            name = manifest_data.get('name', 'Unknown Plugin')
            plugin_type = PluginType(manifest_data.get('type', 'custom'))
            
            # Create metadata
            metadata = PluginMetadata(
                name=name,
                version=manifest_data.get('version', '1.0.0'),
                description=manifest_data.get('description', ''),
                author=manifest_data.get('author', 'Unknown'),
                plugin_type=plugin_type,
                dependencies=manifest_data.get('dependencies', []),
                tags=manifest_data.get('tags', []),
                configuration_schema=manifest_data.get('configuration_schema'),
                api_version=manifest_data.get('api_version', '1.0'),
                min_system_version=manifest_data.get('min_system_version', '4.0.0'),
                max_system_version=manifest_data.get('max_system_version'),
                license=manifest_data.get('license', 'MIT'),
                homepage=manifest_data.get('homepage'),
                repository=manifest_data.get('repository'),
                documentation=manifest_data.get('documentation')
            )
            
            # Get module path and class name
            module_path = manifest_data.get('module', '')
            class_name = manifest_data.get('class', 'Plugin')
            
            # Create plugin info
            plugin_info = PluginInfo(
                id=plugin_id,
                name=name,
                plugin_type=plugin_type,
                metadata=metadata,
                module_path=module_path,
                class_name=class_name
            )
            
            return plugin_info
            
        except Exception as e:
            logger.error(f"Failed to load plugin manifest {manifest_file}: {e}")
            return None
    
    async def _load_python_plugin(self, plugin_file: Path) -> Optional[PluginInfo]:
        """Load plugin from Python file."""
        try:
            # Import the plugin module
            module_name = plugin_file.stem
            spec = importlib.util.spec_from_file_location(module_name, plugin_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Find plugin class
            plugin_class = None
            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and 
                    issubclass(obj, PluginInterface) and 
                    obj != PluginInterface):
                    plugin_class = obj
                    break
            
            if not plugin_class:
                logger.warning(f"No plugin class found in {plugin_file}")
                return None
            
            # Create plugin info
            plugin_info = PluginInfo(
                id=str(uuid.uuid4()),
                name=module_name,
                plugin_type=PluginType.CUSTOM,
                metadata=PluginMetadata(
                    name=module_name,
                    version="1.0.0",
                    description=f"Plugin loaded from {plugin_file}",
                    author="Unknown",
                    plugin_type=PluginType.CUSTOM
                ),
                module_path=module_name,
                class_name=plugin_class.__name__
            )
            
            return plugin_info
            
        except Exception as e:
            logger.error(f"Failed to load Python plugin {plugin_file}: {e}")
            return None
    
    async def load_plugin(self, plugin_info: PluginInfo, configuration: Dict[str, Any] = None) -> bool:
        """Load a plugin."""
        async with self._lock:
            try:
                # Import the plugin module
                module = importlib.import_module(plugin_info.module_path)
                plugin_class = getattr(module, plugin_info.class_name)
                
                # Create plugin instance
                plugin_instance = plugin_class()
                
                # Store plugin info and instance
                plugin_info.instance = plugin_instance
                plugin_info.configuration = configuration or {}
                self._plugins[plugin_info.id] = plugin_info
                self._plugin_instances[plugin_info.id] = plugin_instance
                self._plugin_configurations[plugin_info.id] = plugin_info.configuration
                
                # Initialize plugin
                if hasattr(plugin_instance, 'initialize'):
                    await plugin_instance.initialize(plugin_info.configuration)
                
                plugin_info.status = PluginStatus.LOADED
                
                logger.info(f"Loaded plugin: {plugin_info.name} ({plugin_info.id})")
                return True
                
            except Exception as e:
                plugin_info.status = PluginStatus.ERROR
                plugin_info.last_error = str(e)
                plugin_info.error_count += 1
                
                logger.error(f"Failed to load plugin {plugin_info.name}: {e}")
                return False
    
    async def unload_plugin(self, plugin_id: str) -> bool:
        """Unload a plugin."""
        async with self._lock:
            if plugin_id not in self._plugins:
                return False
            
            plugin_info = self._plugins[plugin_id]
            
            try:
                # Stop plugin if active
                if plugin_info.status == PluginStatus.ACTIVE:
                    await self.stop_plugin(plugin_id)
                
                # Cleanup plugin
                if plugin_info.instance and hasattr(plugin_info.instance, 'cleanup'):
                    await plugin_info.instance.cleanup()
                
                # Remove from storage
                del self._plugins[plugin_id]
                del self._plugin_instances[plugin_id]
                del self._plugin_configurations[plugin_id]
                
                logger.info(f"Unloaded plugin: {plugin_info.name} ({plugin_id})")
                return True
                
            except Exception as e:
                logger.error(f"Failed to unload plugin {plugin_info.name}: {e}")
                return False
    
    async def start_plugin(self, plugin_id: str) -> bool:
        """Start a plugin."""
        if plugin_id not in self._plugins:
            return False
        
        plugin_info = self._plugins[plugin_id]
        
        try:
            if plugin_info.instance and hasattr(plugin_info.instance, 'start'):
                await plugin_info.instance.start()
            
            plugin_info.status = PluginStatus.ACTIVE
            plugin_info.last_used = datetime.utcnow()
            
            logger.info(f"Started plugin: {plugin_info.name} ({plugin_id})")
            return True
            
        except Exception as e:
            plugin_info.status = PluginStatus.ERROR
            plugin_info.last_error = str(e)
            plugin_info.error_count += 1
            
            logger.error(f"Failed to start plugin {plugin_info.name}: {e}")
            return False
    
    async def stop_plugin(self, plugin_id: str) -> bool:
        """Stop a plugin."""
        if plugin_id not in self._plugins:
            return False
        
        plugin_info = self._plugins[plugin_id]
        
        try:
            if plugin_info.instance and hasattr(plugin_info.instance, 'stop'):
                await plugin_info.instance.stop()
            
            plugin_info.status = PluginStatus.INACTIVE
            
            logger.info(f"Stopped plugin: {plugin_info.name} ({plugin_id})")
            return True
            
        except Exception as e:
            plugin_info.status = PluginStatus.ERROR
            plugin_info.last_error = str(e)
            plugin_info.error_count += 1
            
            logger.error(f"Failed to stop plugin {plugin_info.name}: {e}")
            return False
    
    async def get_plugin(self, plugin_id: str) -> Optional[PluginInfo]:
        """Get plugin by ID."""
        return self._plugins.get(plugin_id)
    
    async def get_plugin_instance(self, plugin_id: str) -> Optional[Any]:
        """Get plugin instance by ID."""
        return self._plugin_instances.get(plugin_id)
    
    async def get_plugins_by_type(self, plugin_type: PluginType) -> List[PluginInfo]:
        """Get plugins by type."""
        return [p for p in self._plugins.values() if p.plugin_type == plugin_type]
    
    async def get_active_plugins(self) -> List[PluginInfo]:
        """Get all active plugins."""
        return [p for p in self._plugins.values() if p.status == PluginStatus.ACTIVE]
    
    async def health_check_plugin(self, plugin_id: str) -> float:
        """Perform health check on a plugin."""
        if plugin_id not in self._plugins:
            return 0.0
        
        plugin_info = self._plugins[plugin_id]
        
        try:
            if plugin_info.instance and hasattr(plugin_info.instance, 'health_check'):
                health_score = await plugin_info.instance.health_check()
            else:
                health_score = 1.0 if plugin_info.status == PluginStatus.ACTIVE else 0.0
            
            if health_score < 0.5:
                plugin_info.status = PluginStatus.ERROR
                plugin_info.error_count += 1
            elif plugin_info.status == PluginStatus.ERROR and health_score >= 0.8:
                plugin_info.status = PluginStatus.ACTIVE
            
            return health_score
            
        except Exception as e:
            plugin_info.status = PluginStatus.ERROR
            plugin_info.last_error = str(e)
            plugin_info.error_count += 1
            
            logger.error(f"Health check failed for plugin {plugin_info.name}: {e}")
            return 0.0
    
    async def get_plugin_manager_stats(self) -> Dict[str, Any]:
        """Get plugin manager statistics."""
        total_plugins = len(self._plugins)
        active_plugins = len([p for p in self._plugins.values() if p.status == PluginStatus.ACTIVE])
        error_plugins = len([p for p in self._plugins.values() if p.status == PluginStatus.ERROR])
        
        plugin_type_counts = {}
        for plugin_type in PluginType:
            plugin_type_counts[plugin_type.value] = len(
                [p for p in self._plugins.values() if p.plugin_type == plugin_type]
            )
        
        return {
            'total_plugins': total_plugins,
            'active_plugins': active_plugins,
            'error_plugins': error_plugins,
            'plugin_type_counts': plugin_type_counts,
            'plugin_directories': self.plugin_directories
        }


# Global plugin manager instance
_plugin_manager: Optional[PluginManager] = None


def get_plugin_manager() -> PluginManager:
    """Get global plugin manager instance."""
    global _plugin_manager
    if _plugin_manager is None:
        _plugin_manager = PluginManager()
    return _plugin_manager

















