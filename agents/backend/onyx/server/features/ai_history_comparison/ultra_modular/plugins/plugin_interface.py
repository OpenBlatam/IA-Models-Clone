"""
Plugin Interface
===============

Single responsibility: Define the interface for plugins.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from enum import Enum


class PluginType(Enum):
    """Types of plugins."""
    ANALYZER = "analyzer"
    COMPARATOR = "comparator"
    VISUALIZER = "visualizer"
    EXPORTER = "exporter"
    NOTIFIER = "notifier"
    CUSTOM = "custom"


class PluginStatus(Enum):
    """Plugin status."""
    LOADED = "loaded"
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    DISABLED = "disabled"


class PluginInterface(ABC):
    """
    Base interface for all plugins.
    
    Single Responsibility: Define the contract that all plugins must implement.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """
        Get plugin name.
        
        Returns:
            Plugin name
        """
        pass
    
    @property
    @abstractmethod
    def version(self) -> str:
        """
        Get plugin version.
        
        Returns:
            Plugin version
        """
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """
        Get plugin description.
        
        Returns:
            Plugin description
        """
        pass
    
    @property
    @abstractmethod
    def plugin_type(self) -> PluginType:
        """
        Get plugin type.
        
        Returns:
            Plugin type
        """
        pass
    
    @property
    @abstractmethod
    def author(self) -> str:
        """
        Get plugin author.
        
        Returns:
            Plugin author
        """
        pass
    
    @property
    @abstractmethod
    def dependencies(self) -> List[str]:
        """
        Get plugin dependencies.
        
        Returns:
            List of dependency names
        """
        pass
    
    @abstractmethod
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """
        Initialize the plugin.
        
        Args:
            config: Plugin configuration
            
        Returns:
            True if initialization successful
        """
        pass
    
    @abstractmethod
    async def execute(self, data: Any, **kwargs) -> Any:
        """
        Execute plugin functionality.
        
        Args:
            data: Input data
            **kwargs: Additional parameters
            
        Returns:
            Plugin execution result
        """
        pass
    
    @abstractmethod
    async def cleanup(self) -> None:
        """
        Cleanup plugin resources.
        """
        pass
    
    @abstractmethod
    def get_config_schema(self) -> Dict[str, Any]:
        """
        Get plugin configuration schema.
        
        Returns:
            Configuration schema
        """
        pass
    
    @abstractmethod
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate plugin configuration.
        
        Args:
            config: Configuration to validate
            
        Returns:
            True if configuration is valid
        """
        pass
    
    def get_status(self) -> PluginStatus:
        """
        Get plugin status.
        
        Returns:
            Current plugin status
        """
        return PluginStatus.ACTIVE
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get plugin metadata.
        
        Returns:
            Plugin metadata
        """
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "type": self.plugin_type.value,
            "author": self.author,
            "dependencies": self.dependencies,
            "status": self.get_status().value
        }
    
    def is_compatible(self, system_version: str) -> bool:
        """
        Check if plugin is compatible with system version.
        
        Args:
            system_version: System version
            
        Returns:
            True if compatible
        """
        # Default implementation - can be overridden
        return True
    
    def get_requirements(self) -> List[str]:
        """
        Get plugin requirements.
        
        Returns:
            List of required packages
        """
        return []
    
    def get_permissions(self) -> List[str]:
        """
        Get required permissions.
        
        Returns:
            List of required permissions
        """
        return []
    
    def __str__(self) -> str:
        """String representation."""
        return f"{self.name} v{self.version} ({self.plugin_type.value})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"Plugin(name='{self.name}', version='{self.version}', type='{self.plugin_type.value}')"




