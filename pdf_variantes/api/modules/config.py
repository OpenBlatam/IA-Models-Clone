"""
Module Configuration
Configuration management for modules
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ModuleConfig:
    """Configuration for a module"""
    name: str
    enabled: bool = True
    version: str = "1.0.0"
    dependencies: list = None
    settings: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.settings is None:
            self.settings = {}


class ModulesConfig:
    """Configuration for all modules"""
    
    def __init__(self):
        self.modules: Dict[str, ModuleConfig] = {}
    
    def register_module(self, config: ModuleConfig) -> None:
        """Register module configuration"""
        self.modules[config.name] = config
    
    def get_module_config(self, name: str) -> Optional[ModuleConfig]:
        """Get module configuration"""
        return self.modules.get(name)
    
    def is_module_enabled(self, name: str) -> bool:
        """Check if module is enabled"""
        config = self.get_module_config(name)
        return config.enabled if config else False
    
    def get_module_dependencies(self, name: str) -> list:
        """Get module dependencies"""
        config = self.get_module_config(name)
        return config.dependencies if config else []
    
    def get_module_settings(self, name: str) -> Dict[str, Any]:
        """Get module settings"""
        config = self.get_module_config(name)
        return config.settings if config else {}
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ModulesConfig':
        """Create from dictionary"""
        instance = cls()
        
        for module_name, module_config in config_dict.items():
            config = ModuleConfig(
                name=module_name,
                **module_config
            )
            instance.register_module(config)
        
        return instance


# Default configuration
DEFAULT_MODULES_CONFIG = {
    "document": {
        "enabled": True,
        "version": "1.0.0",
        "dependencies": [],
        "settings": {
            "auto_process": True,
            "max_file_size_mb": 100
        }
    },
    "variant": {
        "enabled": True,
        "version": "1.0.0",
        "dependencies": ["document"],
        "settings": {
            "max_variants": 1000,
            "default_count": 10
        }
    },
    "topic": {
        "enabled": True,
        "version": "1.0.0",
        "dependencies": ["document"],
        "settings": {
            "min_relevance": 0.5,
            "max_topics": 200
        }
    }
}






