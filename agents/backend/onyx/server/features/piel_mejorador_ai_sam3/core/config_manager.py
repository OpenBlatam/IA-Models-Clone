"""
Configuration Manager for Piel Mejorador AI SAM3
================================================

Advanced configuration management with validation and defaults.
"""

import os
import json
import logging
from typing import Dict, Any, Optional, Type, TypeVar
from pathlib import Path
from dataclasses import dataclass, field, asdict
import yaml

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class ConfigSource:
    """Configuration source."""
    name: str
    priority: int  # Higher = more priority
    data: Dict[str, Any]


class ConfigManager:
    """
    Advanced configuration manager.
    
    Features:
    - Multiple configuration sources
    - Priority-based merging
    - Environment variable support
    - File-based configuration
    - Validation
    - Defaults
    """
    
    def __init__(self, config_class: Type[T]):
        """
        Initialize config manager.
        
        Args:
            config_class: Configuration dataclass type
        """
        self.config_class = config_class
        self._sources: list[ConfigSource] = []
        self._config: Optional[T] = None
    
    def add_source(self, name: str, data: Dict[str, Any], priority: int = 0):
        """
        Add configuration source.
        
        Args:
            name: Source name
            data: Configuration data
            priority: Source priority
        """
        self._sources.append(ConfigSource(name=name, priority=priority, data=data))
        # Sort by priority (higher first)
        self._sources.sort(key=lambda x: x.priority, reverse=True)
    
    def load_from_file(self, file_path: Path, priority: int = 10):
        """
        Load configuration from file.
        
        Args:
            file_path: Path to config file
            priority: Source priority
        """
        if not file_path.exists():
            logger.warning(f"Config file not found: {file_path}")
            return
        
        try:
            if file_path.suffix in ['.yaml', '.yml']:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f) or {}
            elif file_path.suffix == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            else:
                logger.warning(f"Unsupported config file format: {file_path.suffix}")
                return
            
            self.add_source(f"file:{file_path}", data, priority)
            logger.info(f"Loaded configuration from {file_path}")
        except Exception as e:
            logger.error(f"Error loading config from {file_path}: {e}")
    
    def load_from_env(self, prefix: str = "PIEL_MEJORADOR_", priority: int = 20):
        """
        Load configuration from environment variables.
        
        Args:
            prefix: Environment variable prefix
            priority: Source priority
        """
        env_data = {}
        
        for key, value in os.environ.items():
            if key.startswith(prefix):
                # Remove prefix and convert to nested dict
                config_key = key[len(prefix):].lower()
                # Convert KEY_VALUE to nested dict
                parts = config_key.split('_')
                current = env_data
                for part in parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                current[parts[-1]] = value
        
        if env_data:
            self.add_source("environment", env_data, priority)
            logger.info(f"Loaded configuration from environment variables")
    
    def merge_configs(self) -> Dict[str, Any]:
        """
        Merge all configuration sources.
        
        Returns:
            Merged configuration dictionary
        """
        merged = {}
        
        for source in self._sources:
            merged = self._deep_merge(merged, source.data)
        
        return merged
    
    def _deep_merge(self, base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries."""
        result = base.copy()
        
        for key, value in update.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def build_config(self) -> T:
        """
        Build configuration object.
        
        Returns:
            Configuration instance
        """
        merged = self.merge_configs()
        
        # Convert to config class
        try:
            config = self.config_class(**merged)
        except TypeError as e:
            logger.error(f"Error building config: {e}")
            # Try with defaults
            config = self.config_class()
            # Update with merged values
            for key, value in merged.items():
                if hasattr(config, key):
                    setattr(config, key, value)
        
        self._config = config
        return config
    
    def get_config(self) -> T:
        """Get current configuration."""
        if self._config is None:
            self.build_config()
        return self._config
    
    def reload(self) -> T:
        """Reload configuration."""
        self._config = None
        return self.build_config()
    
    def get_sources(self) -> list[ConfigSource]:
        """Get all configuration sources."""
        return self._sources.copy()




