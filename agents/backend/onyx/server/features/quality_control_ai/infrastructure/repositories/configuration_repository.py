"""
Configuration Repository

Repository for managing configuration settings.
"""

import logging
from typing import Dict, Any, Optional
from pathlib import Path
import json
import yaml

from ...domain.exceptions import ConfigurationException

logger = logging.getLogger(__name__)


class ConfigurationRepository:
    """
    Repository for configuration management.
    
    Handles loading, saving, and updating configuration from various sources.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize repository.
        
        Args:
            config_path: Base path for configuration files
        """
        self.config_path = Path(config_path) if config_path else Path("./config")
        self.config_path.mkdir(parents=True, exist_ok=True)
        self._config_cache = {}
    
    def load_config(self, config_name: str = "default") -> Dict[str, Any]:
        """
        Load configuration from file.
        
        Args:
            config_name: Name of configuration file (without extension)
        
        Returns:
            Configuration dictionary
        
        Raises:
            ConfigurationException: If config cannot be loaded
        """
        try:
            # Check cache
            if config_name in self._config_cache:
                return self._config_cache[config_name]
            
            # Try YAML first
            yaml_path = self.config_path / f"{config_name}.yaml"
            if yaml_path.exists():
                with open(yaml_path, 'r') as f:
                    config = yaml.safe_load(f)
                    self._config_cache[config_name] = config
                    return config
            
            # Try JSON
            json_path = self.config_path / f"{config_name}.json"
            if json_path.exists():
                with open(json_path, 'r') as f:
                    config = json.load(f)
                    self._config_cache[config_name] = config
                    return config
            
            raise ConfigurationException(
                f"Configuration file not found: {config_name}"
            )
        
        except ConfigurationException:
            raise
        except Exception as e:
            logger.error(f"Failed to load config: {str(e)}", exc_info=True)
            raise ConfigurationException(f"Failed to load config: {str(e)}")
    
    def save_config(
        self,
        config: Dict[str, Any],
        config_name: str = "default",
        format: str = "yaml"
    ) -> bool:
        """
        Save configuration to file.
        
        Args:
            config: Configuration dictionary
            config_name: Name of configuration file
            format: File format ('yaml' or 'json')
        
        Returns:
            True if saved successfully
        """
        try:
            if format == "yaml":
                file_path = self.config_path / f"{config_name}.yaml"
                with open(file_path, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False, indent=2)
            elif format == "json":
                file_path = self.config_path / f"{config_name}.json"
                with open(file_path, 'w') as f:
                    json.dump(config, f, indent=2)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            # Update cache
            self._config_cache[config_name] = config
            
            logger.info(f"Configuration {config_name} saved")
            return True
        
        except Exception as e:
            logger.error(f"Failed to save config: {str(e)}", exc_info=True)
            return False
    
    def update_config(
        self,
        config_name: str,
        updates: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Update configuration with new values.
        
        Args:
            config_name: Name of configuration
            updates: Dictionary with updates
        
        Returns:
            Updated configuration
        """
        try:
            # Load current config
            config = self.load_config(config_name)
            
            # Deep merge updates
            def deep_merge(base: dict, updates: dict):
                for key, value in updates.items():
                    if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                        deep_merge(base[key], value)
                    else:
                        base[key] = value
            
            deep_merge(config, updates)
            
            # Save updated config
            self.save_config(config, config_name)
            
            return config
        
        except Exception as e:
            logger.error(f"Failed to update config: {str(e)}", exc_info=True)
            raise ConfigurationException(f"Failed to update config: {str(e)}")
    
    def get_config_value(
        self,
        config_name: str,
        key_path: str,
        default: Any = None
    ) -> Any:
        """
        Get a specific configuration value by key path.
        
        Args:
            config_name: Name of configuration
            key_path: Dot-separated key path (e.g., 'model.batch_size')
            default: Default value if not found
        
        Returns:
            Configuration value
        """
        try:
            config = self.load_config(config_name)
            
            # Navigate through nested keys
            keys = key_path.split('.')
            value = config
            for key in keys:
                if isinstance(value, dict) and key in value:
                    value = value[key]
                else:
                    return default
            
            return value
        
        except Exception as e:
            logger.error(f"Failed to get config value: {str(e)}", exc_info=True)
            return default



