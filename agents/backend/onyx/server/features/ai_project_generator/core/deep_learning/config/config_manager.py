"""
Configuration Manager - YAML Configuration Handling
====================================================

Manages configuration files for models, training, and experiments.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
import json

logger = logging.getLogger(__name__)


class ConfigManager:
    """
    Manages configuration files for deep learning projects.
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize config manager.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config: Dict[str, Any] = {}
        
        if config_path and config_path.exists():
            self.load(config_path)
    
    def load(self, config_path: Path) -> Dict[str, Any]:
        """
        Load configuration from file.
        
        Args:
            config_path: Path to YAML or JSON config file
            
        Returns:
            Loaded configuration dictionary
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        try:
            if config_path.suffix in ['.yaml', '.yml']:
                with open(config_path, 'r') as f:
                    self.config = yaml.safe_load(f) or {}
            elif config_path.suffix == '.json':
                with open(config_path, 'r') as f:
                    self.config = json.load(f)
            else:
                raise ValueError(f"Unsupported config file format: {config_path.suffix}")
            
            logger.info(f"Configuration loaded from {config_path}")
            return self.config
            
        except Exception as e:
            logger.error(f"Error loading config: {e}", exc_info=True)
            raise
    
    def save(self, config_path: Path, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Save configuration to file.
        
        Args:
            config_path: Path to save config file
            config: Configuration to save (uses self.config if None)
        """
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        config_to_save = config or self.config
        
        try:
            if config_path.suffix in ['.yaml', '.yml']:
                with open(config_path, 'w') as f:
                    yaml.dump(config_to_save, f, default_flow_style=False, sort_keys=False)
            elif config_path.suffix == '.json':
                with open(config_path, 'w') as f:
                    json.dump(config_to_save, f, indent=2)
            else:
                raise ValueError(f"Unsupported config file format: {config_path.suffix}")
            
            logger.info(f"Configuration saved to {config_path}")
            
        except Exception as e:
            logger.error(f"Error saving config: {e}", exc_info=True)
            raise
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value.
        
        Args:
            key: Configuration key (supports dot notation, e.g., 'model.hidden_size')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value.
        
        Args:
            key: Configuration key (supports dot notation)
            value: Value to set
        """
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def merge(self, other_config: Dict[str, Any], overwrite: bool = True) -> None:
        """
        Merge another configuration into current config.
        
        Args:
            other_config: Configuration to merge
            overwrite: Whether to overwrite existing values
        """
        if overwrite:
            self.config = {**self.config, **other_config}
        else:
            # Only add new keys
            for key, value in other_config.items():
                if key not in self.config:
                    self.config[key] = value


def load_config(config_path: Path) -> Dict[str, Any]:
    """
    Load configuration from file.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    manager = ConfigManager()
    return manager.load(config_path)


def save_config(config: Dict[str, Any], config_path: Path) -> None:
    """
    Save configuration to file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save config file
    """
    manager = ConfigManager()
    manager.save(config_path, config)


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two configurations.
    
    Args:
        base_config: Base configuration
        override_config: Configuration to override with
        
    Returns:
        Merged configuration
    """
    merged = base_config.copy()
    merged.update(override_config)
    return merged



