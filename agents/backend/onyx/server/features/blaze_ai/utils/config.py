"""
Configuration management utilities for Blaze AI.

This module provides flexible configuration management including:
- YAML/JSON configuration loading
- Environment variable overrides
- Configuration validation
- Dynamic configuration updates
"""

from __future__ import annotations

import os
import yaml
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field

from ..core.interfaces import CoreConfig

class ConfigManager:
    """Configuration management system for Blaze AI."""
    
    def __init__(self, config_path: Optional[str] = None, env_prefix: str = "BLAZE_AI"):
        self.config_path = config_path
        self.env_prefix = env_prefix
        self.config: Dict[str, Any] = {}
        self.overrides: Dict[str, Any] = {}
        
        if config_path:
            self.load_config(config_path)
        
        self._load_environment_overrides()
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from file."""
        path = Path(config_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        if path.suffix.lower() in ['.yaml', '.yml']:
            with open(path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f) or {}
        elif path.suffix.lower() == '.json':
            with open(path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
        else:
            raise ValueError(f"Unsupported configuration file format: {path.suffix}")
        
        return self.config
    
    def _load_environment_overrides(self):
        """Load configuration overrides from environment variables."""
        for key, value in os.environ.items():
            if key.startswith(self.env_prefix):
                # Convert BLAZE_AI_SYSTEM_MODE to system_mode
                config_key = key[len(self.env_prefix) + 1:].lower().replace('_', '.')
                self.overrides[config_key] = self._parse_env_value(value)
    
    def _parse_env_value(self, value: str) -> Any:
        """Parse environment variable value to appropriate type."""
        # Try to parse as JSON first
        try:
            return json.loads(value)
        except (json.JSONDecodeError, ValueError):
            pass
        
        # Try to parse as boolean
        if value.lower() in ['true', 'false']:
            return value.lower() == 'true'
        
        # Try to parse as number
        try:
            if '.' in value:
                return float(value)
            return int(value)
        except ValueError:
            pass
        
        # Return as string
        return value
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key (supports dot notation)."""
        # Check overrides first
        if key in self.overrides:
            return self.overrides[key]
        
        # Check config with dot notation support
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """Set configuration value by key (supports dot notation)."""
        keys = key.split('.')
        config = self.config
        
        # Navigate to the parent of the target key
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # Set the value
        config[keys[-1]] = value
    
    def override(self, key: str, value: Any):
        """Override configuration value (takes precedence over file config)."""
        self.overrides[key] = value
    
    def get_nested(self, key: str, default: Any = None) -> Dict[str, Any]:
        """Get nested configuration section."""
        value = self.get(key, default)
        if isinstance(value, dict):
            return value
        return default or {}
    
    def validate_config(self) -> List[str]:
        """Validate configuration and return list of errors."""
        errors = []
        
        # Basic validation
        if not self.config:
            errors.append("No configuration loaded")
            return errors
        
        # Validate required fields
        required_fields = ['system_mode', 'log_level']
        for field in required_fields:
            if not self.get(field):
                errors.append(f"Missing required field: {field}")
        
        # Validate port conflicts
        api_port = self.get('api.port')
        gradio_port = self.get('gradio.gradio_port')
        metrics_port = self.get('monitoring.metrics_port')
        
        if api_port and gradio_port and api_port == gradio_port:
            errors.append("API port cannot be the same as Gradio port")
        
        if metrics_port and (metrics_port == api_port or metrics_port == gradio_port):
            errors.append("Metrics port cannot conflict with API or Gradio ports")
        
        return errors
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary with overrides applied."""
        config = self.config.copy()
        
        # Apply overrides
        for key, value in self.overrides.items():
            keys = key.split('.')
            target = config
            
            # Navigate to the target location
            for k in keys[:-1]:
                if k not in target:
                    target[k] = {}
                target = target[k]
            
            # Set the value
            target[keys[-1]] = value
        
        return config
    
    def to_core_config(self) -> CoreConfig:
        """Convert to CoreConfig object."""
        config_dict = self.to_dict()
        return CoreConfig(**config_dict)
    
    def save_config(self, output_path: str, format: str = 'yaml'):
        """Save current configuration to file."""
        config_dict = self.to_dict()
        path = Path(output_path)
        
        if format.lower() == 'yaml':
            with open(path, 'w', encoding='utf-8') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        elif format.lower() == 'json':
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2)
        else:
            raise ValueError(f"Unsupported output format: {format}")
    
    def reload(self):
        """Reload configuration from file."""
        if self.config_path:
            self.load_config(self.config_path)
        self._load_environment_overrides()
    
    def get_effective_config(self) -> Dict[str, Any]:
        """Get effective configuration with all overrides applied."""
        return self.to_dict()

class ConfigValidator:
    """Configuration validation utilities."""
    
    @staticmethod
    def validate_port(port: int, name: str = "port") -> List[str]:
        """Validate port number."""
        errors = []
        if not isinstance(port, int) or port < 1024 or port > 65535:
            errors.append(f"{name} must be an integer between 1024 and 65535")
        return errors
    
    @staticmethod
    def validate_path(path: str, name: str = "path") -> List[str]:
        """Validate file/directory path."""
        errors = []
        if not path or not isinstance(path, str):
            errors.append(f"{name} must be a non-empty string")
        return errors
    
    @staticmethod
    def validate_boolean(value: Any, name: str = "boolean") -> List[str]:
        """Validate boolean value."""
        errors = []
        if not isinstance(value, bool):
            errors.append(f"{name} must be a boolean value")
        return errors
    
    @staticmethod
    def validate_integer(value: Any, min_val: Optional[int] = None, max_val: Optional[int] = None, name: str = "integer") -> List[str]:
        """Validate integer value."""
        errors = []
        if not isinstance(value, int):
            errors.append(f"{name} must be an integer")
        else:
            if min_val is not None and value < min_val:
                errors.append(f"{name} must be >= {min_val}")
            if max_val is not None and value > max_val:
                errors.append(f"{name} must be <= {max_val}")
        return errors

# Utility functions
def load_config_from_file(config_path: str) -> Dict[str, Any]:
    """Quick configuration loading from file."""
    manager = ConfigManager(config_path)
    return manager.to_dict()

def create_config_manager(config_path: Optional[str] = None, env_prefix: str = "BLAZE_AI") -> ConfigManager:
    """Create a new configuration manager."""
    return ConfigManager(config_path, env_prefix)

def validate_config_file(config_path: str) -> List[str]:
    """Validate configuration file and return list of errors."""
    try:
        manager = ConfigManager(config_path)
        return manager.validate_config()
    except Exception as e:
        return [f"Configuration file error: {e}"]

# Export main classes
__all__ = [
    "ConfigManager",
    "ConfigValidator",
    "load_config_from_file",
    "create_config_manager",
    "validate_config_file"
]


