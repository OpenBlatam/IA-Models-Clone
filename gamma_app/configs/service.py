"""
Configuration Service Implementation
"""

import os
import logging
import json
from typing import Any, Optional, Dict, Type, TypeVar, Union
from pathlib import Path

from .base import ConfigBase, Config, ConfigSource
from ..utils.exceptions import ConfigurationError

logger = logging.getLogger(__name__)

T = TypeVar('T')


class ConfigService(ConfigBase):
    """Configuration service implementation with type validation"""
    
    def __init__(self, env_prefix: str = "", config_file: Optional[str] = None):
        """Initialize config service"""
        self.env_prefix = env_prefix
        self.config_file = config_file
        self._config: Dict[str, Config] = {}
        self._schemas: Dict[str, Type] = {}
        self._load_from_env()
        if config_file:
            self._load_from_file(config_file)
    
    def _load_from_env(self):
        """Load configuration from environment variables"""
        for key, value in os.environ.items():
            if self.env_prefix and not key.startswith(self.env_prefix):
                continue
            
            config_key = key.replace(self.env_prefix, "").lower()
            # Try to parse as JSON for complex types
            try:
                parsed_value = json.loads(value)
            except (json.JSONDecodeError, ValueError):
                parsed_value = value
            
            self._config[config_key] = Config(
                key=config_key,
                value=parsed_value,
                source=ConfigSource.ENV
            )
    
    def _load_from_file(self, file_path: str):
        """Load configuration from file"""
        try:
            path = Path(file_path)
            if not path.exists():
                logger.warning(f"Config file not found: {file_path}")
                return
            
            with open(path, 'r') as f:
                if path.suffix == '.json':
                    data = json.load(f)
                elif path.suffix in ['.yaml', '.yml']:
                    import yaml
                    data = yaml.safe_load(f)
                else:
                    logger.warning(f"Unsupported config file format: {path.suffix}")
                    return
            
            for key, value in data.items():
                self._config[key.lower()] = Config(
                    key=key.lower(),
                    value=value,
                    source=ConfigSource.FILE
                )
        except Exception as e:
            logger.error(f"Error loading config file: {e}")
            raise ConfigurationError(f"Failed to load config file: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        config = self._config.get(key.lower())
        if config:
            return config.value
        return default
    
    def get_typed(self, key: str, expected_type: Type[T], default: Optional[T] = None) -> T:
        """Get configuration value with type validation"""
        value = self.get(key, default)
        if value is None and default is None:
            raise ConfigurationError(f"Config key '{key}' not found and no default provided")
        
        if not isinstance(value, expected_type):
            raise ConfigurationError(
                f"Config key '{key}' has wrong type. "
                f"Expected {expected_type.__name__}, got {type(value).__name__}"
            )
        return value
    
    def get_int(self, key: str, default: Optional[int] = None) -> int:
        """Get integer configuration value"""
        value = self.get(key, default)
        if value is None:
            raise ConfigurationError(f"Config key '{key}' not found")
        try:
            return int(value)
        except (ValueError, TypeError) as e:
            raise ConfigurationError(f"Config key '{key}' is not a valid integer: {e}")
    
    def get_bool(self, key: str, default: Optional[bool] = None) -> bool:
        """Get boolean configuration value"""
        value = self.get(key, default)
        if value is None:
            raise ConfigurationError(f"Config key '{key}' not found")
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ('true', '1', 'yes', 'on')
        return bool(value)
    
    def get_list(self, key: str, default: Optional[list] = None) -> list:
        """Get list configuration value"""
        value = self.get(key, default)
        if value is None:
            return default or []
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            # Try to parse as JSON list
            try:
                parsed = json.loads(value)
                if isinstance(parsed, list):
                    return parsed
            except json.JSONDecodeError:
                # Treat as comma-separated string
                return [item.strip() for item in value.split(',') if item.strip()]
        raise ConfigurationError(f"Config key '{key}' cannot be converted to list")
    
    def set(self, key: str, value: Any) -> bool:
        """Set configuration value"""
        try:
            # Validate against schema if exists
            if key.lower() in self._schemas:
                expected_type = self._schemas[key.lower()]
                if not isinstance(value, expected_type):
                    raise ConfigurationError(
                        f"Value for '{key}' must be of type {expected_type.__name__}"
                    )
            
            self._config[key.lower()] = Config(
                key=key.lower(),
                value=value,
                source=ConfigSource.MEMORY
            )
            return True
        except Exception as e:
            logger.error(f"Error setting config: {e}")
            return False
    
    def register_schema(self, key: str, value_type: Type):
        """Register type schema for a config key"""
        self._schemas[key.lower()] = value_type
    
    def require(self, *keys: str) -> Dict[str, Any]:
        """Require configuration keys to be present"""
        missing = []
        result = {}
        for key in keys:
            value = self.get(key)
            if value is None:
                missing.append(key)
            else:
                result[key] = value
        
        if missing:
            raise ConfigurationError(f"Required config keys missing: {', '.join(missing)}")
        
        return result
    
    def reload(self) -> bool:
        """Reload configuration"""
        try:
            self._config.clear()
            self._load_from_env()
            if self.config_file:
                self._load_from_file(self.config_file)
            return True
        except Exception as e:
            logger.error(f"Error reloading config: {e}")
            return False
    
    def get_all(self) -> Dict[str, Any]:
        """Get all configuration"""
        return {k: v.value for k, v in self._config.items()}
    
    def get_all_with_source(self) -> Dict[str, Dict[str, Any]]:
        """Get all configuration with source information"""
        return {
            k: {
                "value": v.value,
                "source": v.source.value,
                "description": v.description
            }
            for k, v in self._config.items()
        }

