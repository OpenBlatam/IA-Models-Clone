"""
Configuration System
===================

Ultra-modular configuration system with advanced patterns.
"""

import logging
import os
import json
import yaml
import threading
from typing import Dict, Any, Optional, Type, List, Callable, Union, get_type_hints
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
import inspect
import functools
from pathlib import Path

logger = logging.getLogger(__name__)

class ConfigType(str, Enum):
    """Configuration types."""
    ENVIRONMENT = "environment"
    FILE = "file"
    DATABASE = "database"
    CACHE = "cache"
    SECRET = "secret"
    DYNAMIC = "dynamic"

class ConfigScope(str, Enum):
    """Configuration scopes."""
    GLOBAL = "global"
    APPLICATION = "application"
    MODULE = "module"
    USER = "user"
    SESSION = "session"

@dataclass
class ConfigItem:
    """Configuration item."""
    key: str
    value: Any
    config_type: ConfigType
    scope: ConfigScope
    description: str = ""
    required: bool = False
    default_value: Any = None
    validation_rules: List[Callable] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=lambda: __import__('time').time())
    updated_at: float = field(default_factory=lambda: __import__('time').time())

class ConfigurationManager:
    """Configuration manager."""
    
    def __init__(self):
        self._config: Dict[str, ConfigItem] = {}
        self._lock = threading.RLock()
        self._watchers: Dict[str, List[Callable]] = {}
        self._validators: Dict[str, List[Callable]] = {}
        self._sources: Dict[str, Callable] = {}
        self._enabled: bool = True
    
    def register_config(self, key: str, value: Any, config_type: ConfigType = ConfigType.ENVIRONMENT,
                       scope: ConfigScope = ConfigScope.GLOBAL, description: str = "",
                       required: bool = False, default_value: Any = None,
                       validation_rules: List[Callable] = None, metadata: Dict[str, Any] = None) -> None:
        """Register a configuration item."""
        try:
            with self._lock:
                config_item = ConfigItem(
                    key=key,
                    value=value,
                    config_type=config_type,
                    scope=scope,
                    description=description,
                    required=required,
                    default_value=default_value,
                    validation_rules=validation_rules or [],
                    metadata=metadata or {}
                )
                
                self._config[key] = config_item
                logger.info(f"Configuration {key} registered")
        except Exception as e:
            logger.error(f"Failed to register configuration {key}: {str(e)}")
            raise
    
    def get_config(self, key: str, default_value: Any = None) -> Any:
        """Get configuration value."""
        try:
            with self._lock:
                if key in self._config:
                    return self._config[key].value
                return default_value
        except Exception as e:
            logger.error(f"Failed to get configuration {key}: {str(e)}")
            return default_value
    
    def set_config(self, key: str, value: Any) -> None:
        """Set configuration value."""
        try:
            with self._lock:
                if key in self._config:
                    # Validate value
                    if not self._validate_config(key, value):
                        logger.error(f"Configuration {key} validation failed")
                        return
                    
                    # Update value
                    self._config[key].value = value
                    self._config[key].updated_at = __import__('time').time()
                    
                    # Notify watchers
                    self._notify_watchers(key, value)
                    
                    logger.info(f"Configuration {key} updated")
                else:
                    logger.warning(f"Configuration {key} not found")
        except Exception as e:
            logger.error(f"Failed to set configuration {key}: {str(e)}")
            raise
    
    def delete_config(self, key: str) -> None:
        """Delete configuration item."""
        try:
            with self._lock:
                if key in self._config:
                    del self._config[key]
                    logger.info(f"Configuration {key} deleted")
                else:
                    logger.warning(f"Configuration {key} not found")
        except Exception as e:
            logger.error(f"Failed to delete configuration {key}: {str(e)}")
            raise
    
    def list_config(self, scope: ConfigScope = None, config_type: ConfigType = None) -> List[str]:
        """List configuration keys."""
        try:
            with self._lock:
                keys = []
                for key, item in self._config.items():
                    if scope and item.scope != scope:
                        continue
                    if config_type and item.config_type != config_type:
                        continue
                    keys.append(key)
                return keys
        except Exception as e:
            logger.error(f"Failed to list configuration: {str(e)}")
            return []
    
    def add_watcher(self, key: str, watcher: Callable) -> None:
        """Add configuration watcher."""
        try:
            with self._lock:
                if key not in self._watchers:
                    self._watchers[key] = []
                self._watchers[key].append(watcher)
                logger.info(f"Watcher added for configuration {key}")
        except Exception as e:
            logger.error(f"Failed to add watcher for {key}: {str(e)}")
            raise
    
    def remove_watcher(self, key: str, watcher: Callable) -> None:
        """Remove configuration watcher."""
        try:
            with self._lock:
                if key in self._watchers and watcher in self._watchers[key]:
                    self._watchers[key].remove(watcher)
                    logger.info(f"Watcher removed for configuration {key}")
        except Exception as e:
            logger.error(f"Failed to remove watcher for {key}: {str(e)}")
            raise
    
    def add_validator(self, key: str, validator: Callable) -> None:
        """Add configuration validator."""
        try:
            with self._lock:
                if key not in self._validators:
                    self._validators[key] = []
                self._validators[key].append(validator)
                logger.info(f"Validator added for configuration {key}")
        except Exception as e:
            logger.error(f"Failed to add validator for {key}: {str(e)}")
            raise
    
    def remove_validator(self, key: str, validator: Callable) -> None:
        """Remove configuration validator."""
        try:
            with self._lock:
                if key in self._validators and validator in self._validators[key]:
                    self._validators[key].remove(validator)
                    logger.info(f"Validator removed for configuration {key}")
        except Exception as e:
            logger.error(f"Failed to remove validator for {key}: {str(e)}")
            raise
    
    def _validate_config(self, key: str, value: Any) -> bool:
        """Validate configuration value."""
        try:
            if key in self._validators:
                for validator in self._validators[key]:
                    if not validator(value):
                        return False
            return True
        except Exception as e:
            logger.error(f"Configuration validation failed for {key}: {str(e)}")
            return False
    
    def _notify_watchers(self, key: str, value: Any) -> None:
        """Notify configuration watchers."""
        try:
            if key in self._watchers:
                for watcher in self._watchers[key]:
                    try:
                        watcher(key, value)
                    except Exception as e:
                        logger.error(f"Configuration watcher failed for {key}: {str(e)}")
        except Exception as e:
            logger.error(f"Failed to notify watchers for {key}: {str(e)}")
    
    def load_from_file(self, file_path: str, file_format: str = "json") -> None:
        """Load configuration from file."""
        try:
            with open(file_path, 'r') as f:
                if file_format.lower() == 'json':
                    data = json.load(f)
                elif file_format.lower() == 'yaml':
                    data = yaml.safe_load(f)
                else:
                    logger.error(f"Unsupported file format: {file_format}")
                    return
                
                for key, value in data.items():
                    self.set_config(key, value)
                
                logger.info(f"Configuration loaded from {file_path}")
        except Exception as e:
            logger.error(f"Failed to load configuration from {file_path}: {str(e)}")
            raise
    
    def save_to_file(self, file_path: str, file_format: str = "json") -> None:
        """Save configuration to file."""
        try:
            with self._lock:
                data = {}
                for key, item in self._config.items():
                    data[key] = item.value
                
                with open(file_path, 'w') as f:
                    if file_format.lower() == 'json':
                        json.dump(data, f, indent=2)
                    elif file_format.lower() == 'yaml':
                        yaml.dump(data, f, default_flow_style=False)
                    else:
                        logger.error(f"Unsupported file format: {file_format}")
                        return
                
                logger.info(f"Configuration saved to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save configuration to {file_path}: {str(e)}")
            raise
    
    def load_from_environment(self, prefix: str = "") -> None:
        """Load configuration from environment variables."""
        try:
            for key, value in os.environ.items():
                if prefix and not key.startswith(prefix):
                    continue
                
                config_key = key[len(prefix):] if prefix else key
                self.set_config(config_key, value)
            
            logger.info(f"Configuration loaded from environment with prefix {prefix}")
        except Exception as e:
            logger.error(f"Failed to load configuration from environment: {str(e)}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Get configuration statistics."""
        try:
            with self._lock:
                return {
                    'total_configs': len(self._config),
                    'configs_by_scope': {
                        scope.value: len([item for item in self._config.values() if item.scope == scope])
                        for scope in ConfigScope
                    },
                    'configs_by_type': {
                        config_type.value: len([item for item in self._config.values() if item.config_type == config_type])
                        for config_type in ConfigType
                    },
                    'watchers_count': sum(len(watchers) for watchers in self._watchers.values()),
                    'validators_count': sum(len(validators) for validators in self._validators.values())
                }
        except Exception as e:
            logger.error(f"Failed to get configuration stats: {str(e)}")
            return {}

class EnvironmentConfig:
    """Environment-specific configuration."""
    
    def __init__(self, environment: str = "development"):
        self.environment = environment
        self.config_manager = ConfigurationManager()
        self._load_environment_config()
    
    def _load_environment_config(self) -> None:
        """Load environment-specific configuration."""
        try:
            # Load base configuration
            self.config_manager.load_from_environment("APP_")
            
            # Load environment-specific configuration
            env_file = f"config/{self.environment}.json"
            if os.path.exists(env_file):
                self.config_manager.load_from_file(env_file)
            
            logger.info(f"Environment configuration loaded for {self.environment}")
        except Exception as e:
            logger.error(f"Failed to load environment configuration: {str(e)}")
            raise
    
    def get(self, key: str, default_value: Any = None) -> Any:
        """Get configuration value."""
        return self.config_manager.get_config(key, default_value)
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value."""
        self.config_manager.set_config(key, value)

class SecretManager:
    """Secret management system."""
    
    def __init__(self):
        self._secrets: Dict[str, str] = {}
        self._lock = threading.RLock()
        self._encryption_key: Optional[str] = None
    
    def set_encryption_key(self, key: str) -> None:
        """Set encryption key."""
        try:
            with self._lock:
                self._encryption_key = key
                logger.info("Encryption key set")
        except Exception as e:
            logger.error(f"Failed to set encryption key: {str(e)}")
            raise
    
    def store_secret(self, name: str, value: str) -> None:
        """Store a secret."""
        try:
            with self._lock:
                if self._encryption_key:
                    # Encrypt secret
                    encrypted_value = self._encrypt(value)
                    self._secrets[name] = encrypted_value
                else:
                    self._secrets[name] = value
                
                logger.info(f"Secret {name} stored")
        except Exception as e:
            logger.error(f"Failed to store secret {name}: {str(e)}")
            raise
    
    def get_secret(self, name: str) -> Optional[str]:
        """Get a secret."""
        try:
            with self._lock:
                if name not in self._secrets:
                    return None
                
                value = self._secrets[name]
                if self._encryption_key:
                    return self._decrypt(value)
                return value
        except Exception as e:
            logger.error(f"Failed to get secret {name}: {str(e)}")
            return None
    
    def delete_secret(self, name: str) -> None:
        """Delete a secret."""
        try:
            with self._lock:
                if name in self._secrets:
                    del self._secrets[name]
                    logger.info(f"Secret {name} deleted")
        except Exception as e:
            logger.error(f"Failed to delete secret {name}: {str(e)}")
            raise
    
    def _encrypt(self, value: str) -> str:
        """Encrypt value."""
        # This would implement actual encryption
        return value
    
    def _decrypt(self, value: str) -> str:
        """Decrypt value."""
        # This would implement actual decryption
        return value

# Global configuration managers
config_manager = ConfigurationManager()
environment_config = EnvironmentConfig()
secret_manager = SecretManager()

# Configuration decorators
def config_item(key: str, config_type: ConfigType = ConfigType.ENVIRONMENT,
                scope: ConfigScope = ConfigScope.GLOBAL, description: str = "",
                required: bool = False, default_value: Any = None,
                validation_rules: List[Callable] = None, metadata: Dict[str, Any] = None):
    """Decorator to register a configuration item."""
    def decorator(func):
        config_manager.register_config(
            key, func(), config_type, scope, description, required, default_value, validation_rules, metadata
        )
        return func
    return decorator

def config_watcher(key: str):
    """Decorator to register a configuration watcher."""
    def decorator(func):
        config_manager.add_watcher(key, func)
        return func
    return decorator

def config_validator(key: str):
    """Decorator to register a configuration validator."""
    def decorator(func):
        config_manager.add_validator(key, func)
        return func
    return decorator









