"""
Refactored Configuration Manager for Final Ultimate AI System

Enhanced configuration management with:
- Hot reloading and dynamic updates
- Environment-specific configurations
- Validation and type checking
- Secret management and encryption
- Configuration versioning
- Performance optimization
- Caching and memoization
- Audit logging and change tracking
"""

from __future__ import annotations
from typing import Dict, Any, Optional, List, Union, Type, Callable
import asyncio
import json
import yaml
import os
import time
import uuid
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import structlog
from pathlib import Path
import threading
from collections import defaultdict
import hashlib
import base64
from cryptography.fernet import Fernet
import jsonschema
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import weakref

logger = structlog.get_logger("refactored_config_manager")

class ConfigType(Enum):
    """Configuration type enumeration."""
    YAML = "yaml"
    JSON = "json"
    ENV = "env"
    INI = "ini"
    TOML = "toml"

class ConfigSource(Enum):
    """Configuration source enumeration."""
    FILE = "file"
    ENVIRONMENT = "environment"
    DATABASE = "database"
    API = "api"
    MEMORY = "memory"

class ValidationLevel(Enum):
    """Validation level enumeration."""
    NONE = "none"
    BASIC = "basic"
    STRICT = "strict"
    CUSTOM = "custom"

@dataclass
class ConfigItem:
    """Configuration item data structure."""
    key: str
    value: Any
    config_type: str
    source: ConfigSource
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    version: int = 1
    is_secret: bool = False
    is_encrypted: bool = False
    validation_schema: Optional[Dict[str, Any]] = None
    description: Optional[str] = None
    tags: List[str] = field(default_factory=list)

@dataclass
class ConfigChange:
    """Configuration change tracking."""
    change_id: str
    key: str
    old_value: Any
    new_value: Any
    change_type: str
    timestamp: datetime = field(default_factory=datetime.now)
    user_id: Optional[str] = None
    reason: Optional[str] = None

@dataclass
class ConfigValidation:
    """Configuration validation result."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)

class SecretManager:
    """Secret management and encryption system."""
    
    def __init__(self, encryption_key: Optional[str] = None):
        self.encryption_key = encryption_key or self._generate_key()
        self.cipher = Fernet(self.encryption_key.encode())
        self._secrets_cache: Dict[str, str] = {}
        self._lock = threading.Lock()
    
    def _generate_key(self) -> str:
        """Generate encryption key."""
        return Fernet.generate_key().decode()
    
    def encrypt(self, value: str) -> str:
        """Encrypt a value."""
        try:
            encrypted = self.cipher.encrypt(value.encode())
            return base64.b64encode(encrypted).decode()
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            raise
    
    def decrypt(self, encrypted_value: str) -> str:
        """Decrypt a value."""
        try:
            decoded = base64.b64decode(encrypted_value.encode())
            decrypted = self.cipher.decrypt(decoded)
            return decrypted.decode()
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise
    
    def store_secret(self, key: str, value: str) -> str:
        """Store a secret value."""
        with self._lock:
            encrypted_value = self.encrypt(value)
            self._secrets_cache[key] = encrypted_value
            return encrypted_value
    
    def retrieve_secret(self, key: str) -> Optional[str]:
        """Retrieve a secret value."""
        with self._lock:
            encrypted_value = self._secrets_cache.get(key)
            if encrypted_value:
                return self.decrypt(encrypted_value)
            return None

class ConfigValidator:
    """Configuration validation system."""
    
    def __init__(self):
        self.validators: Dict[str, Callable] = {}
        self.schemas: Dict[str, Dict[str, Any]] = {}
        self._register_default_validators()
    
    def _register_default_validators(self):
        """Register default validators."""
        self.validators["string"] = self._validate_string
        self.validators["integer"] = self._validate_integer
        self.validators["float"] = self._validate_float
        self.validators["boolean"] = self._validate_boolean
        self.validators["list"] = self._validate_list
        self.validators["dict"] = self._validate_dict
        self.validators["url"] = self._validate_url
        self.validators["email"] = self._validate_email
        self.validators["path"] = self._validate_path
    
    def register_validator(self, name: str, validator: Callable):
        """Register a custom validator."""
        self.validators[name] = validator
    
    def register_schema(self, name: str, schema: Dict[str, Any]):
        """Register a JSON schema."""
        self.schemas[name] = schema
    
    def validate(self, value: Any, validation_type: str, **kwargs) -> ConfigValidation:
        """Validate a configuration value."""
        try:
            if validation_type in self.validators:
                return self.validators[validation_type](value, **kwargs)
            elif validation_type in self.schemas:
                return self._validate_with_schema(value, self.schemas[validation_type])
            else:
                return ConfigValidation(False, [f"Unknown validation type: {validation_type}"])
        except Exception as e:
            return ConfigValidation(False, [f"Validation error: {str(e)}"])
    
    def _validate_string(self, value: Any, min_length: int = 0, max_length: int = None, pattern: str = None) -> ConfigValidation:
        """Validate string value."""
        errors = []
        warnings = []
        
        if not isinstance(value, str):
            errors.append("Value must be a string")
            return ConfigValidation(False, errors)
        
        if len(value) < min_length:
            errors.append(f"String length must be at least {min_length}")
        
        if max_length and len(value) > max_length:
            errors.append(f"String length must be at most {max_length}")
        
        if pattern and not re.match(pattern, value):
            errors.append(f"String does not match pattern: {pattern}")
        
        return ConfigValidation(len(errors) == 0, errors, warnings)
    
    def _validate_integer(self, value: Any, min_value: int = None, max_value: int = None) -> ConfigValidation:
        """Validate integer value."""
        errors = []
        
        if not isinstance(value, int):
            errors.append("Value must be an integer")
            return ConfigValidation(False, errors)
        
        if min_value is not None and value < min_value:
            errors.append(f"Value must be at least {min_value}")
        
        if max_value is not None and value > max_value:
            errors.append(f"Value must be at most {max_value}")
        
        return ConfigValidation(len(errors) == 0, errors)
    
    def _validate_float(self, value: Any, min_value: float = None, max_value: float = None) -> ConfigValidation:
        """Validate float value."""
        errors = []
        
        if not isinstance(value, (int, float)):
            errors.append("Value must be a number")
            return ConfigValidation(False, errors)
        
        if min_value is not None and value < min_value:
            errors.append(f"Value must be at least {min_value}")
        
        if max_value is not None and value > max_value:
            errors.append(f"Value must be at most {max_value}")
        
        return ConfigValidation(len(errors) == 0, errors)
    
    def _validate_boolean(self, value: Any) -> ConfigValidation:
        """Validate boolean value."""
        if not isinstance(value, bool):
            return ConfigValidation(False, ["Value must be a boolean"])
        
        return ConfigValidation(True)
    
    def _validate_list(self, value: Any, item_type: str = None, min_length: int = 0, max_length: int = None) -> ConfigValidation:
        """Validate list value."""
        errors = []
        
        if not isinstance(value, list):
            errors.append("Value must be a list")
            return ConfigValidation(False, errors)
        
        if len(value) < min_length:
            errors.append(f"List length must be at least {min_length}")
        
        if max_length and len(value) > max_length:
            errors.append(f"List length must be at most {max_length}")
        
        if item_type and value:
            for i, item in enumerate(value):
                item_validation = self.validate(item, item_type)
                if not item_validation.is_valid:
                    errors.append(f"Item {i}: {', '.join(item_validation.errors)}")
        
        return ConfigValidation(len(errors) == 0, errors)
    
    def _validate_dict(self, value: Any, required_keys: List[str] = None, allowed_keys: List[str] = None) -> ConfigValidation:
        """Validate dictionary value."""
        errors = []
        
        if not isinstance(value, dict):
            errors.append("Value must be a dictionary")
            return ConfigValidation(False, errors)
        
        if required_keys:
            missing_keys = set(required_keys) - set(value.keys())
            if missing_keys:
                errors.append(f"Missing required keys: {', '.join(missing_keys)}")
        
        if allowed_keys:
            invalid_keys = set(value.keys()) - set(allowed_keys)
            if invalid_keys:
                errors.append(f"Invalid keys: {', '.join(invalid_keys)}")
        
        return ConfigValidation(len(errors) == 0, errors)
    
    def _validate_url(self, value: Any) -> ConfigValidation:
        """Validate URL value."""
        import re
        url_pattern = re.compile(
            r'^https?://'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
            r'localhost|'  # localhost...
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)
        
        if not isinstance(value, str) or not url_pattern.match(value):
            return ConfigValidation(False, ["Value must be a valid URL"])
        
        return ConfigValidation(True)
    
    def _validate_email(self, value: Any) -> ConfigValidation:
        """Validate email value."""
        import re
        email_pattern = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
        
        if not isinstance(value, str) or not email_pattern.match(value):
            return ConfigValidation(False, ["Value must be a valid email address"])
        
        return ConfigValidation(True)
    
    def _validate_path(self, value: Any, must_exist: bool = False) -> ConfigValidation:
        """Validate path value."""
        if not isinstance(value, str):
            return ConfigValidation(False, ["Value must be a string"])
        
        path = Path(value)
        
        if must_exist and not path.exists():
            return ConfigValidation(False, [f"Path does not exist: {value}"])
        
        return ConfigValidation(True)
    
    def _validate_with_schema(self, value: Any, schema: Dict[str, Any]) -> ConfigValidation:
        """Validate value with JSON schema."""
        try:
            jsonschema.validate(value, schema)
            return ConfigValidation(True)
        except jsonschema.ValidationError as e:
            return ConfigValidation(False, [f"Schema validation error: {str(e)}"])
        except Exception as e:
            return ConfigValidation(False, [f"Schema validation error: {str(e)}"])

class ConfigFileWatcher(FileSystemEventHandler):
    """File system watcher for configuration files."""
    
    def __init__(self, config_manager: 'RefactoredConfigManager'):
        self.config_manager = config_manager
        self.logger = structlog.get_logger("config_file_watcher")
    
    def on_modified(self, event):
        """Handle file modification events."""
        if event.is_directory:
            return
        
        file_path = Path(event.src_path)
        if file_path.suffix in ['.yaml', '.yml', '.json', '.ini', '.toml']:
            self.logger.info(f"Configuration file modified: {file_path}")
            asyncio.create_task(self.config_manager.reload_config(file_path))

class RefactoredConfigManager:
    """Refactored configuration manager with advanced features."""
    
    def __init__(self, config_dir: str = "config", environment: str = "development"):
        self.config_dir = Path(config_dir)
        self.environment = environment
        self.config_items: Dict[str, ConfigItem] = {}
        self.config_changes: List[ConfigChange] = []
        
        # Initialize components
        self.secret_manager = SecretManager()
        self.validator = ConfigValidator()
        self.file_watcher = ConfigFileWatcher(self)
        self.observer = Observer()
        
        # Configuration cache
        self._cache: Dict[str, Any] = {}
        self._cache_timestamps: Dict[str, datetime] = {}
        self._cache_ttl = 300  # 5 minutes
        self._lock = threading.Lock()
        
        # Change listeners
        self._change_listeners: List[Callable] = []
        
        # Performance metrics
        self._metrics = {
            "load_count": 0,
            "reload_count": 0,
            "validation_count": 0,
            "cache_hits": 0,
            "cache_misses": 0
        }
    
    async def initialize(self) -> bool:
        """Initialize the configuration manager."""
        try:
            # Create config directory if it doesn't exist
            self.config_dir.mkdir(parents=True, exist_ok=True)
            
            # Start file watcher
            self.observer.schedule(self.file_watcher, str(self.config_dir), recursive=True)
            self.observer.start()
            
            # Load initial configuration
            await self.load_all_configs()
            
            logger.info("Refactored configuration manager initialized")
            return True
            
        except Exception as e:
            logger.error(f"Configuration manager initialization failed: {e}")
            return False
    
    async def shutdown(self) -> None:
        """Shutdown the configuration manager."""
        try:
            # Stop file watcher
            self.observer.stop()
            self.observer.join()
            
            logger.info("Refactored configuration manager shutdown complete")
            
        except Exception as e:
            logger.error(f"Configuration manager shutdown error: {e}")
    
    async def load_all_configs(self) -> None:
        """Load all configuration files."""
        try:
            config_files = []
            
            # Find all config files
            for pattern in ["*.yaml", "*.yml", "*.json", "*.ini", "*.toml"]:
                config_files.extend(self.config_dir.glob(pattern))
            
            # Load each config file
            for config_file in config_files:
                await self.load_config_file(config_file)
            
            self._metrics["load_count"] += 1
            logger.info(f"Loaded {len(config_files)} configuration files")
            
        except Exception as e:
            logger.error(f"Failed to load configurations: {e}")
    
    async def load_config_file(self, file_path: Path) -> None:
        """Load a specific configuration file."""
        try:
            config_type = self._detect_config_type(file_path)
            config_data = await self._parse_config_file(file_path, config_type)
            
            # Process configuration data
            for key, value in config_data.items():
                await self.set_config(
                    key=key,
                    value=value,
                    config_type=config_type.value,
                    source=ConfigSource.FILE,
                    file_path=str(file_path)
                )
            
            logger.info(f"Loaded configuration file: {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to load config file {file_path}: {e}")
    
    def _detect_config_type(self, file_path: Path) -> ConfigType:
        """Detect configuration file type."""
        suffix = file_path.suffix.lower()
        
        if suffix in ['.yaml', '.yml']:
            return ConfigType.YAML
        elif suffix == '.json':
            return ConfigType.JSON
        elif suffix == '.ini':
            return ConfigType.INI
        elif suffix == '.toml':
            return ConfigType.TOML
        else:
            return ConfigType.YAML  # Default
    
    async def _parse_config_file(self, file_path: Path, config_type: ConfigType) -> Dict[str, Any]:
        """Parse configuration file based on type."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if config_type == ConfigType.YAML:
                return yaml.safe_load(content) or {}
            elif config_type == ConfigType.JSON:
                return json.loads(content)
            elif config_type == ConfigType.INI:
                import configparser
                config = configparser.ConfigParser()
                config.read(file_path)
                return {section: dict(config[section]) for section in config.sections()}
            elif config_type == ConfigType.TOML:
                import toml
                return toml.loads(content)
            else:
                return {}
                
        except Exception as e:
            logger.error(f"Failed to parse config file {file_path}: {e}")
            return {}
    
    async def set_config(self, key: str, value: Any, config_type: str = "string",
                        source: ConfigSource = ConfigSource.MEMORY, **kwargs) -> bool:
        """Set a configuration value."""
        try:
            # Check cache first
            cache_key = f"{key}:{config_type}"
            if cache_key in self._cache:
                cache_time = self._cache_timestamps.get(cache_key)
                if cache_time and (datetime.now() - cache_time).total_seconds() < self._cache_ttl:
                    self._metrics["cache_hits"] += 1
                    if self._cache[cache_key] == value:
                        return True
                else:
                    self._metrics["cache_misses"] += 1
            
            # Validate value if validation is enabled
            validation_level = kwargs.get('validation_level', ValidationLevel.BASIC)
            if validation_level != ValidationLevel.NONE:
                validation_result = await self._validate_config_value(key, value, config_type, **kwargs)
                if not validation_result.is_valid:
                    logger.error(f"Configuration validation failed for {key}: {validation_result.errors}")
                    return False
            
            # Handle secrets
            is_secret = kwargs.get('is_secret', False)
            is_encrypted = False
            
            if is_secret and isinstance(value, str):
                encrypted_value = self.secret_manager.store_secret(key, value)
                value = encrypted_value
                is_encrypted = True
            
            # Create or update config item
            if key in self.config_items:
                old_value = self.config_items[key].value
                self.config_items[key].value = value
                self.config_items[key].updated_at = datetime.now()
                self.config_items[key].version += 1
                self.config_items[key].is_secret = is_secret
                self.config_items[key].is_encrypted = is_encrypted
                
                # Track change
                change = ConfigChange(
                    change_id=str(uuid.uuid4()),
                    key=key,
                    old_value=old_value,
                    new_value=value,
                    change_type="update",
                    user_id=kwargs.get('user_id'),
                    reason=kwargs.get('reason')
                )
                self.config_changes.append(change)
            else:
                self.config_items[key] = ConfigItem(
                    key=key,
                    value=value,
                    config_type=config_type,
                    source=source,
                    is_secret=is_secret,
                    is_encrypted=is_encrypted,
                    validation_schema=kwargs.get('validation_schema'),
                    description=kwargs.get('description'),
                    tags=kwargs.get('tags', [])
                )
                
                # Track change
                change = ConfigChange(
                    change_id=str(uuid.uuid4()),
                    key=key,
                    old_value=None,
                    new_value=value,
                    change_type="create",
                    user_id=kwargs.get('user_id'),
                    reason=kwargs.get('reason')
                )
                self.config_changes.append(change)
            
            # Update cache
            with self._lock:
                self._cache[cache_key] = value
                self._cache_timestamps[cache_key] = datetime.now()
            
            # Notify listeners
            await self._notify_change_listeners(key, value, "set")
            
            logger.info(f"Configuration set: {key}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to set configuration {key}: {e}")
            return False
    
    async def get_config(self, key: str, default: Any = None, decrypt_secret: bool = True) -> Any:
        """Get a configuration value."""
        try:
            # Check cache first
            cache_key = f"{key}:*"
            if cache_key in self._cache:
                cache_time = self._cache_timestamps.get(cache_key)
                if cache_time and (datetime.now() - cache_time).total_seconds() < self._cache_ttl:
                    self._metrics["cache_hits"] += 1
                    value = self._cache[cache_key]
                else:
                    self._metrics["cache_misses"] += 1
                    value = None
            else:
                self._metrics["cache_misses"] += 1
                value = None
            
            if value is None:
                # Get from config items
                if key in self.config_items:
                    config_item = self.config_items[key]
                    value = config_item.value
                    
                    # Decrypt secret if needed
                    if config_item.is_secret and config_item.is_encrypted and decrypt_secret:
                        value = self.secret_manager.retrieve_secret(key)
                    
                    # Update cache
                    with self._lock:
                        self._cache[cache_key] = value
                        self._cache_timestamps[cache_key] = datetime.now()
                else:
                    return default
            
            return value
            
        except Exception as e:
            logger.error(f"Failed to get configuration {key}: {e}")
            return default
    
    async def delete_config(self, key: str, user_id: Optional[str] = None) -> bool:
        """Delete a configuration value."""
        try:
            if key not in self.config_items:
                return False
            
            old_value = self.config_items[key].value
            del self.config_items[key]
            
            # Clear from cache
            cache_key = f"{key}:*"
            with self._lock:
                self._cache.pop(cache_key, None)
                self._cache_timestamps.pop(cache_key, None)
            
            # Track change
            change = ConfigChange(
                change_id=str(uuid.uuid4()),
                key=key,
                old_value=old_value,
                new_value=None,
                change_type="delete",
                user_id=user_id
            )
            self.config_changes.append(change)
            
            # Notify listeners
            await self._notify_change_listeners(key, None, "delete")
            
            logger.info(f"Configuration deleted: {key}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete configuration {key}: {e}")
            return False
    
    async def reload_config(self, file_path: Optional[Path] = None) -> None:
        """Reload configuration from files."""
        try:
            if file_path:
                await self.load_config_file(file_path)
            else:
                await self.load_all_configs()
            
            self._metrics["reload_count"] += 1
            logger.info("Configuration reloaded")
            
        except Exception as e:
            logger.error(f"Failed to reload configuration: {e}")
    
    async def _validate_config_value(self, key: str, value: Any, config_type: str, **kwargs) -> ConfigValidation:
        """Validate a configuration value."""
        try:
            self._metrics["validation_count"] += 1
            
            # Get validation schema if provided
            validation_schema = kwargs.get('validation_schema')
            if validation_schema:
                return self.validator.validate(value, "custom", schema=validation_schema)
            
            # Use config type for validation
            return self.validator.validate(value, config_type, **kwargs)
            
        except Exception as e:
            return ConfigValidation(False, [f"Validation error: {str(e)}"])
    
    async def _notify_change_listeners(self, key: str, value: Any, change_type: str) -> None:
        """Notify change listeners."""
        for listener in self._change_listeners:
            try:
                await listener(key, value, change_type)
            except Exception as e:
                logger.error(f"Change listener error: {e}")
    
    def add_change_listener(self, listener: Callable) -> None:
        """Add a configuration change listener."""
        self._change_listeners.append(listener)
    
    def remove_change_listener(self, listener: Callable) -> None:
        """Remove a configuration change listener."""
        if listener in self._change_listeners:
            self._change_listeners.remove(listener)
    
    async def get_all_configs(self, include_secrets: bool = False) -> Dict[str, Any]:
        """Get all configuration values."""
        try:
            configs = {}
            
            for key, config_item in self.config_items.items():
                if config_item.is_secret and not include_secrets:
                    configs[key] = "***SECRET***"
                else:
                    value = config_item.value
                    
                    # Decrypt secret if needed
                    if config_item.is_secret and config_item.is_encrypted and include_secrets:
                        value = self.secret_manager.retrieve_secret(key)
                    
                    configs[key] = value
            
            return configs
            
        except Exception as e:
            logger.error(f"Failed to get all configurations: {e}")
            return {}
    
    async def get_config_history(self, key: str) -> List[ConfigChange]:
        """Get configuration change history for a key."""
        return [change for change in self.config_changes if change.key == key]
    
    async def export_config(self, file_path: Path, format: str = "yaml") -> bool:
        """Export configuration to file."""
        try:
            configs = await self.get_all_configs(include_secrets=False)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                if format.lower() == "yaml":
                    yaml.dump(configs, f, default_flow_style=False, indent=2)
                elif format.lower() == "json":
                    json.dump(configs, f, indent=2)
                else:
                    return False
            
            logger.info(f"Configuration exported to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export configuration: {e}")
            return False
    
    async def clear_cache(self) -> None:
        """Clear configuration cache."""
        with self._lock:
            self._cache.clear()
            self._cache_timestamps.clear()
        
        logger.info("Configuration cache cleared")
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get configuration manager metrics."""
        return {
            "total_configs": len(self.config_items),
            "cache_size": len(self._cache),
            "change_count": len(self.config_changes),
            "metrics": self._metrics
        }
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get configuration manager health status."""
        try:
            # Check if config directory exists and is readable
            config_dir_healthy = self.config_dir.exists() and self.config_dir.is_dir()
            
            # Check if file watcher is running
            file_watcher_healthy = self.observer.is_alive()
            
            # Check cache health
            cache_health = len(self._cache) < 10000  # Reasonable cache size
            
            return {
                "healthy": all([config_dir_healthy, file_watcher_healthy, cache_health]),
                "config_directory": config_dir_healthy,
                "file_watcher": file_watcher_healthy,
                "cache_health": cache_health,
                "total_configs": len(self.config_items),
                "cache_size": len(self._cache)
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {"healthy": False, "error": str(e)}

# Example usage
async def main():
    """Example usage of refactored configuration manager."""
    config_manager = RefactoredConfigManager(
        config_dir="config",
        environment="development"
    )
    
    # Initialize configuration manager
    success = await config_manager.initialize()
    if not success:
        print("Failed to initialize configuration manager")
        return
    
    # Set some configurations
    await config_manager.set_config("database.host", "localhost", "string")
    await config_manager.set_config("database.port", 5432, "integer")
    await config_manager.set_config("api.timeout", 30.0, "float")
    await config_manager.set_config("debug.enabled", True, "boolean")
    await config_manager.set_config("api.secret_key", "secret123", "string", is_secret=True)
    
    # Get configurations
    db_host = await config_manager.get_config("database.host")
    db_port = await config_manager.get_config("database.port")
    secret_key = await config_manager.get_config("api.secret_key")
    
    print(f"Database host: {db_host}")
    print(f"Database port: {db_port}")
    print(f"Secret key: {secret_key}")
    
    # Get all configurations
    all_configs = await config_manager.get_all_configs()
    print(f"All configurations: {all_configs}")
    
    # Get metrics
    metrics = await config_manager.get_metrics()
    print(f"Metrics: {metrics}")
    
    # Get health status
    health = await config_manager.get_health_status()
    print(f"Health status: {health}")
    
    # Shutdown configuration manager
    await config_manager.shutdown()

if __name__ == "__main__":
    asyncio.run(main())


