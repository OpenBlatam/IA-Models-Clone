from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import os
import json
import yaml
from typing import Dict, List, Any, Optional, Union, Callable, Awaitable
from datetime import datetime, timezone, timedelta
from contextlib import asynccontextmanager
import structlog
from dataclasses import dataclass, asdict
from enum import Enum
import weakref
from functools import lru_cache, wraps
import inspect
import gc
from pathlib import Path
from typing import Any, List, Dict, Optional
import logging
import asyncio
#!/usr/bin/env python3
"""
Configuration Manager for FastAPI Dependency Injection
Manages application configuration and settings.
"""


logger = structlog.get_logger()

# =============================================================================
# Configuration Types
# =============================================================================

class ConfigSource(Enum):
    """Configuration source enumeration."""
    ENVIRONMENT = "environment"
    FILE = "file"
    DATABASE = "database"
    REMOTE = "remote"
    MEMORY = "memory"

class ConfigFormat(Enum):
    """Configuration format enumeration."""
    JSON = "json"
    YAML = "yaml"
    INI = "ini"
    ENV = "env"

@dataclass
class ConfigItem:
    """Configuration item."""
    key: str
    value: Any
    source: ConfigSource
    format: ConfigFormat
    description: Optional[str] = None
    validation_rules: Optional[Dict[str, Any]] = None
    created_at: datetime = None
    updated_at: datetime = None
    
    def __post_init__(self) -> Any:
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)
        if self.updated_at is None:
            self.updated_at = datetime.now(timezone.utc)

@dataclass
class ConfigStats:
    """Configuration statistics."""
    total_items: int = 0
    sources: Dict[str, int] = None
    formats: Dict[str, int] = None
    last_reload: Optional[datetime] = None
    reload_count: int = 0
    validation_errors: int = 0

# =============================================================================
# Configuration Validators
# =============================================================================

class ConfigValidator:
    """Configuration validator."""
    
    @staticmethod
    def validate_string(value: Any, rules: Dict[str, Any]) -> bool:
        """Validate string value."""
        if not isinstance(value, str):
            return False
        
        min_length = rules.get('min_length', 0)
        max_length = rules.get('max_length', float('inf'))
        
        return min_length <= len(value) <= max_length
    
    @staticmethod
    def validate_integer(value: Any, rules: Dict[str, Any]) -> bool:
        """Validate integer value."""
        try:
            int_value = int(value)
        except (ValueError, TypeError):
            return False
        
        min_value = rules.get('min_value', float('-inf'))
        max_value = rules.get('max_value', float('inf'))
        
        return min_value <= int_value <= max_value
    
    @staticmethod
    def validate_float(value: Any, rules: Dict[str, Any]) -> bool:
        """Validate float value."""
        try:
            float_value = float(value)
        except (ValueError, TypeError):
            return False
        
        min_value = rules.get('min_value', float('-inf'))
        max_value = rules.get('max_value', float('inf'))
        
        return min_value <= float_value <= max_value
    
    @staticmethod
    def validate_boolean(value: Any, rules: Dict[str, Any]) -> bool:
        """Validate boolean value."""
        if isinstance(value, bool):
            return True
        
        if isinstance(value, str):
            return value.lower() in ['true', 'false', '1', '0', 'yes', 'no']
        
        return False
    
    @staticmethod
    def validate_list(value: Any, rules: Dict[str, Any]) -> bool:
        """Validate list value."""
        if not isinstance(value, list):
            return False
        
        min_length = rules.get('min_length', 0)
        max_length = rules.get('max_length', float('inf'))
        
        return min_length <= len(value) <= max_length
    
    @staticmethod
    def validate_dict(value: Any, rules: Dict[str, Any]) -> bool:
        """Validate dictionary value."""
        if not isinstance(value, dict):
            return False
        
        required_keys = rules.get('required_keys', [])
        for key in required_keys:
            if key not in value:
                return False
        
        return True

# =============================================================================
# Configuration Sources
# =============================================================================

class ConfigSourceBase:
    """Base class for configuration sources."""
    
    def __init__(self, source_type: ConfigSource):
        
    """__init__ function."""
self.source_type = source_type
        self._is_initialized = False
    
    async def initialize(self) -> None:
        """Initialize the configuration source."""
        if self._is_initialized:
            return
        
        try:
            await self._initialize_internal()
            self._is_initialized = True
            logger.info(f"Configuration source {self.source_type.value} initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize configuration source {self.source_type.value}: {e}")
            raise
    
    async def _initialize_internal(self) -> None:
        """Internal initialization method to be implemented by subclasses."""
        pass
    
    async def load_config(self) -> Dict[str, ConfigItem]:
        """Load configuration from source."""
        if not self._is_initialized:
            raise RuntimeError(f"Configuration source {self.source_type.value} not initialized")
        
        return await self._load_config_internal()
    
    async def _load_config_internal(self) -> Dict[str, ConfigItem]:
        """Internal load method to be implemented by subclasses."""
        raise NotImplementedError
    
    async def save_config(self, config_items: Dict[str, ConfigItem]) -> bool:
        """Save configuration to source."""
        if not self._is_initialized:
            raise RuntimeError(f"Configuration source {self.source_type.value} not initialized")
        
        return await self._save_config_internal(config_items)
    
    async def _save_config_internal(self, config_items: Dict[str, ConfigItem]) -> bool:
        """Internal save method to be implemented by subclasses."""
        raise NotImplementedError

class EnvironmentConfigSource(ConfigSourceBase):
    """Environment variable configuration source."""
    
    def __init__(self, prefix: str = ""):
        
    """__init__ function."""
super().__init__(ConfigSource.ENVIRONMENT)
        self.prefix = prefix
    
    async def _initialize_internal(self) -> None:
        """Initialize environment source."""
        # Nothing special to initialize for environment variables
        pass
    
    async def _load_config_internal(self) -> Dict[str, ConfigItem]:
        """Load configuration from environment variables."""
        config_items = {}
        
        for key, value in os.environ.items():
            if self.prefix and not key.startswith(self.prefix):
                continue
            
            # Remove prefix if present
            if self.prefix:
                config_key = key[len(self.prefix):]
            else:
                config_key = key
            
            # Convert value to appropriate type
            converted_value = self._convert_value(value)
            
            config_items[config_key] = ConfigItem(
                key=config_key,
                value=converted_value,
                source=self.source_type,
                format=ConfigFormat.ENV,
                description=f"Environment variable: {key}"
            )
        
        return config_items
    
    def _convert_value(self, value: str) -> Any:
        """Convert string value to appropriate type."""
        # Try to convert to boolean
        if value.lower() in ['true', 'false']:
            return value.lower() == 'true'
        
        # Try to convert to integer
        try:
            return int(value)
        except ValueError:
            pass
        
        # Try to convert to float
        try:
            return float(value)
        except ValueError:
            pass
        
        # Return as string
        return value
    
    async def _save_config_internal(self, config_items: Dict[str, ConfigItem]) -> bool:
        """Save configuration to environment variables."""
        # Environment variables are typically read-only in applications
        # This would be used for setting environment variables in development
        logger.warning("Saving to environment variables is not supported")
        return False

class FileConfigSource(ConfigSourceBase):
    """File-based configuration source."""
    
    def __init__(self, file_path: str, format: ConfigFormat = ConfigFormat.JSON):
        
    """__init__ function."""
super().__init__(ConfigSource.FILE)
        self.file_path = file_path
        self.format = format
    
    async def _initialize_internal(self) -> None:
        """Initialize file source."""
        # Ensure file exists
        path = Path(self.file_path)
        if not path.exists():
            # Create default configuration file
            await self._create_default_config()
    
    async def _create_default_config(self) -> None:
        """Create default configuration file."""
        default_config = {
            "app": {
                "name": "HeyGen AI API",
                "version": "1.0.0",
                "debug": False
            },
            "database": {
                "url": "postgresql+asyncpg://user:password@localhost/heygen_ai",
                "pool_size": 20,
                "max_overflow": 30
            },
            "redis": {
                "url": "redis://localhost:6379",
                "pool_size": 10
            },
            "security": {
                "secret_key": "your-secret-key-here",
                "algorithm": "HS256",
                "access_token_expire_minutes": 30
            }
        }
        
        await self._save_config_internal({
            key: ConfigItem(
                key=key,
                value=value,
                source=self.source_type,
                format=self.format,
                description=f"Default configuration for {key}"
            )
            for key, value in self._flatten_dict(default_config).items()
        })
    
    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = '') -> Dict[str, Any]:
        """Flatten nested dictionary."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}.{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    async def _load_config_internal(self) -> Dict[str, ConfigItem]:
        """Load configuration from file."""
        try:
            path = Path(self.file_path)
            if not path.exists():
                return {}
            
            with open(path, 'r', encoding='utf-8') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                if self.format == ConfigFormat.JSON:
                    data = json.load(f)
                elif self.format == ConfigFormat.YAML:
                    data = yaml.safe_load(f)
                else:
                    raise ValueError(f"Unsupported format: {self.format}")
            
            config_items = {}
            for key, value in self._flatten_dict(data).items():
                config_items[key] = ConfigItem(
                    key=key,
                    value=value,
                    source=self.source_type,
                    format=self.format,
                    description=f"Configuration from {self.file_path}"
                )
            
            return config_items
            
        except Exception as e:
            logger.error(f"Failed to load configuration from {self.file_path}: {e}")
            return {}
    
    async def _save_config_internal(self, config_items: Dict[str, ConfigItem]) -> bool:
        """Save configuration to file."""
        try:
            # Convert flat config items back to nested structure
            nested_data = {}
            for key, item in config_items.items():
                keys = key.split('.')
                current = nested_data
                for k in keys[:-1]:
                    if k not in current:
                        current[k] = {}
                    current = current[k]
                current[keys[-1]] = item.value
            
            path = Path(self.file_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(path, 'w', encoding='utf-8') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                if self.format == ConfigFormat.JSON:
                    json.dump(nested_data, f, indent=2, ensure_ascii=False)
                elif self.format == ConfigFormat.YAML:
                    yaml.dump(nested_data, f, default_flow_style=False, allow_unicode=True)
                else:
                    raise ValueError(f"Unsupported format: {self.format}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to save configuration to {self.file_path}: {e}")
            return False

# =============================================================================
# Configuration Manager
# =============================================================================

class ConfigManager:
    """Main configuration manager."""
    
    def __init__(self) -> Any:
        self.sources: List[ConfigSourceBase] = []
        self.config_items: Dict[str, ConfigItem] = {}
        self.stats = ConfigStats()
        self.validator = ConfigValidator()
        self._is_initialized = False
        self._reload_task: Optional[asyncio.Task] = None
    
    def add_source(self, source: ConfigSourceBase) -> None:
        """Add a configuration source."""
        self.sources.append(source)
        logger.info(f"Added configuration source: {source.source_type.value}")
    
    async def initialize(self) -> None:
        """Initialize configuration manager."""
        if self._is_initialized:
            return
        
        logger.info("Initializing configuration manager...")
        
        # Initialize all sources
        for source in self.sources:
            await source.initialize()
        
        # Load configuration from all sources
        await self.reload_config()
        
        self._is_initialized = True
        logger.info("Configuration manager initialized successfully")
    
    async def reload_config(self) -> None:
        """Reload configuration from all sources."""
        logger.info("Reloading configuration...")
        
        new_config_items = {}
        source_counts = {}
        format_counts = {}
        
        # Load from all sources
        for source in self.sources:
            try:
                source_items = await source.load_config()
                
                # Merge with existing items (later sources override earlier ones)
                for key, item in source_items.items():
                    new_config_items[key] = item
                    
                    # Update statistics
                    source_counts[source.source_type.value] = source_counts.get(source.source_type.value, 0) + 1
                    format_counts[item.format.value] = format_counts.get(item.format.value, 0) + 1
                
            except Exception as e:
                logger.error(f"Failed to load configuration from {source.source_type.value}: {e}")
                self.stats.validation_errors += 1
        
        # Validate configuration items
        for key, item in new_config_items.items():
            if item.validation_rules:
                if not self._validate_item(item):
                    logger.warning(f"Configuration item {key} failed validation")
                    self.stats.validation_errors += 1
        
        self.config_items = new_config_items
        self.stats.total_items = len(new_config_items)
        self.stats.sources = source_counts
        self.stats.formats = format_counts
        self.stats.last_reload = datetime.now(timezone.utc)
        self.stats.reload_count += 1
        
        logger.info(f"Configuration reloaded: {len(new_config_items)} items")
    
    def _validate_item(self, item: ConfigItem) -> bool:
        """Validate a configuration item."""
        if not item.validation_rules:
            return True
        
        value_type = item.validation_rules.get('type', 'string')
        
        if value_type == 'string':
            return self.validator.validate_string(item.value, item.validation_rules)
        elif value_type == 'integer':
            return self.validator.validate_integer(item.value, item.validation_rules)
        elif value_type == 'float':
            return self.validator.validate_float(item.value, item.validation_rules)
        elif value_type == 'boolean':
            return self.validator.validate_boolean(item.value, item.validation_rules)
        elif value_type == 'list':
            return self.validator.validate_list(item.value, item.validation_rules)
        elif value_type == 'dict':
            return self.validator.validate_dict(item.value, item.validation_rules)
        
        return True
    
    def get_setting(self, key: str, default: Any = None) -> Optional[Dict[str, Any]]:
        """Get a configuration setting."""
        if not self._is_initialized:
            raise RuntimeError("Configuration manager not initialized")
        
        item = self.config_items.get(key)
        if item is None:
            return default
        
        return item.value
    
    def get_setting_with_metadata(self, key: str) -> Optional[ConfigItem]:
        """Get a configuration setting with metadata."""
        if not self._is_initialized:
            raise RuntimeError("Configuration manager not initialized")
        
        return self.config_items.get(key)
    
    def set_setting(self, key: str, value: Any, description: Optional[str] = None) -> bool:
        """Set a configuration setting."""
        if not self._is_initialized:
            raise RuntimeError("Configuration manager not initialized")
        
        # Create or update config item
        if key in self.config_items:
            item = self.config_items[key]
            item.value = value
            item.updated_at = datetime.now(timezone.utc)
            if description:
                item.description = description
        else:
            self.config_items[key] = ConfigItem(
                key=key,
                value=value,
                source=ConfigSource.MEMORY,
                format=ConfigFormat.JSON,
                description=description
            )
        
        return True
    
    def get_all_settings(self) -> Dict[str, Any]:
        """Get all configuration settings."""
        if not self._is_initialized:
            raise RuntimeError("Configuration manager not initialized")
        
        return {key: item.value for key, item in self.config_items.items()}
    
    def get_settings_by_prefix(self, prefix: str) -> Dict[str, Any]:
        """Get configuration settings by prefix."""
        if not self._is_initialized:
            raise RuntimeError("Configuration manager not initialized")
        
        return {
            key: item.value 
            for key, item in self.config_items.items() 
            if key.startswith(prefix)
        }
    
    async def save_config(self) -> bool:
        """Save configuration to all writable sources."""
        if not self._is_initialized:
            raise RuntimeError("Configuration manager not initialized")
        
        success = True
        for source in self.sources:
            try:
                if not await source.save_config(self.config_items):
                    success = False
            except Exception as e:
                logger.error(f"Failed to save configuration to {source.source_type.value}: {e}")
                success = False
        
        return success
    
    def get_stats(self) -> Dict[str, Any]:
        """Get configuration statistics."""
        return {
            "total_items": self.stats.total_items,
            "sources": self.stats.sources,
            "formats": self.stats.formats,
            "last_reload": self.stats.last_reload.isoformat() if self.stats.last_reload else None,
            "reload_count": self.stats.reload_count,
            "validation_errors": self.stats.validation_errors
        }
    
    async def cleanup(self) -> None:
        """Cleanup configuration manager."""
        if not self._is_initialized:
            return
        
        # Cancel reload task if running
        if self._reload_task:
            self._reload_task.cancel()
            try:
                await self._reload_task
            except asyncio.CancelledError:
                pass
        
        self._is_initialized = False
        logger.info("Configuration manager cleaned up successfully")

# =============================================================================
# Configuration Decorators
# =============================================================================

def config_setting(key: str, default: Any = None, description: Optional[str] = None):
    """Decorator for configuration settings."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # This would be used with the configuration manager
            # The actual configuration access would happen at the function level
            return func(*args, **kwargs)
        return wrapper
    return decorator

# =============================================================================
# Export
# =============================================================================

__all__ = [
    "ConfigSource",
    "ConfigFormat",
    "ConfigItem",
    "ConfigStats",
    "ConfigValidator",
    "ConfigSourceBase",
    "EnvironmentConfigSource",
    "FileConfigSource",
    "ConfigManager",
    "config_setting",
] 