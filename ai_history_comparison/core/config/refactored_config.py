"""
Refactored Configuration System

Sistema de configuración centralizada y optimizada para el AI History Comparison System.
Maneja configuraciones dinámicas, validación, hot-reloading y optimización automática.
"""

import asyncio
import logging
import os
import json
import yaml
import toml
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type, TypeVar, Generic, Callable, Union, Set
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
import weakref
from contextlib import asynccontextmanager
import threading
import hashlib

logger = logging.getLogger(__name__)

T = TypeVar('T')


class ConfigType(Enum):
    """Configuration type enumeration"""
    SYSTEM = "system"
    DATABASE = "database"
    CACHE = "cache"
    SECURITY = "security"
    MONITORING = "monitoring"
    AI = "ai"
    API = "api"
    LOGGING = "logging"
    PERFORMANCE = "performance"
    CUSTOM = "custom"


class ConfigSource(Enum):
    """Configuration source enumeration"""
    ENVIRONMENT = "environment"
    FILE = "file"
    DATABASE = "database"
    API = "api"
    MEMORY = "memory"
    CACHE = "cache"


class ConfigPriority(Enum):
    """Configuration priority enumeration"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class ConfigMetadata:
    """Configuration metadata"""
    name: str
    config_type: ConfigType
    source: ConfigSource
    priority: ConfigPriority = ConfigPriority.NORMAL
    version: str = "1.0.0"
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    checksum: str = ""
    tags: Set[str] = field(default_factory=set)
    dependencies: Set[str] = field(default_factory=set)
    validation_rules: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConfigValue:
    """Configuration value with metadata"""
    value: Any
    metadata: ConfigMetadata
    is_encrypted: bool = False
    is_sensitive: bool = False
    is_readonly: bool = False
    default_value: Any = None
    validation_errors: List[str] = field(default_factory=list)


class ConfigValidator(ABC):
    """Abstract configuration validator"""
    
    @abstractmethod
    def validate(self, config: ConfigValue) -> List[str]:
        """Validate configuration value"""
        pass


class TypeValidator(ConfigValidator):
    """Type-based validator"""
    
    def __init__(self, expected_type: Type):
        self.expected_type = expected_type
    
    def validate(self, config: ConfigValue) -> List[str]:
        errors = []
        if not isinstance(config.value, self.expected_type):
            errors.append(f"Expected {self.expected_type.__name__}, got {type(config.value).__name__}")
        return errors


class RangeValidator(ConfigValidator):
    """Range-based validator"""
    
    def __init__(self, min_value: Optional[float] = None, max_value: Optional[float] = None):
        self.min_value = min_value
        self.max_value = max_value
    
    def validate(self, config: ConfigValue) -> List[str]:
        errors = []
        try:
            value = float(config.value)
            if self.min_value is not None and value < self.min_value:
                errors.append(f"Value {value} is below minimum {self.min_value}")
            if self.max_value is not None and value > self.max_value:
                errors.append(f"Value {value} is above maximum {self.max_value}")
        except (ValueError, TypeError):
            errors.append("Value is not numeric")
        return errors


class EnumValidator(ConfigValidator):
    """Enum-based validator"""
    
    def __init__(self, allowed_values: List[Any]):
        self.allowed_values = allowed_values
    
    def validate(self, config: ConfigValue) -> List[str]:
        errors = []
        if config.value not in self.allowed_values:
            errors.append(f"Value {config.value} not in allowed values: {self.allowed_values}")
        return errors


class RegexValidator(ConfigValidator):
    """Regex-based validator"""
    
    def __init__(self, pattern: str):
        import re
        self.pattern = re.compile(pattern)
    
    def validate(self, config: ConfigValue) -> List[str]:
        errors = []
        if not self.pattern.match(str(config.value)):
            errors.append(f"Value {config.value} does not match pattern {self.pattern.pattern}")
        return errors


class ConfigSection:
    """Configuration section with validation and metadata"""
    
    def __init__(self, name: str, config_type: ConfigType):
        self.name = name
        self.config_type = config_type
        self._values: Dict[str, ConfigValue] = {}
        self._validators: Dict[str, List[ConfigValidator]] = {}
        self._callbacks: List[Callable] = []
        self._lock = asyncio.Lock()
    
    def add_value(self, key: str, value: Any, metadata: ConfigMetadata, 
                  validators: List[ConfigValidator] = None) -> None:
        """Add configuration value"""
        config_value = ConfigValue(
            value=value,
            metadata=metadata,
            default_value=value
        )
        
        # Validate if validators provided
        if validators:
            for validator in validators:
                errors = validator.validate(config_value)
                config_value.validation_errors.extend(errors)
        
        self._values[key] = config_value
        if validators:
            self._validators[key] = validators
    
    def get_value(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        if key in self._values:
            return self._values[key].value
        return default
    
    def set_value(self, key: str, value: Any) -> bool:
        """Set configuration value with validation"""
        if key not in self._values:
            return False
        
        config_value = self._values[key]
        if config_value.is_readonly:
            return False
        
        # Validate new value
        if key in self._validators:
            errors = []
            for validator in self._validators[key]:
                errors.extend(validator.validate(ConfigValue(value=value, metadata=config_value.metadata)))
            
            if errors:
                logger.warning(f"Validation errors for {key}: {errors}")
                return False
        
        # Update value
        old_value = config_value.value
        config_value.value = value
        config_value.metadata.updated_at = datetime.utcnow()
        
        # Notify callbacks
        for callback in self._callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    asyncio.create_task(callback(key, old_value, value))
                else:
                    callback(key, old_value, value)
            except Exception as e:
                logger.error(f"Error in config callback: {e}")
        
        return True
    
    def add_callback(self, callback: Callable) -> None:
        """Add configuration change callback"""
        self._callbacks.append(callback)
    
    def remove_callback(self, callback: Callable) -> None:
        """Remove configuration change callback"""
        if callback in self._callbacks:
            self._callbacks.remove(callback)
    
    def get_metadata(self, key: str) -> Optional[ConfigMetadata]:
        """Get configuration metadata"""
        if key in self._values:
            return self._values[key].metadata
        return None
    
    def get_all_values(self) -> Dict[str, Any]:
        """Get all configuration values"""
        return {key: value.value for key, value in self._values.items()}
    
    def get_all_metadata(self) -> Dict[str, ConfigMetadata]:
        """Get all configuration metadata"""
        return {key: value.metadata for key, value in self._values.items()}
    
    def validate_all(self) -> Dict[str, List[str]]:
        """Validate all configuration values"""
        errors = {}
        for key, config_value in self._values.items():
            if key in self._validators:
                key_errors = []
                for validator in self._validators[key]:
                    key_errors.extend(validator.validate(config_value))
                if key_errors:
                    errors[key] = key_errors
        return errors


class RefactoredConfigManager:
    """Refactored configuration manager with advanced features"""
    
    def __init__(self):
        self._sections: Dict[str, ConfigSection] = {}
        self._sources: Dict[ConfigSource, Callable] = {}
        self._encryption_key: Optional[str] = None
        self._hot_reload_enabled: bool = False
        self._hot_reload_interval: float = 5.0
        self._hot_reload_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()
        self._callbacks: List[Callable] = []
        self._cache: Dict[str, Any] = {}
        self._cache_ttl: float = 300.0  # 5 minutes
        self._cache_timestamps: Dict[str, datetime] = {}
    
    async def initialize(self) -> None:
        """Initialize configuration manager"""
        # Register default sources
        self._register_default_sources()
        
        # Start hot reload if enabled
        if self._hot_reload_enabled:
            self._hot_reload_task = asyncio.create_task(self._hot_reload_loop())
        
        logger.info("Refactored configuration manager initialized")
    
    def _register_default_sources(self) -> None:
        """Register default configuration sources"""
        self._sources[ConfigSource.ENVIRONMENT] = self._load_from_environment
        self._sources[ConfigSource.FILE] = self._load_from_file
        self._sources[ConfigSource.MEMORY] = self._load_from_memory
    
    async def create_section(self, name: str, config_type: ConfigType) -> ConfigSection:
        """Create configuration section"""
        async with self._lock:
            if name in self._sections:
                return self._sections[name]
            
            section = ConfigSection(name, config_type)
            self._sections[name] = section
            logger.info(f"Created configuration section: {name}")
            return section
    
    async def get_section(self, name: str) -> Optional[ConfigSection]:
        """Get configuration section"""
        return self._sections.get(name)
    
    async def load_config(self, source: ConfigSource, **kwargs) -> None:
        """Load configuration from source"""
        if source not in self._sources:
            raise ValueError(f"Unknown configuration source: {source}")
        
        try:
            config_data = await self._sources[source](**kwargs)
            await self._process_config_data(config_data)
            logger.info(f"Loaded configuration from {source.value}")
        except Exception as e:
            logger.error(f"Failed to load configuration from {source.value}: {e}")
            raise
    
    async def _load_from_environment(self, prefix: str = "AI_HISTORY_") -> Dict[str, Any]:
        """Load configuration from environment variables"""
        config = {}
        for key, value in os.environ.items():
            if key.startswith(prefix):
                config_key = key[len(prefix):].lower()
                config[config_key] = value
        return config
    
    async def _load_from_file(self, file_path: str) -> Dict[str, Any]:
        """Load configuration from file"""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            if path.suffix.lower() == '.json':
                return json.load(f)
            elif path.suffix.lower() in ['.yml', '.yaml']:
                return yaml.safe_load(f)
            elif path.suffix.lower() == '.toml':
                return toml.load(f)
            else:
                raise ValueError(f"Unsupported configuration file format: {path.suffix}")
    
    async def _load_from_memory(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Load configuration from memory"""
        return config_data
    
    async def _process_config_data(self, config_data: Dict[str, Any]) -> None:
        """Process loaded configuration data"""
        for section_name, section_data in config_data.items():
            section = await self.get_section(section_name)
            if not section:
                section = await self.create_section(section_name, ConfigType.CUSTOM)
            
            for key, value in section_data.items():
                metadata = ConfigMetadata(
                    name=key,
                    config_type=section.config_type,
                    source=ConfigSource.MEMORY
                )
                section.add_value(key, value, metadata)
    
    async def get_value(self, section: str, key: str, default: Any = None) -> Any:
        """Get configuration value with caching"""
        cache_key = f"{section}.{key}"
        
        # Check cache first
        if cache_key in self._cache:
            if datetime.utcnow() - self._cache_timestamps[cache_key] < timedelta(seconds=self._cache_ttl):
                return self._cache[cache_key]
        
        # Get from section
        config_section = await self.get_section(section)
        if config_section:
            value = config_section.get_value(key, default)
            # Cache the value
            self._cache[cache_key] = value
            self._cache_timestamps[cache_key] = datetime.utcnow()
            return value
        
        return default
    
    async def set_value(self, section: str, key: str, value: Any) -> bool:
        """Set configuration value"""
        config_section = await self.get_section(section)
        if config_section:
            success = config_section.set_value(key, value)
            if success:
                # Update cache
                cache_key = f"{section}.{key}"
                self._cache[cache_key] = value
                self._cache_timestamps[cache_key] = datetime.utcnow()
                
                # Notify callbacks
                for callback in self._callbacks:
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            await callback(section, key, value)
                        else:
                            callback(section, key, value)
                    except Exception as e:
                        logger.error(f"Error in config callback: {e}")
            
            return success
        
        return False
    
    async def save_config(self, section: str, file_path: str) -> None:
        """Save configuration section to file"""
        config_section = await self.get_section(section)
        if not config_section:
            raise ValueError(f"Configuration section not found: {section}")
        
        config_data = {section: config_section.get_all_values()}
        
        path = Path(file_path)
        with open(path, 'w', encoding='utf-8') as f:
            if path.suffix.lower() == '.json':
                json.dump(config_data, f, indent=2, default=str)
            elif path.suffix.lower() in ['.yml', '.yaml']:
                yaml.dump(config_data, f, default_flow_style=False)
            elif path.suffix.lower() == '.toml':
                toml.dump(config_data, f)
            else:
                raise ValueError(f"Unsupported configuration file format: {path.suffix}")
    
    def enable_hot_reload(self, interval: float = 5.0) -> None:
        """Enable hot reload of configuration files"""
        self._hot_reload_enabled = True
        self._hot_reload_interval = interval
        logger.info(f"Hot reload enabled with interval: {interval}s")
    
    def disable_hot_reload(self) -> None:
        """Disable hot reload"""
        self._hot_reload_enabled = False
        if self._hot_reload_task:
            self._hot_reload_task.cancel()
        logger.info("Hot reload disabled")
    
    async def _hot_reload_loop(self) -> None:
        """Hot reload loop"""
        while self._hot_reload_enabled:
            try:
                await asyncio.sleep(self._hot_reload_interval)
                # Check for file changes and reload if necessary
                # This would be implemented based on specific requirements
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in hot reload loop: {e}")
    
    def add_callback(self, callback: Callable) -> None:
        """Add global configuration change callback"""
        self._callbacks.append(callback)
    
    def remove_callback(self, callback: Callable) -> None:
        """Remove global configuration change callback"""
        if callback in self._callbacks:
            self._callbacks.remove(callback)
    
    def clear_cache(self) -> None:
        """Clear configuration cache"""
        self._cache.clear()
        self._cache_timestamps.clear()
        logger.info("Configuration cache cleared")
    
    async def validate_all(self) -> Dict[str, Dict[str, List[str]]]:
        """Validate all configuration sections"""
        errors = {}
        for section_name, section in self._sections.items():
            section_errors = section.validate_all()
            if section_errors:
                errors[section_name] = section_errors
        return errors
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get configuration manager health status"""
        return {
            "sections_count": len(self._sections),
            "cache_size": len(self._cache),
            "hot_reload_enabled": self._hot_reload_enabled,
            "hot_reload_interval": self._hot_reload_interval,
            "callbacks_count": len(self._callbacks),
            "validation_errors": await self.validate_all()
        }
    
    async def shutdown(self) -> None:
        """Shutdown configuration manager"""
        self.disable_hot_reload()
        self.clear_cache()
        logger.info("Refactored configuration manager shutdown")


# Global configuration manager
config_manager = RefactoredConfigManager()


# Convenience functions
async def get_config(section: str, key: str, default: Any = None) -> Any:
    """Get configuration value"""
    return await config_manager.get_value(section, key, default)


async def set_config(section: str, key: str, value: Any) -> bool:
    """Set configuration value"""
    return await config_manager.set_value(section, key, value)


async def create_config_section(name: str, config_type: ConfigType) -> ConfigSection:
    """Create configuration section"""
    return await config_manager.create_section(name, config_type)


# Configuration decorators
def config_value(section: str, key: str, default: Any = None):
    """Decorator for configuration values"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            value = await get_config(section, key, default)
            return await func(value, *args, **kwargs)
        return wrapper
    return decorator


def config_section(section_name: str, config_type: ConfigType):
    """Decorator for configuration sections"""
    def decorator(cls):
        async def init_section():
            return await create_config_section(section_name, config_type)
        
        cls._config_section = init_section
        return cls
    return decorator





















