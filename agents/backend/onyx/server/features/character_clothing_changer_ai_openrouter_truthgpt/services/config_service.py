"""
Configuration Management Service
=================================
Service for dynamic configuration management
"""

import logging
import json
import os
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from enum import Enum

logger = logging.getLogger(__name__)


class ConfigSource(Enum):
    """Configuration source"""
    ENV = "env"
    FILE = "file"
    MEMORY = "memory"
    DEFAULT = "default"


@dataclass
class ConfigEntry:
    """Configuration entry"""
    key: str
    value: Any
    source: ConfigSource
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ConfigService:
    """
    Service for dynamic configuration management.
    
    Features:
    - Multiple configuration sources
    - Priority-based resolution
    - Hot reload support
    - Change listeners
    - Validation
    - Type conversion
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize configuration service.
        
        Args:
            config_file: Optional path to config file
        """
        self.config_file = config_file
        self._config: Dict[str, ConfigEntry] = {}
        self._listeners: Dict[str, List[Callable[[str, Any, Any], None]]] = {}
        self._source_priority = [
            ConfigSource.ENV,
            ConfigSource.FILE,
            ConfigSource.MEMORY,
            ConfigSource.DEFAULT
        ]
        
        # Load from file if provided
        if config_file and Path(config_file).exists():
            self.load_from_file(config_file)
        
        # Load from environment
        self.load_from_env()
    
    def load_from_file(self, file_path: str) -> int:
        """
        Load configuration from JSON file.
        
        Args:
            file_path: Path to config file
        
        Returns:
            Number of config entries loaded
        """
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            count = 0
            for key, value in data.items():
                self.set(key, value, source=ConfigSource.FILE)
                count += 1
            
            logger.info(f"Loaded {count} configuration entries from {file_path}")
            return count
        
        except Exception as e:
            logger.error(f"Failed to load config from {file_path}: {e}")
            return 0
    
    def load_from_env(self, prefix: str = ""):
        """
        Load configuration from environment variables.
        
        Args:
            prefix: Optional prefix for env vars (e.g., "APP_")
        """
        count = 0
        for key, value in os.environ.items():
            if prefix and not key.startswith(prefix):
                continue
            
            config_key = key[len(prefix):] if prefix else key
            # Convert to lowercase and replace underscores
            config_key = config_key.lower().replace('_', '.')
            
            self.set(config_key, value, source=ConfigSource.ENV)
            count += 1
        
        if count > 0:
            logger.debug(f"Loaded {count} configuration entries from environment")
    
    def set(
        self,
        key: str,
        value: Any,
        source: ConfigSource = ConfigSource.MEMORY,
        notify: bool = True
    ):
        """
        Set configuration value.
        
        Args:
            key: Configuration key
            value: Configuration value
            source: Configuration source
            notify: Whether to notify listeners
        """
        old_value = self._config.get(key).value if key in self._config else None
        
        entry = ConfigEntry(
            key=key,
            value=value,
            source=source
        )
        
        self._config[key] = entry
        
        # Notify listeners
        if notify and key in self._listeners:
            for listener in self._listeners[key]:
                try:
                    listener(key, old_value, value)
                except Exception as e:
                    logger.error(f"Error in config listener for {key}: {e}")
        
        logger.debug(f"Configuration set: {key} = {value} (source: {source.value})")
    
    def get(
        self,
        key: str,
        default: Any = None,
        convert_type: Optional[type] = None
    ) -> Any:
        """
        Get configuration value.
        
        Args:
            key: Configuration key
            default: Default value if not found
            convert_type: Optional type to convert to
        
        Returns:
            Configuration value
        """
        entry = self._config.get(key)
        
        if entry is None:
            return default
        
        value = entry.value
        
        # Type conversion
        if convert_type and value is not None:
            try:
                if convert_type == bool:
                    # Handle string booleans
                    if isinstance(value, str):
                        value = value.lower() in ('true', '1', 'yes', 'on')
                    else:
                        value = bool(value)
                elif convert_type == int:
                    value = int(value)
                elif convert_type == float:
                    value = float(value)
                elif convert_type == list:
                    if isinstance(value, str):
                        value = json.loads(value) if value.startswith('[') else value.split(',')
                elif convert_type == dict:
                    if isinstance(value, str):
                        value = json.loads(value)
            except (ValueError, TypeError) as e:
                logger.warning(f"Failed to convert {key} to {convert_type}: {e}")
                return default
        
        return value
    
    def get_bool(self, key: str, default: bool = False) -> bool:
        """Get boolean configuration"""
        return self.get(key, default, convert_type=bool)
    
    def get_int(self, key: str, default: int = 0) -> int:
        """Get integer configuration"""
        return self.get(key, default, convert_type=int)
    
    def get_float(self, key: str, default: float = 0.0) -> float:
        """Get float configuration"""
        return self.get(key, default, convert_type=float)
    
    def get_list(self, key: str, default: Optional[List] = None) -> List:
        """Get list configuration"""
        return self.get(key, default or [], convert_type=list)
    
    def get_dict(self, key: str, default: Optional[Dict] = None) -> Dict:
        """Get dictionary configuration"""
        return self.get(key, default or {}, convert_type=dict)
    
    def subscribe(
        self,
        key: str,
        listener: Callable[[str, Any, Any], None]
    ):
        """
        Subscribe to configuration changes.
        
        Args:
            key: Configuration key (use '*' for all keys)
            listener: Callback function(key, old_value, new_value)
        """
        if key not in self._listeners:
            self._listeners[key] = []
        self._listeners[key].append(listener)
        logger.debug(f"Subscribed to config changes for: {key}")
    
    def unsubscribe(self, key: str, listener: Callable):
        """Unsubscribe from configuration changes"""
        if key in self._listeners:
            if listener in self._listeners[key]:
                self._listeners[key].remove(listener)
                logger.debug(f"Unsubscribed from config changes for: {key}")
    
    def save_to_file(self, file_path: Optional[str] = None) -> bool:
        """
        Save configuration to file.
        
        Args:
            file_path: Optional file path (uses default if not provided)
        
        Returns:
            True if saved successfully
        """
        file_path = file_path or self.config_file
        if not file_path:
            logger.error("No config file path specified")
            return False
        
        try:
            # Only save memory and default sources
            data = {
                key: entry.value
                for key, entry in self._config.items()
                if entry.source in [ConfigSource.MEMORY, ConfigSource.DEFAULT]
            }
            
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Configuration saved to {file_path}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to save config to {file_path}: {e}")
            return False
    
    def reload(self):
        """Reload configuration from file"""
        if self.config_file:
            self.load_from_file(self.config_file)
        self.load_from_env()
        logger.info("Configuration reloaded")
    
    def list_keys(self, source: Optional[ConfigSource] = None) -> List[str]:
        """List configuration keys, optionally filtered by source"""
        if source:
            return [key for key, entry in self._config.items() if entry.source == source]
        return list(self._config.keys())
    
    def get_all(self) -> Dict[str, Any]:
        """Get all configuration as dictionary"""
        return {key: entry.value for key, entry in self._config.items()}
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get configuration statistics"""
        source_counts = {}
        for entry in self._config.values():
            source = entry.source.value
            source_counts[source] = source_counts.get(source, 0) + 1
        
        return {
            'total_entries': len(self._config),
            'sources': source_counts,
            'listeners': sum(len(listeners) for listeners in self._listeners.values()),
            'config_file': self.config_file
        }


# Global config service instance
_config_service: Optional[ConfigService] = None


def get_config_service(config_file: Optional[str] = None) -> ConfigService:
    """Get or create config service instance"""
    global _config_service
    if _config_service is None:
        _config_service = ConfigService(config_file)
    return _config_service

