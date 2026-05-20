"""
Dynamic Configuration for Color Grading AI
===========================================

Dynamic configuration system with hot-reloading and validation.
"""

import logging
import json
import asyncio
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from enum import Enum
import watchfiles

logger = logging.getLogger(__name__)


class ConfigSource(Enum):
    """Configuration sources."""
    FILE = "file"
    ENV = "env"
    DATABASE = "database"
    API = "api"
    MEMORY = "memory"


@dataclass
class ConfigChange:
    """Configuration change event."""
    key: str
    old_value: Any
    new_value: Any
    source: ConfigSource
    timestamp: datetime = field(default_factory=datetime.now)


class DynamicConfig:
    """
    Dynamic configuration system.
    
    Features:
    - Hot-reloading
    - Multiple sources
    - Change notifications
    - Validation
    - Type conversion
    - Defaults
    """
    
    def __init__(self, config_file: Optional[Path] = None):
        """
        Initialize dynamic config.
        
        Args:
            config_file: Optional config file path
        """
        self.config_file = config_file
        self._config: Dict[str, Any] = {}
        self._defaults: Dict[str, Any] = {}
        self._validators: Dict[str, Callable] = {}
        self._watchers: List[Callable] = []
        self._sources: Dict[str, ConfigSource] = {}
        self._change_history: List[ConfigChange] = []
        self._max_history = 1000
        self._watch_task: Optional[asyncio.Task] = None
    
    def set_default(self, key: str, value: Any):
        """
        Set default value.
        
        Args:
            key: Config key
            value: Default value
        """
        self._defaults[key] = value
        if key not in self._config:
            self._config[key] = value
    
    def set(self, key: str, value: Any, source: ConfigSource = ConfigSource.MEMORY):
        """
        Set configuration value.
        
        Args:
            key: Config key
            value: Value
            source: Configuration source
        """
        old_value = self._config.get(key)
        
        # Validate
        if key in self._validators:
            validator = self._validators[key]
            if not validator(value):
                raise ValueError(f"Invalid value for {key}: {value}")
        
        # Set value
        self._config[key] = value
        self._sources[key] = source
        
        # Record change
        change = ConfigChange(
            key=key,
            old_value=old_value,
            new_value=value,
            source=source
        )
        self._change_history.append(change)
        if len(self._change_history) > self._max_history:
            self._change_history = self._change_history[-self._max_history:]
        
        # Notify watchers
        self._notify_watchers(change)
        
        logger.info(f"Config changed: {key} = {value} (source: {source.value})")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value.
        
        Args:
            key: Config key
            default: Default value if not found
            
        Returns:
            Configuration value
        """
        if key in self._config:
            return self._config[key]
        
        if key in self._defaults:
            return self._defaults[key]
        
        return default
    
    def get_bool(self, key: str, default: bool = False) -> bool:
        """Get boolean config value."""
        value = self.get(key, default)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ("true", "1", "yes", "on")
        return bool(value)
    
    def get_int(self, key: str, default: int = 0) -> int:
        """Get integer config value."""
        value = self.get(key, default)
        try:
            return int(value)
        except (ValueError, TypeError):
            return default
    
    def get_float(self, key: str, default: float = 0.0) -> float:
        """Get float config value."""
        value = self.get(key, default)
        try:
            return float(value)
        except (ValueError, TypeError):
            return default
    
    def get_str(self, key: str, default: str = "") -> str:
        """Get string config value."""
        value = self.get(key, default)
        return str(value)
    
    def load_from_file(self, file_path: Path):
        """
        Load configuration from file.
        
        Args:
            file_path: Config file path
        """
        if not file_path.exists():
            logger.warning(f"Config file not found: {file_path}")
            return
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_path.suffix == '.json':
                    data = json.load(f)
                else:
                    # Assume YAML or other format
                    import yaml
                    data = yaml.safe_load(f)
            
            for key, value in data.items():
                self.set(key, value, ConfigSource.FILE)
            
            logger.info(f"Loaded config from {file_path}")
        
        except Exception as e:
            logger.error(f"Error loading config from {file_path}: {e}")
    
    def load_from_env(self, prefix: str = "COLOR_GRADING_"):
        """
        Load configuration from environment variables.
        
        Args:
            prefix: Environment variable prefix
        """
        import os
        
        for key, value in os.environ.items():
            if key.startswith(prefix):
                config_key = key[len(prefix):].lower().replace("_", ".")
                self.set(config_key, value, ConfigSource.ENV)
        
        logger.info("Loaded config from environment")
    
    def register_validator(self, key: str, validator: Callable):
        """
        Register validator for config key.
        
        Args:
            key: Config key
            validator: Validator function
        """
        self._validators[key] = validator
        logger.debug(f"Registered validator for {key}")
    
    def watch(self, callback: Callable):
        """
        Watch for configuration changes.
        
        Args:
            callback: Callback function (change: ConfigChange) -> None
        """
        self._watchers.append(callback)
        logger.debug("Registered config watcher")
    
    async def start_file_watching(self):
        """Start watching config file for changes."""
        if not self.config_file or not self.config_file.exists():
            return
        
        async for changes in watchfiles.awatch(self.config_file.parent):
            for change in changes:
                if change[0] == watchfiles.Change.modified and change[1] == str(self.config_file):
                    logger.info(f"Config file changed: {self.config_file}")
                    self.load_from_file(self.config_file)
    
    def _notify_watchers(self, change: ConfigChange):
        """Notify all watchers of config change."""
        for watcher in self._watchers:
            try:
                watcher(change)
            except Exception as e:
                logger.error(f"Error in config watcher: {e}")
    
    def get_all(self) -> Dict[str, Any]:
        """Get all configuration."""
        return self._config.copy()
    
    def get_change_history(self, key: Optional[str] = None, limit: int = 100) -> List[ConfigChange]:
        """Get configuration change history."""
        changes = self._change_history
        
        if key:
            changes = [c for c in changes if c.key == key]
        
        return changes[-limit:]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get configuration statistics."""
        sources_count = {}
        for source in self._sources.values():
            sources_count[source.value] = sources_count.get(source.value, 0) + 1
        
        return {
            "config_count": len(self._config),
            "defaults_count": len(self._defaults),
            "validators_count": len(self._validators),
            "watchers_count": len(self._watchers),
            "changes_count": len(self._change_history),
            "sources": sources_count,
        }




