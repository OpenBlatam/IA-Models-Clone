"""
Config Manager
==============

Advanced configuration management.
"""

import logging
import os
from typing import Dict, Any, Optional, List
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class ConfigSource(Enum):
    """Configuration sources."""
    ENV = "env"
    FILE = "file"
    DATABASE = "database"
    SECRETS = "secrets"


class ConfigManager:
    """Configuration manager."""
    
    def __init__(self):
        self._config: Dict[str, Any] = {}
        self._sources: List[ConfigSource] = []
        self._watchers: Dict[str, List[callable]] = {}
    
    def load_from_env(self, prefix: str = ""):
        """Load configuration from environment variables."""
        for key, value in os.environ.items():
            if prefix and not key.startswith(prefix):
                continue
            
            config_key = key[len(prefix):] if prefix else key
            self._config[config_key] = self._parse_value(value)
        
        self._sources.append(ConfigSource.ENV)
        logger.info("Loaded configuration from environment")
    
    def load_from_file(self, file_path: str, format: str = "json"):
        """Load configuration from file."""
        path = Path(file_path)
        if not path.exists():
            logger.warning(f"Config file not found: {file_path}")
            return
        
        if format == "json":
            import json
            with open(path) as f:
                self._config.update(json.load(f))
        elif format == "yaml":
            import yaml
            with open(path) as f:
                self._config.update(yaml.safe_load(f))
        
        self._sources.append(ConfigSource.FILE)
        logger.info(f"Loaded configuration from file: {file_path}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self._config.get(key, default)
    
    def set(self, key: str, value: Any):
        """Set configuration value."""
        old_value = self._config.get(key)
        self._config[key] = value
        
        # Notify watchers
        if key in self._watchers:
            for watcher in self._watchers[key]:
                watcher(key, old_value, value)
    
    def watch(self, key: str, callback: callable):
        """Watch configuration key for changes."""
        if key not in self._watchers:
            self._watchers[key] = []
        self._watchers[key].append(callback)
    
    def get_all(self) -> Dict[str, Any]:
        """Get all configuration."""
        return self._config.copy()
    
    def _parse_value(self, value: str) -> Any:
        """Parse string value to appropriate type."""
        # Try boolean
        if value.lower() in ("true", "false"):
            return value.lower() == "true"
        
        # Try integer
        try:
            return int(value)
        except ValueError:
            pass
        
        # Try float
        try:
            return float(value)
        except ValueError:
            pass
        
        # Return string
        return value















