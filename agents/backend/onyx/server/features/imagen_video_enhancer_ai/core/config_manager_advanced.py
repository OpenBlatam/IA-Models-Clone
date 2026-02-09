"""
Advanced Configuration Manager
==============================

Advanced configuration management with hot-reloading, validation, and multiple sources.
"""

import asyncio
import logging
import json
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from enum import Enum

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False
    Observer = None
    FileSystemEventHandler = None

logger = logging.getLogger(__name__)


class ConfigSource(Enum):
    """Configuration sources."""
    ENV = "env"
    FILE = "file"
    DEFAULT = "default"
    CLI = "cli"


@dataclass
class ConfigChange:
    """Configuration change event."""
    key: str
    old_value: Any
    new_value: Any
    source: ConfigSource
    timestamp: datetime = field(default_factory=datetime.now)


if WATCHDOG_AVAILABLE:
    class ConfigFileHandler(FileSystemEventHandler):
        """File system event handler for config files."""
        
        def __init__(self, config_manager):
            """Initialize file handler."""
            self.config_manager = config_manager
        
        def on_modified(self, event):
            """Handle file modification."""
            if not event.is_directory:
                self.config_manager._reload_file(event.src_path)
else:
    class ConfigFileHandler:
        """Dummy handler when watchdog is not available."""
        pass


class AdvancedConfigManager:
    """Advanced configuration manager with hot-reloading."""
    
    def __init__(self, config_file: Optional[Path] = None):
        """
        Initialize advanced config manager.
        
        Args:
            config_file: Optional configuration file path
        """
        self.config_file = config_file
        self.config: Dict[str, Any] = {}
        self.defaults: Dict[str, Any] = {}
        self.sources: List[ConfigSource] = []
        self.change_handlers: List[Callable] = []
        self.observer: Optional[Observer] = None
        self.watching = False
        
        if config_file:
            self._load_file()
            self._start_watching()
    
    def _load_file(self):
        """Load configuration from file."""
        if not self.config_file or not self.config_file.exists():
            return
        
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                if self.config_file.suffix in ['.yaml', '.yml']:
                    if YAML_AVAILABLE:
                        self.config = yaml.safe_load(f) or {}
                    else:
                        logger.warning("YAML not available, falling back to JSON")
                        self.config = json.load(f)
                else:
                    self.config = json.load(f)
            
            self.sources.append(ConfigSource.FILE)
            logger.info(f"Loaded configuration from {self.config_file}")
        except Exception as e:
            logger.error(f"Error loading config file: {e}")
    
    def _reload_file(self, file_path: str):
        """Reload configuration file."""
        if Path(file_path) == self.config_file:
            old_config = self.config.copy()
            self._load_file()
            self._notify_changes(old_config, self.config)
    
    def _start_watching(self):
        """Start watching config file for changes."""
        if not self.config_file or self.watching or not WATCHDOG_AVAILABLE:
            return
        
        try:
            self.observer = Observer()
            handler = ConfigFileHandler(self)
            self.observer.schedule(handler, str(self.config_file.parent), recursive=False)
            self.observer.start()
            self.watching = True
            logger.info(f"Started watching config file: {self.config_file}")
        except Exception as e:
            logger.warning(f"Could not start file watcher: {e}")
    
    def _stop_watching(self):
        """Stop watching config file."""
        if self.observer:
            self.observer.stop()
            self.observer.join()
            self.watching = False
    
    def set_default(self, key: str, value: Any):
        """
        Set default value.
        
        Args:
            key: Configuration key
            value: Default value
        """
        self.defaults[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value.
        
        Args:
            key: Configuration key (supports dot notation)
            default: Default value
            
        Returns:
            Configuration value
        """
        # Check environment first
        import os
        env_key = key.upper().replace('.', '_')
        if env_key in os.environ:
            return self._parse_env_value(os.environ[env_key])
        
        # Check config dict
        value = self._get_nested(self.config, key)
        if value is not None:
            return value
        
        # Check defaults
        value = self._get_nested(self.defaults, key)
        if value is not None:
            return value
        
        return default
    
    def _get_nested(self, data: Dict[str, Any], key: str) -> Any:
        """Get nested value using dot notation."""
        keys = key.split('.')
        value = data
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return None
        
        return value
    
    def _parse_env_value(self, value: str) -> Any:
        """Parse environment variable value."""
        # Try to parse as JSON
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
        
        return value
    
    def set(self, key: str, value: Any, source: ConfigSource = ConfigSource.DEFAULT):
        """
        Set configuration value.
        
        Args:
            key: Configuration key
            value: Value to set
            source: Configuration source
        """
        old_value = self.get(key)
        
        # Set nested value
        keys = key.split('.')
        current = self.config
        
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        
        current[keys[-1]] = value
        
        # Notify change
        if old_value != value:
            change = ConfigChange(
                key=key,
                old_value=old_value,
                new_value=value,
                source=source
            )
            self._notify_change(change)
    
    def _notify_change(self, change: ConfigChange):
        """Notify configuration change."""
        for handler in self.change_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    asyncio.create_task(handler(change))
                else:
                    handler(change)
            except Exception as e:
                logger.error(f"Config change handler failed: {e}")
    
    def _notify_changes(self, old_config: Dict[str, Any], new_config: Dict[str, Any]):
        """Notify multiple configuration changes."""
        all_keys = set(list(old_config.keys()) + list(new_config.keys()))
        
        for key in all_keys:
            old_value = self._get_nested(old_config, key)
            new_value = self._get_nested(new_config, key)
            
            if old_value != new_value:
                change = ConfigChange(
                    key=key,
                    old_value=old_value,
                    new_value=new_value,
                    source=ConfigSource.FILE
                )
                self._notify_change(change)
    
    def add_change_handler(self, handler: Callable):
        """
        Add configuration change handler.
        
        Args:
            handler: Handler function
        """
        self.change_handlers.append(handler)
    
    def save(self, file_path: Optional[Path] = None):
        """
        Save configuration to file.
        
        Args:
            file_path: Optional file path
        """
        target_file = file_path or self.config_file
        if not target_file:
            raise ValueError("No file path specified")
        
        target_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(target_file, 'w', encoding='utf-8') as f:
            if target_file.suffix in ['.yaml', '.yml']:
                yaml.dump(self.config, f, default_flow_style=False)
            else:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved configuration to {target_file}")
    
    def reload(self):
        """Reload configuration."""
        if self.config_file:
            self._load_file()
    
    def get_all(self) -> Dict[str, Any]:
        """
        Get all configuration.
        
        Returns:
            Configuration dictionary
        """
        return self.config.copy()
    
    def clear(self):
        """Clear configuration."""
        self.config.clear()
    
    def __del__(self):
        """Cleanup on deletion."""
        self._stop_watching()


