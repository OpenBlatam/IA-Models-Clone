"""
Dynamic Configuration
=====================

System for dynamic configuration management with hot-reloading.
"""

import logging
import json
import yaml
from typing import Dict, Any, Optional, List, Callable
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

logger = logging.getLogger(__name__)


@dataclass
class ConfigChange:
    """Configuration change event."""
    key: str
    old_value: Any
    new_value: Any
    timestamp: datetime = field(default_factory=datetime.now)


class ConfigFileHandler(FileSystemEventHandler):
    """File system handler for config changes."""
    
    def __init__(self, config_manager: 'DynamicConfigManager'):
        """
        Initialize file handler.
        
        Args:
            config_manager: Config manager instance
        """
        self.config_manager = config_manager
    
    def on_modified(self, event):
        """Handle file modification."""
        if not event.is_directory:
            self.config_manager.reload()


class DynamicConfigManager:
    """Dynamic configuration manager with hot-reloading."""
    
    def __init__(self, config_file: Optional[Path] = None, watch: bool = True):
        """
        Initialize dynamic config manager.
        
        Args:
            config_file: Configuration file path
            watch: Whether to watch for file changes
        """
        self.config_file = config_file
        self.config: Dict[str, Any] = {}
        self.defaults: Dict[str, Any] = {}
        self.watchers: List[Callable[[ConfigChange], None]] = []
        self.observer: Optional[Observer] = None
        self.watch = watch
        
        if config_file:
            self.load(config_file)
            if watch:
                self._start_watching()
    
    def _start_watching(self):
        """Start watching config file for changes."""
        if not self.config_file or not self.config_file.exists():
            return
        
        self.observer = Observer()
        handler = ConfigFileHandler(self)
        self.observer.schedule(handler, str(self.config_file.parent), recursive=False)
        self.observer.start()
        logger.info(f"Watching config file: {self.config_file}")
    
    def _stop_watching(self):
        """Stop watching config file."""
        if self.observer:
            self.observer.stop()
            self.observer.join()
            self.observer = None
    
    def load(self, config_file: Path):
        """
        Load configuration from file.
        
        Args:
            config_file: Configuration file path
        """
        self.config_file = config_file
        
        if not config_file.exists():
            logger.warning(f"Config file not found: {config_file}")
            return
        
        try:
            if config_file.suffix in ['.yaml', '.yml']:
                with open(config_file, 'r', encoding='utf-8') as f:
                    self.config = yaml.safe_load(f) or {}
            else:
                with open(config_file, 'r', encoding='utf-8') as f:
                    self.config = json.load(f)
            
            logger.info(f"Loaded configuration from {config_file}")
        except Exception as e:
            logger.error(f"Error loading config: {e}")
    
    def reload(self):
        """Reload configuration from file."""
        if self.config_file:
            old_config = self.config.copy()
            self.load(self.config_file)
            
            # Notify watchers of changes
            self._notify_changes(old_config, self.config)
    
    def _notify_changes(self, old_config: Dict[str, Any], new_config: Dict[str, Any]):
        """Notify watchers of configuration changes."""
        # Find changed keys
        all_keys = set(old_config.keys()) | set(new_config.keys())
        
        for key in all_keys:
            old_value = old_config.get(key)
            new_value = new_config.get(key)
            
            if old_value != new_value:
                change = ConfigChange(
                    key=key,
                    old_value=old_value,
                    new_value=new_value
                )
                
                for watcher in self.watchers:
                    try:
                        watcher(change)
                    except Exception as e:
                        logger.error(f"Error in config watcher: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value.
        
        Args:
            key: Configuration key (supports dot notation)
            default: Default value
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return self.defaults.get(key, default)
            else:
                return self.defaults.get(key, default)
        
        return value if value is not None else self.defaults.get(key, default)
    
    def set(self, key: str, value: Any):
        """
        Set configuration value.
        
        Args:
            key: Configuration key (supports dot notation)
            value: Configuration value
        """
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        old_value = config.get(keys[-1])
        config[keys[-1]] = value
        
        # Notify watchers
        if old_value != value:
            change = ConfigChange(
                key=key,
                old_value=old_value,
                new_value=value
            )
            
            for watcher in self.watchers:
                try:
                    watcher(change)
                except Exception as e:
                    logger.error(f"Error in config watcher: {e}")
    
    def set_default(self, key: str, value: Any):
        """
        Set default value.
        
        Args:
            key: Configuration key
            value: Default value
        """
        self.defaults[key] = value
    
    def watch_change(self, callback: Callable[[ConfigChange], None]):
        """
        Watch for configuration changes.
        
        Args:
            callback: Callback function
        """
        self.watchers.append(callback)
    
    def save(self, config_file: Optional[Path] = None):
        """
        Save configuration to file.
        
        Args:
            config_file: Optional output file path
        """
        output_file = config_file or self.config_file
        if not output_file:
            raise ValueError("No config file specified")
        
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            if output_file.suffix in ['.yaml', '.yml']:
                with open(output_file, 'w', encoding='utf-8') as f:
                    yaml.dump(self.config, f, default_flow_style=False)
            else:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(self.config, f, indent=2)
            
            logger.info(f"Saved configuration to {output_file}")
        except Exception as e:
            logger.error(f"Error saving config: {e}")
    
    def get_all(self) -> Dict[str, Any]:
        """Get all configuration."""
        return self.config.copy()
    
    def close(self):
        """Close config manager and stop watching."""
        self._stop_watching()




