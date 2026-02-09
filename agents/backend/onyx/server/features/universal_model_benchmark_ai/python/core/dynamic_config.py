"""
Dynamic Configuration Module - Runtime configuration management.

Provides:
- Runtime configuration updates
- Configuration hot-reload
- Environment-based config
- Configuration validation
"""

import logging
import json
import yaml
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from threading import Lock
import watchdog
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

logger = logging.getLogger(__name__)


@dataclass
class ConfigChange:
    """Configuration change record."""
    key: str
    old_value: Any
    new_value: Any
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class ConfigWatcher(FileSystemEventHandler):
    """File system watcher for config changes."""
    
    def __init__(self, config_manager):
        """Initialize watcher."""
        self.config_manager = config_manager
    
    def on_modified(self, event):
        """Handle file modification."""
        if not event.is_directory and event.src_path.endswith(('.json', '.yaml', '.yml')):
            logger.info(f"Config file changed: {event.src_path}")
            self.config_manager.reload()


class DynamicConfigManager:
    """Dynamic configuration manager."""
    
    def __init__(self, config_path: Optional[str] = None, watch: bool = False):
        """
        Initialize config manager.
        
        Args:
            config_path: Path to config file
            watch: Watch for file changes
        """
        self.config_path = Path(config_path) if config_path else None
        self.config: Dict[str, Any] = {}
        self.defaults: Dict[str, Any] = {}
        self.lock = Lock()
        self.watchers: List[Callable] = []
        self.change_history: List[ConfigChange] = []
        self.observer: Optional[Observer] = None
        
        if self.config_path and self.config_path.exists():
            self.load()
        
        if watch and self.config_path:
            self.start_watching()
    
    def load(self) -> None:
        """Load configuration from file."""
        if not self.config_path or not self.config_path.exists():
            return
        
        with self.lock:
            try:
                if self.config_path.suffix in ['.yaml', '.yml']:
                    with open(self.config_path, 'r') as f:
                        self.config = yaml.safe_load(f) or {}
                else:
                    with open(self.config_path, 'r') as f:
                        self.config = json.load(f)
                
                logger.info(f"Loaded configuration from {self.config_path}")
            except Exception as e:
                logger.error(f"Error loading config: {e}")
    
    def save(self) -> None:
        """Save configuration to file."""
        if not self.config_path:
            return
        
        with self.lock:
            try:
                if self.config_path.suffix in ['.yaml', '.yml']:
                    with open(self.config_path, 'w') as f:
                        yaml.dump(self.config, f)
                else:
                    with open(self.config_path, 'w') as f:
                        json.dump(self.config, f, indent=2)
                
                logger.info(f"Saved configuration to {self.config_path}")
            except Exception as e:
                logger.error(f"Error saving config: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value.
        
        Args:
            key: Configuration key (supports dot notation)
            default: Default value
            
        Returns:
            Configuration value
        """
        with self.lock:
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
    
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value.
        
        Args:
            key: Configuration key (supports dot notation)
            value: Configuration value
        """
        with self.lock:
            old_value = self.get(key)
            
            keys = key.split('.')
            config = self.config
            
            for k in keys[:-1]:
                if k not in config:
                    config[k] = {}
                config = config[k]
            
            config[keys[-1]] = value
            
            # Record change
            change = ConfigChange(key=key, old_value=old_value, new_value=value)
            self.change_history.append(change)
            if len(self.change_history) > 100:
                self.change_history.pop(0)
            
            # Notify watchers
            for watcher in self.watchers:
                try:
                    watcher(key, old_value, value)
                except Exception as e:
                    logger.error(f"Error in config watcher: {e}")
            
            logger.info(f"Updated config: {key} = {value}")
    
    def set_default(self, key: str, value: Any) -> None:
        """
        Set default value.
        
        Args:
            key: Configuration key
            value: Default value
        """
        self.defaults[key] = value
    
    def watch(self, callback: Callable) -> None:
        """
        Watch for configuration changes.
        
        Args:
            callback: Callback function(key, old_value, new_value)
        """
        self.watchers.append(callback)
    
    def reload(self) -> None:
        """Reload configuration from file."""
        old_config = self.config.copy()
        self.load()
        
        # Notify watchers of changes
        for key in set(list(old_config.keys()) + list(self.config.keys())):
            old_value = old_config.get(key)
            new_value = self.config.get(key)
            if old_value != new_value:
                for watcher in self.watchers:
                    try:
                        watcher(key, old_value, new_value)
                    except Exception as e:
                        logger.error(f"Error in config watcher: {e}")
    
    def start_watching(self) -> None:
        """Start watching config file for changes."""
        if not self.config_path:
            return
        
        try:
            self.observer = Observer()
            event_handler = ConfigWatcher(self)
            self.observer.schedule(
                event_handler,
                str(self.config_path.parent),
                recursive=False
            )
            self.observer.start()
            logger.info(f"Started watching config file: {self.config_path}")
        except Exception as e:
            logger.error(f"Error starting file watcher: {e}")
    
    def stop_watching(self) -> None:
        """Stop watching config file."""
        if self.observer:
            self.observer.stop()
            self.observer.join()
            logger.info("Stopped watching config file")
    
    def get_all(self) -> Dict[str, Any]:
        """Get all configuration."""
        with self.lock:
            return self.config.copy()
    
    def get_change_history(self, limit: int = 50) -> List[ConfigChange]:
        """Get configuration change history."""
        return self.change_history[-limit:]












