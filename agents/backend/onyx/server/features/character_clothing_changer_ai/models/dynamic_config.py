"""
Dynamic Configuration System for Flux2 Clothing Changer
======================================================

Dynamic configuration management with hot-reloading.
"""

import json
import yaml
from typing import Dict, Any, Optional, List, Callable
from pathlib import Path
from dataclasses import dataclass
import logging
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

logger = logging.getLogger(__name__)


@dataclass
class ConfigChange:
    """Configuration change record."""
    key: str
    old_value: Any
    new_value: Any
    timestamp: float
    source: str


class ConfigFileHandler(FileSystemEventHandler):
    """File system event handler for config files."""
    
    def __init__(self, config_manager):
        self.config_manager = config_manager
    
    def on_modified(self, event):
        if not event.is_directory and event.src_path.endswith(('.json', '.yaml', '.yml')):
            logger.info(f"Config file changed: {event.src_path}")
            self.config_manager.reload_config()


class DynamicConfig:
    """Dynamic configuration management system."""
    
    def __init__(
        self,
        config_file: Optional[Path] = None,
        enable_hot_reload: bool = True,
        default_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize dynamic configuration.
        
        Args:
            config_file: Path to configuration file
            enable_hot_reload: Enable hot-reloading on file changes
            default_config: Default configuration
        """
        self.config_file = config_file
        self.enable_hot_reload = enable_hot_reload
        self.config: Dict[str, Any] = default_config or {}
        self.change_history: List[ConfigChange] = []
        self.change_callbacks: Dict[str, List[Callable]] = {}
        
        # Load initial config
        if config_file and config_file.exists():
            self.load_config()
        
        # Setup file watcher
        if enable_hot_reload and config_file:
            self._setup_file_watcher()
    
    def load_config(self) -> None:
        """Load configuration from file."""
        if not self.config_file or not self.config_file.exists():
            return
        
        try:
            if self.config_file.suffix in ['.yaml', '.yml']:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    self.config = yaml.safe_load(f) or {}
            else:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    self.config = json.load(f)
            
            logger.info(f"Configuration loaded from {self.config_file}")
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
    
    def reload_config(self) -> None:
        """Reload configuration and notify callbacks."""
        old_config = self.config.copy()
        self.load_config()
        
        # Detect changes and notify
        self._detect_and_notify_changes(old_config, self.config)
    
    def _detect_and_notify_changes(
        self,
        old_config: Dict[str, Any],
        new_config: Dict[str, Any],
    ) -> None:
        """Detect changes and notify callbacks."""
        all_keys = set(old_config.keys()) | set(new_config.keys())
        
        for key in all_keys:
            old_value = old_config.get(key)
            new_value = new_config.get(key)
            
            if old_value != new_value:
                change = ConfigChange(
                    key=key,
                    old_value=old_value,
                    new_value=new_value,
                    timestamp=time.time(),
                    source="file_reload",
                )
                
                self.change_history.append(change)
                
                # Notify callbacks
                if key in self.change_callbacks:
                    for callback in self.change_callbacks[key]:
                        try:
                            callback(change)
                        except Exception as e:
                            logger.error(f"Config change callback failed: {e}")
    
    def get(
        self,
        key: str,
        default: Any = None,
    ) -> Any:
        """
        Get configuration value.
        
        Args:
            key: Configuration key (supports dot notation)
            default: Default value if not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        
        return value if value is not None else default
    
    def set(
        self,
        key: str,
        value: Any,
        persist: bool = False,
    ) -> None:
        """
        Set configuration value.
        
        Args:
            key: Configuration key (supports dot notation)
            value: Configuration value
            persist: Persist to file
        """
        keys = key.split('.')
        config = self.config
        
        # Navigate/create nested structure
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # Set value
        old_value = config.get(keys[-1])
        config[keys[-1]] = value
        
        # Record change
        change = ConfigChange(
            key=key,
            old_value=old_value,
            new_value=value,
            timestamp=time.time(),
            source="programmatic",
        )
        self.change_history.append(change)
        
        # Notify callbacks
        if key in self.change_callbacks:
            for callback in self.change_callbacks[key]:
                try:
                    callback(change)
                except Exception as e:
                    logger.error(f"Config change callback failed: {e}")
        
        # Persist if requested
        if persist and self.config_file:
            self.save_config()
    
    def register_change_callback(
        self,
        key: str,
        callback: Callable[[ConfigChange], None],
    ) -> None:
        """
        Register callback for configuration changes.
        
        Args:
            key: Configuration key
            callback: Callback function
        """
        if key not in self.change_callbacks:
            self.change_callbacks[key] = []
        self.change_callbacks[key].append(callback)
    
    def save_config(self) -> None:
        """Save configuration to file."""
        if not self.config_file:
            return
        
        try:
            if self.config_file.suffix in ['.yaml', '.yml']:
                with open(self.config_file, 'w', encoding='utf-8') as f:
                    yaml.dump(self.config, f, default_flow_style=False)
            else:
                with open(self.config_file, 'w', encoding='utf-8') as f:
                    json.dump(self.config, f, indent=2)
            
            logger.info(f"Configuration saved to {self.config_file}")
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
    
    def _setup_file_watcher(self) -> None:
        """Setup file system watcher for hot-reload."""
        if not WATCHDOG_AVAILABLE:
            logger.warning("watchdog not available, hot-reload disabled")
            self.observer = None
            return
        
        try:
            observer = Observer()
            handler = ConfigFileHandler(self)
            observer.schedule(handler, str(self.config_file.parent), recursive=False)
            observer.start()
            self.observer = observer
            logger.info(f"File watcher started for {self.config_file}")
        except Exception as e:
            logger.warning(f"Failed to setup file watcher: {e}")
            self.observer = None
    
    def get_all(self) -> Dict[str, Any]:
        """Get all configuration."""
        return self.config.copy()
    
    def get_change_history(
        self,
        key: Optional[str] = None,
        limit: int = 100,
    ) -> List[ConfigChange]:
        """Get configuration change history."""
        history = self.change_history
        
        if key:
            history = [c for c in history if c.key == key]
        
        return history[-limit:]

