"""Dynamic configuration with hot-reload"""
from typing import Dict, Any, Optional, Callable
from pathlib import Path
import json
import yaml
import logging
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import threading

logger = logging.getLogger(__name__)


class ConfigWatcher(FileSystemEventHandler):
    """Watch for config file changes"""
    
    def __init__(self, callback: Callable):
        """
        Initialize watcher
        
        Args:
            callback: Callback function when config changes
        """
        self.callback = callback
    
    def on_modified(self, event):
        """Handle file modification"""
        if not event.is_directory:
            logger.info(f"Config file changed: {event.src_path}")
            self.callback()


class DynamicConfig:
    """Dynamic configuration manager with hot-reload"""
    
    def __init__(self, config_file: str):
        """
        Initialize dynamic config
        
        Args:
            config_file: Path to config file
        """
        self.config_file = Path(config_file)
        self.config: Dict[str, Any] = {}
        self.callbacks: list[Callable] = []
        self.observer: Optional[Observer] = None
        self._load_config()
        self._start_watcher()
    
    def _load_config(self):
        """Load configuration from file"""
        try:
            if not self.config_file.exists():
                logger.warning(f"Config file not found: {self.config_file}")
                self.config = {}
                return
            
            if self.config_file.suffix == '.json':
                with open(self.config_file, 'r') as f:
                    self.config = json.load(f)
            elif self.config_file.suffix in ['.yaml', '.yml']:
                with open(self.config_file, 'r') as f:
                    self.config = yaml.safe_load(f)
            else:
                logger.error(f"Unsupported config file format: {self.config_file.suffix}")
                self.config = {}
            
            logger.info(f"Config loaded from {self.config_file}")
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            self.config = {}
    
    def _start_watcher(self):
        """Start file watcher"""
        try:
            self.observer = Observer()
            event_handler = ConfigWatcher(self._on_config_change)
            self.observer.schedule(
                event_handler,
                str(self.config_file.parent),
                recursive=False
            )
            self.observer.start()
            logger.info("Config watcher started")
        except Exception as e:
            logger.error(f"Error starting config watcher: {e}")
    
    def _on_config_change(self):
        """Handle config file change"""
        old_config = self.config.copy()
        self._load_config()
        
        # Notify callbacks
        for callback in self.callbacks:
            try:
                callback(old_config, self.config)
            except Exception as e:
                logger.error(f"Error in config change callback: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get config value
        
        Args:
            key: Config key (supports dot notation)
            default: Default value
            
        Returns:
            Config value
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
    
    def set(self, key: str, value: Any):
        """
        Set config value
        
        Args:
            key: Config key (supports dot notation)
            value: Config value
        """
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
        self._save_config()
    
    def _save_config(self):
        """Save configuration to file"""
        try:
            if self.config_file.suffix == '.json':
                with open(self.config_file, 'w') as f:
                    json.dump(self.config, f, indent=2)
            elif self.config_file.suffix in ['.yaml', '.yml']:
                with open(self.config_file, 'w') as f:
                    yaml.dump(self.config, f, default_flow_style=False)
        except Exception as e:
            logger.error(f"Error saving config: {e}")
    
    def register_callback(self, callback: Callable):
        """
        Register callback for config changes
        
        Args:
            callback: Callback function(old_config, new_config)
        """
        self.callbacks.append(callback)
    
    def stop(self):
        """Stop config watcher"""
        if self.observer:
            self.observer.stop()
            self.observer.join()


# Global dynamic config
_dynamic_config: Optional[DynamicConfig] = None


def get_dynamic_config() -> DynamicConfig:
    """Get global dynamic config"""
    global _dynamic_config
    if _dynamic_config is None:
        from config import settings
        config_file = getattr(settings, 'config_file', 'config.json')
        _dynamic_config = DynamicConfig(config_file)
    return _dynamic_config

