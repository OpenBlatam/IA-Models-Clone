"""
Configuration Manager for Document Analyzer
============================================

Dynamic configuration management with hot reloading and validation.
"""

import asyncio
import logging
import json
import os
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

logger = logging.getLogger(__name__)

@dataclass
class ConfigChange:
    """Configuration change event"""
    key: str
    old_value: Any
    new_value: Any
    timestamp: datetime = field(default_factory=datetime.now)

class ConfigFileHandler(FileSystemEventHandler):
    """Handler for config file changes"""
    
    def __init__(self, config_manager):
        self.config_manager = config_manager
    
    def on_modified(self, event):
        if event.is_directory:
            return
        if event.src_path.endswith('.json') or event.src_path.endswith('.yaml'):
            logger.info(f"Config file changed: {event.src_path}")
            asyncio.create_task(self.config_manager.reload_config())

class ConfigManager:
    """Advanced configuration manager"""
    
    def __init__(self, config_path: Optional[str] = None, watch_files: bool = True):
        self.config_path = config_path or os.getenv("CONFIG_PATH", "config.json")
        self.watch_files = watch_files
        self.config: Dict[str, Any] = {}
        self.defaults: Dict[str, Any] = {}
        self.change_handlers: List[Callable] = []
        self.change_history: List[ConfigChange] = []
        self.observer = None
        
        self._load_config()
        
        if self.watch_files:
            self._start_watching()
        
        logger.info(f"ConfigManager initialized. Config path: {self.config_path}")
    
    def _load_config(self):
        """Load configuration from file"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    self.config = json.load(f)
                logger.info(f"Configuration loaded from {self.config_path}")
            else:
                logger.warning(f"Config file not found: {self.config_path}, using defaults")
                self.config = self.defaults.copy()
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            self.config = self.defaults.copy()
    
    def _start_watching(self):
        """Start watching config files for changes"""
        try:
            self.observer = Observer()
            handler = ConfigFileHandler(self)
            config_dir = os.path.dirname(os.path.abspath(self.config_path)) or "."
            self.observer.schedule(handler, config_dir, recursive=False)
            self.observer.start()
            logger.info(f"Started watching config directory: {config_dir}")
        except Exception as e:
            logger.warning(f"Could not start file watcher: {e}")
    
    async def reload_config(self):
        """Reload configuration"""
        old_config = self.config.copy()
        self._load_config()
        
        # Detect changes
        for key in set(list(old_config.keys()) + list(self.config.keys())):
            old_value = old_config.get(key)
            new_value = self.config.get(key)
            
            if old_value != new_value:
                change = ConfigChange(key=key, old_value=old_value, new_value=new_value)
                self.change_history.append(change)
                
                # Notify handlers
                for handler in self.change_handlers:
                    try:
                        if asyncio.iscoroutinefunction(handler):
                            await handler(change)
                        else:
                            handler(change)
                    except Exception as e:
                        logger.error(f"Error in config change handler: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
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
        """Set configuration value"""
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        old_value = config.get(keys[-1])
        config[keys[-1]] = value
        
        # Track change
        if old_value != value:
            change = ConfigChange(key=key, old_value=old_value, new_value=value)
            self.change_history.append(change)
            
            # Notify handlers
            for handler in self.change_handlers:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        asyncio.create_task(handler(change))
                    else:
                        handler(change)
                except Exception as e:
                    logger.error(f"Error in config change handler: {e}")
    
    def on_change(self, handler: Callable):
        """Register handler for config changes"""
        self.change_handlers.append(handler)
    
    def save_config(self):
        """Save configuration to file"""
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            logger.info(f"Configuration saved to {self.config_path}")
        except Exception as e:
            logger.error(f"Error saving config: {e}")
    
    def get_all(self) -> Dict[str, Any]:
        """Get all configuration"""
        return self.config.copy()
    
    def reset_to_defaults(self):
        """Reset configuration to defaults"""
        self.config = self.defaults.copy()
        logger.info("Configuration reset to defaults")

# Global instance
config_manager = ConfigManager()
















