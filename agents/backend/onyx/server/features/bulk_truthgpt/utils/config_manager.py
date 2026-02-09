"""
Configuration Manager
=====================

Advanced configuration management with hot reloading and validation.
"""

import asyncio
import logging
import json
from typing import Dict, Any, Optional, Callable
from pathlib import Path
from datetime import datetime
import aiofiles
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

logger = logging.getLogger(__name__)

class ConfigReloadHandler(FileSystemEventHandler):
    """Handler for config file changes."""
    
    def __init__(self, config_manager):
        self.config_manager = config_manager
    
    def on_modified(self, event):
        if not event.is_directory and event.src_path.endswith('.json'):
            asyncio.create_task(self.config_manager.reload_config())

class ConfigurationManager:
    """Advanced configuration manager."""
    
    def __init__(self, config_path: str = "config.json"):
        self.config_path = Path(config_path)
        self.config: Dict[str, Any] = {}
        self.default_config: Dict[str, Any] = {}
        self.validators: Dict[str, Callable] = {}
        self.reload_callbacks: list[Callable] = []
        self.observer = None
    
    async def load_config(self) -> Dict[str, Any]:
        """Load configuration from file."""
        try:
            if self.config_path.exists():
                async with aiofiles.open(self.config_path, 'r') as f:
                    content = await f.read()
                    self.config = json.loads(content)
                    logger.info(f"Configuration loaded from {self.config_path}")
            else:
                # Use default config
                self.config = self.default_config.copy()
                await self.save_config()
                logger.info(f"Created default configuration at {self.config_path}")
            
            # Validate config
            self._validate_config()
            
            return self.config
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            return self.default_config.copy()
    
    async def save_config(self):
        """Save configuration to file."""
        try:
            async with aiofiles.open(self.config_path, 'w') as f:
                await f.write(json.dumps(self.config, indent=2))
            logger.info(f"Configuration saved to {self.config_path}")
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
    
    async def reload_config(self):
        """Reload configuration."""
        logger.info("Reloading configuration...")
        old_config = self.config.copy()
        await self.load_config()
        
        # Notify callbacks
        for callback in self.reload_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(self.config, old_config)
                else:
                    callback(self.config, old_config)
            except Exception as e:
                logger.error(f"Config reload callback error: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """Set configuration value."""
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def register_validator(self, key: str, validator: Callable):
        """Register validator for a config key."""
        self.validators[key] = validator
    
    def register_reload_callback(self, callback: Callable):
        """Register callback for config reload."""
        self.reload_callbacks.append(callback)
    
    def _validate_config(self):
        """Validate configuration."""
        for key, validator in self.validators.items():
            value = self.get(key)
            if value is not None:
                try:
                    if not validator(value):
                        logger.warning(f"Invalid value for config key '{key}': {value}")
                except Exception as e:
                    logger.error(f"Validator error for '{key}': {e}")
    
    def start_watching(self):
        """Start watching config file for changes."""
        if self.observer:
            return
        
        handler = ConfigReloadHandler(self)
        self.observer = Observer()
        self.observer.schedule(handler, str(self.config_path.parent), recursive=False)
        self.observer.start()
        logger.info(f"Started watching config file: {self.config_path}")
    
    def stop_watching(self):
        """Stop watching config file."""
        if self.observer:
            self.observer.stop()
            self.observer.join()
            self.observer = None
            logger.info("Stopped watching config file")

# Global instance
config_manager = ConfigurationManager()
































