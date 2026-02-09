"""
Dynamic Configuration for Piel Mejorador AI SAM3
================================================

Hot-reloadable configuration system.
"""

import asyncio
import logging
import json
from typing import Dict, Any, Optional, Callable
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
    """File system event handler for config file changes."""
    
    def __init__(self, callback: Callable):
        """
        Initialize handler.
        
        Args:
            callback: Callback function when config changes
        """
        self.callback = callback
    
    def on_modified(self, event):
        """Handle file modification."""
        if not event.is_directory:
            self.callback(event.src_path)


class DynamicConfigManager:
    """
    Manages dynamic configuration with hot-reload.
    
    Features:
    - File-based configuration
    - Hot-reload on file changes
    - Change callbacks
    - Configuration validation
    """
    
    def __init__(self, config_file: Optional[Path] = None):
        """
        Initialize dynamic config manager.
        
        Args:
            config_file: Path to configuration file
        """
        self.config_file = config_file or Path("config.json")
        self._config: Dict[str, Any] = {}
        self._callbacks: List[Callable] = []
        self._observer: Optional[Observer] = None
        self._lock = asyncio.Lock()
        
        self._change_history: list = []
    
    def load_config(self) -> Dict[str, Any]:
        """
        Load configuration from file.
        
        Returns:
            Configuration dictionary
        """
        if not self.config_file.exists():
            logger.warning(f"Config file not found: {self.config_file}, using defaults")
            return self._get_default_config()
        
        try:
            with open(self.config_file, "r", encoding="utf-8") as f:
                config = json.load(f)
            
            # Compare with existing
            changes = self._detect_changes(self._config, config)
            self._config = config
            
            # Notify callbacks
            for change in changes:
                for callback in self._callbacks:
                    try:
                        callback(change)
                    except Exception as e:
                        logger.error(f"Error in config change callback: {e}")
            
            if changes:
                logger.info(f"Configuration reloaded: {len(changes)} changes")
            
            return config
            
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return self._config or self._get_default_config()
    
    def _detect_changes(self, old_config: Dict, new_config: Dict) -> List[ConfigChange]:
        """Detect changes between configurations."""
        changes = []
        
        # Check all keys in new config
        for key, new_value in new_config.items():
            old_value = old_config.get(key)
            if old_value != new_value:
                changes.append(ConfigChange(
                    key=key,
                    old_value=old_value,
                    new_value=new_value
                ))
        
        # Check removed keys
        for key in old_config:
            if key not in new_config:
                changes.append(ConfigChange(
                    key=key,
                    old_value=old_config[key],
                    new_value=None
                ))
        
        return changes
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "max_parallel_tasks": 5,
            "rate_limit_rps": 10.0,
            "cache_ttl_hours": 24,
            "memory_max_percent": 80.0,
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value.
        
        Args:
            key: Configuration key (supports dot notation: "section.key")
            default: Default value if not found
            
        Returns:
            Configuration value
        """
        if not self._config:
            self.load_config()
        
        keys = key.split(".")
        value = self._config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any, save: bool = True):
        """
        Set configuration value.
        
        Args:
            key: Configuration key
            value: Value to set
            save: Whether to save to file
        """
        if not self._config:
            self.load_config()
        
        keys = key.split(".")
        config = self._config
        
        # Navigate/create nested structure
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # Set value
        old_value = config.get(keys[-1])
        config[keys[-1]] = value
        
        # Create change event
        change = ConfigChange(
            key=key,
            old_value=old_value,
            new_value=value
        )
        self._change_history.append(change)
        
        # Notify callbacks
        for callback in self._callbacks:
            try:
                callback(change)
            except Exception as e:
                logger.error(f"Error in config change callback: {e}")
        
        # Save to file
        if save:
            self.save_config()
    
    def save_config(self):
        """Save configuration to file."""
        try:
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_file, "w", encoding="utf-8") as f:
                json.dump(self._config, f, indent=2, ensure_ascii=False)
            logger.debug(f"Configuration saved to {self.config_file}")
        except Exception as e:
            logger.error(f"Error saving config: {e}")
    
    def register_callback(self, callback: Callable):
        """
        Register callback for configuration changes.
        
        Args:
            callback: Function to call on changes (change: ConfigChange) -> None
        """
        self._callbacks.append(callback)
    
    def start_watching(self):
        """Start watching config file for changes."""
        if self._observer:
            return
        
        try:
            self._observer = Observer()
            handler = ConfigFileHandler(self._on_config_change)
            self._observer.schedule(handler, str(self.config_file.parent), recursive=False)
            self._observer.start()
            logger.info(f"Started watching config file: {self.config_file}")
        except Exception as e:
            logger.warning(f"Could not start file watcher: {e}")
    
    def stop_watching(self):
        """Stop watching config file."""
        if self._observer:
            self._observer.stop()
            self._observer.join()
            self._observer = None
            logger.info("Stopped watching config file")
    
    def _on_config_change(self, file_path: str):
        """Handle config file change."""
        if Path(file_path) == self.config_file:
            logger.info(f"Config file changed: {file_path}")
            self.load_config()
    
    def get_change_history(self, limit: int = 50) -> List[ConfigChange]:
        """Get configuration change history."""
        return self._change_history[-limit:]




