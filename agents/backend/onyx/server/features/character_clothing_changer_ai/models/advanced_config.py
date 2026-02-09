"""
Advanced Configuration for Flux2 Clothing Changer
==================================================

Advanced configuration management with validation and hot-reload.
"""

import json
import yaml
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from pathlib import Path
import logging
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

logger = logging.getLogger(__name__)


@dataclass
class ConfigSection:
    """Configuration section."""
    name: str
    data: Dict[str, Any]
    validator: Optional[Callable] = None
    default_values: Dict[str, Any] = field(default_factory=dict)


class AdvancedConfig:
    """Advanced configuration management system."""
    
    def __init__(
        self,
        config_file: Optional[Path] = None,
        enable_hot_reload: bool = True,
    ):
        """
        Initialize advanced config.
        
        Args:
            config_file: Optional configuration file path
            enable_hot_reload: Enable hot-reload on file changes
        """
        self.config_file = config_file
        self.enable_hot_reload = enable_hot_reload
        
        self.sections: Dict[str, ConfigSection] = {}
        self.reload_callbacks: List[Callable] = []
        self.observer: Optional[Observer] = None
        
        if config_file and config_file.exists():
            self.load_from_file(config_file)
        
        if enable_hot_reload and config_file and WATCHDOG_AVAILABLE:
            self._setup_file_watcher()
    
    def load_from_file(self, config_file: Path) -> None:
        """
        Load configuration from file.
        
        Args:
            config_file: Configuration file path
        """
        try:
            if config_file.suffix == ".json":
                with open(config_file, "r") as f:
                    data = json.load(f)
            elif config_file.suffix in [".yaml", ".yml"]:
                with open(config_file, "r") as f:
                    data = yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported config file format: {config_file.suffix}")
            
            # Load sections
            for section_name, section_data in data.items():
                self.set_section(section_name, section_data)
            
            logger.info(f"Loaded configuration from {config_file}")
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
    
    def set_section(
        self,
        name: str,
        data: Dict[str, Any],
        validator: Optional[Callable] = None,
        default_values: Optional[Dict[str, Any]] = None,
    ) -> ConfigSection:
        """
        Set configuration section.
        
        Args:
            name: Section name
            data: Section data
            validator: Optional validator function
            default_values: Optional default values
            
        Returns:
            Created section
        """
        # Merge with defaults
        if default_values:
            merged_data = {**default_values, **data}
        else:
            merged_data = data
        
        # Validate if validator provided
        if validator:
            try:
                validator(merged_data)
            except Exception as e:
                logger.warning(f"Validation warning for {name}: {e}")
        
        section = ConfigSection(
            name=name,
            data=merged_data,
            validator=validator,
            default_values=default_values or {},
        )
        
        self.sections[name] = section
        logger.debug(f"Set configuration section: {name}")
        return section
    
    def get(
        self,
        section: str,
        key: str,
        default: Any = None,
    ) -> Any:
        """
        Get configuration value.
        
        Args:
            section: Section name
            key: Configuration key
            default: Default value
            
        Returns:
            Configuration value
        """
        if section not in self.sections:
            return default
        
        return self.sections[section].data.get(key, default)
    
    def set(
        self,
        section: str,
        key: str,
        value: Any,
    ) -> None:
        """
        Set configuration value.
        
        Args:
            section: Section name
            key: Configuration key
            value: Configuration value
        """
        if section not in self.sections:
            self.sections[section] = ConfigSection(
                name=section,
                data={},
            )
        
        self.sections[section].data[key] = value
        logger.debug(f"Set {section}.{key} = {value}")
    
    def register_reload_callback(self, callback: Callable) -> None:
        """
        Register callback for config reload.
        
        Args:
            callback: Callback function
        """
        self.reload_callbacks.append(callback)
    
    def _setup_file_watcher(self) -> None:
        """Setup file system watcher for hot-reload."""
        if not WATCHDOG_AVAILABLE:
            return
        try:
            class ConfigFileHandler(FileSystemEventHandler):
                def __init__(self, config_manager):
                    self.config_manager = config_manager
                
                def on_modified(self, event):
                    if not event.is_directory and event.src_path == str(self.config_manager.config_file):
                        logger.info("Config file changed, reloading...")
                        self.config_manager.load_from_file(self.config_manager.config_file)
                        for callback in self.config_manager.reload_callbacks:
                            try:
                                callback()
                            except Exception as e:
                                logger.error(f"Error in reload callback: {e}")
            
            self.observer = Observer()
            self.observer.schedule(
                ConfigFileHandler(self),
                str(self.config_file.parent),
                recursive=False,
            )
            self.observer.start()
            logger.info("File watcher started for hot-reload")
        except Exception as e:
            logger.warning(f"Failed to setup file watcher: {e}")
    
    def save_to_file(self, config_file: Optional[Path] = None) -> None:
        """
        Save configuration to file.
        
        Args:
            config_file: Optional file path (uses default if None)
        """
        target_file = config_file or self.config_file
        if not target_file:
            raise ValueError("No config file specified")
        
        data = {
            section.name: section.data
            for section in self.sections.values()
        }
        
        try:
            if target_file.suffix == ".json":
                with open(target_file, "w") as f:
                    json.dump(data, f, indent=2)
            elif target_file.suffix in [".yaml", ".yml"]:
                with open(target_file, "w") as f:
                    yaml.dump(data, f, default_flow_style=False)
            
            logger.info(f"Saved configuration to {target_file}")
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get configuration statistics."""
        return {
            "total_sections": len(self.sections),
            "sections": list(self.sections.keys()),
            "hot_reload_enabled": self.enable_hot_reload,
            "file_watcher_active": (
                WATCHDOG_AVAILABLE and
                self.observer is not None and
                self.observer.is_alive()
            ),
        }

