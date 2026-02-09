"""
Configuration Manager
=====================

Advanced configuration management with validation and hot-reloading.
"""

import yaml
import json
import os
import logging
from typing import Dict, Any, Optional, Union, List, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False
    logger.warning("watchdog not available. Install with: pip install watchdog")

logger = logging.getLogger(__name__)


@dataclass
class ConfigSection:
    """Configuration section."""
    name: str
    data: Dict[str, Any]
    required: bool = True


class ConfigFileHandler(FileSystemEventHandler):
    """Handler for config file changes."""
    
    def __init__(self, config_manager):
        """Initialize handler."""
        self.config_manager = config_manager
    
    def on_modified(self, event):
        """Handle file modification."""
        if event.src_path == str(self.config_manager.config_path):
            logger.info("Config file modified, reloading...")
            self.config_manager.reload()


class ConfigManager:
    """
    Advanced configuration manager.
    
    Features:
    - Multiple format support (YAML, JSON)
    - Environment variable override
    - Hot-reloading
    - Validation
    - Type conversion
    - Default values
    """
    
    def __init__(
        self,
        config_path: Union[str, Path],
        watch_for_changes: bool = False,
        env_prefix: Optional[str] = None
    ):
        """
        Initialize config manager.
        
        Args:
            config_path: Path to config file
            watch_for_changes: Watch for file changes
            env_prefix: Environment variable prefix
        """
        self.config_path = Path(config_path)
        self.watch_for_changes = watch_for_changes
        self.env_prefix = env_prefix
        self.config: Dict[str, Any] = {}
        self.observer: Optional[Observer] = None
        self._logger = logger
        
        # Load initial config
        self.load()
        
        # Start watching if enabled
        if watch_for_changes:
            self._start_watching()
    
    def load(self) -> Dict[str, Any]:
        """
        Load configuration from file.
        
        Returns:
            Configuration dictionary
        """
        if not self.config_path.exists():
            self._logger.warning(f"Config file not found: {self.config_path}")
            self.config = {}
            return self.config
        
        # Detect format from extension
        ext = self.config_path.suffix.lower()
        
        if ext in ['.yaml', '.yml']:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f) or {}
        elif ext == '.json':
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {ext}")
        
        # Apply environment variable overrides
        self._apply_env_overrides()
        
        self._logger.info(f"Configuration loaded from {self.config_path}")
        return self.config
    
    def reload(self) -> Dict[str, Any]:
        """Reload configuration."""
        return self.load()
    
    def _apply_env_overrides(self) -> None:
        """Apply environment variable overrides."""
        if not self.env_prefix:
            return
        
        prefix = f"{self.env_prefix}_"
        
        for key, value in os.environ.items():
            if key.startswith(prefix):
                # Remove prefix and convert to nested key
                config_key = key[len(prefix):].lower()
                
                # Convert to nested structure (e.g., DB_HOST -> db.host)
                keys = config_key.split('_')
                self._set_nested(self.config, keys, value)
    
    def _set_nested(self, d: Dict, keys: List[str], value: Any) -> None:
        """Set nested dictionary value."""
        for key in keys[:-1]:
            d = d.setdefault(key, {})
        d[keys[-1]] = self._convert_type(value)
    
    def _convert_type(self, value: str) -> Any:
        """Convert string value to appropriate type."""
        # Try boolean
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        
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
        
        # Return as string
        return value
    
    def get(
        self,
        key: str,
        default: Any = None,
        required: bool = False
    ) -> Any:
        """
        Get configuration value.
        
        Args:
            key: Configuration key (supports dot notation)
            default: Default value
            required: Whether key is required
        
        Returns:
            Configuration value
        
        Raises:
            KeyError: If required key is missing
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    if required:
                        raise KeyError(f"Required config key not found: {key}")
                    return default
            else:
                if required:
                    raise KeyError(f"Config key path invalid: {key}")
                return default
        
        return value if value is not None else default
    
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value.
        
        Args:
            key: Configuration key (supports dot notation)
            value: Value to set
        """
        keys = key.split('.')
        d = self.config
        
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        
        d[keys[-1]] = value
    
    def save(self, filepath: Optional[Union[str, Path]] = None) -> None:
        """
        Save configuration to file.
        
        Args:
            filepath: Path to save (None = use original path)
        """
        save_path = Path(filepath) if filepath else self.config_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        ext = save_path.suffix.lower()
        
        if ext in ['.yaml', '.yml']:
            with open(save_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False)
        elif ext == '.json':
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {ext}")
        
        self._logger.info(f"Configuration saved to {save_path}")
    
    def validate(self, schema: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate configuration against schema.
        
        Args:
            schema: Validation schema
        
        Returns:
            (is_valid, errors) tuple
        """
        errors = []
        
        def validate_section(section_name: str, section_schema: Dict, config_data: Dict):
            for key, rules in section_schema.items():
                full_key = f"{section_name}.{key}" if section_name else key
                
                if rules.get('required', False) and key not in config_data:
                    errors.append(f"Required key missing: {full_key}")
                    continue
                
                if key in config_data:
                    value = config_data[key]
                    
                    # Type validation
                    expected_type = rules.get('type')
                    if expected_type and not isinstance(value, expected_type):
                        errors.append(
                            f"Invalid type for {full_key}: "
                            f"expected {expected_type.__name__}, got {type(value).__name__}"
                        )
                    
                    # Range validation
                    if 'min' in rules and value < rules['min']:
                        errors.append(f"{full_key} must be >= {rules['min']}")
                    
                    if 'max' in rules and value > rules['max']:
                        errors.append(f"{full_key} must be <= {rules['max']}")
        
        # Validate top-level
        validate_section("", schema, self.config)
        
        return len(errors) == 0, errors
    
    def _start_watching(self) -> None:
        """Start watching config file for changes."""
        if not WATCHDOG_AVAILABLE:
            self._logger.warning("watchdog not available, cannot watch for changes")
            return
        
        if self.observer:
            return
        
        self.observer = Observer()
        handler = ConfigFileHandler(self)
        self.observer.schedule(handler, str(self.config_path.parent), recursive=False)
        self.observer.start()
        self._logger.info(f"Watching config file: {self.config_path}")
    
    def stop_watching(self) -> None:
        """Stop watching config file."""
        if self.observer:
            self.observer.stop()
            self.observer.join()
            self.observer = None
            self._logger.info("Stopped watching config file")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_watching()

