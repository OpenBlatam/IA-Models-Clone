#!/usr/bin/env python3
"""
Configuration Manager - Ultra-Modular Architecture v3.7
Hierarchical configuration management with validation and hot-reloading
"""
import os
import json
import yaml
import toml
import configparser
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Callable
from datetime import datetime
import threading
import logging
import hashlib
from dataclasses import dataclass, field, asdict
from enum import Enum

logger = logging.getLogger(__name__)

class ConfigFormat(Enum):
    """Supported configuration formats"""
    JSON = "json"
    YAML = "yaml"
    TOML = "toml"
    INI = "ini"
    ENV = "env"

@dataclass
class ConfigSource:
    """Configuration source information"""
    path: str
    format: ConfigFormat
    priority: int = 0
    last_modified: Optional[datetime] = None
    checksum: Optional[str] = None
    enabled: bool = True

@dataclass
class ConfigValidation:
    """Configuration validation result"""
    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    schema_errors: List[str] = field(default_factory=list)

class ConfigManager:
    """
    Manages hierarchical configuration with validation, hot-reloading, and multiple formats
    """
    
    def __init__(self, config_dir: str = "config", default_config: Dict = None):
        """Initialize configuration manager"""
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        
        # Configuration storage
        self._config: Dict[str, Any] = default_config or {}
        self._config_sources: List[ConfigSource] = []
        self._config_schemas: Dict[str, Dict] = {}
        self._config_validators: Dict[str, Callable] = {}
        
        # Configuration watchers
        self._config_watchers: List[Callable] = []
        self._file_watcher = None
        self._watch_enabled = False
        
        # Threading
        self._lock = threading.RLock()
        self._config_hash = None
        
        # Initialize
        self._discover_config_files()
        self._load_configurations()
    
    def _discover_config_files(self):
        """Discover configuration files in the config directory"""
        try:
            logger.info(f"Discovering configuration files in: {self.config_dir}")
            
            # Look for configuration files
            config_patterns = [
                "*.json",
                "*.yaml", "*.yml",
                "*.toml",
                "*.ini", "*.cfg"
            ]
            
            for pattern in config_patterns:
                for config_file in self.config_dir.glob(pattern):
                    self._add_config_source(config_file)
            
            # Sort sources by priority
            self._config_sources.sort(key=lambda x: x.priority, reverse=True)
            
            logger.info(f"Discovered {len(self._config_sources)} configuration sources")
            
        except Exception as e:
            logger.error(f"Error during configuration discovery: {e}")
    
    def _add_config_source(self, config_file: Path):
        """Add a configuration source"""
        try:
            # Determine format
            if config_file.suffix in ['.json']:
                format_type = ConfigFormat.JSON
                priority = 100
            elif config_file.suffix in ['.yaml', '.yml']:
                format_type = ConfigFormat.YAML
                priority = 90
            elif config_file.suffix in ['.toml']:
                format_type = ConfigFormat.TOML
                priority = 80
            elif config_file.suffix in ['.ini', '.cfg']:
                format_type = ConfigFormat.INI
                priority = 70
            else:
                logger.warning(f"Unsupported configuration format: {config_file}")
                return
            
            # Create config source
            source = ConfigSource(
                path=str(config_file),
                format=format_type,
                priority=priority
            )
            
            # Update file information
            self._update_source_info(source)
            
            self._config_sources.append(source)
            logger.info(f"Added configuration source: {config_file.name} ({format_type.value})")
            
        except Exception as e:
            logger.error(f"Error adding configuration source {config_file}: {e}")
    
    def _update_source_info(self, source: ConfigSource):
        """Update source file information"""
        try:
            config_path = Path(source.path)
            if config_path.exists():
                stat = config_path.stat()
                source.last_modified = datetime.fromtimestamp(stat.st_mtime)
                
                # Calculate checksum
                with open(config_path, 'rb') as f:
                    content = f.read()
                    source.checksum = hashlib.md5(content).hexdigest()
                    
        except Exception as e:
            logger.warning(f"Error updating source info for {source.path}: {e}")
    
    def _load_configurations(self):
        """Load all configuration sources"""
        try:
            logger.info("Loading configurations...")
            
            # Load configurations in priority order
            for source in self._config_sources:
                if source.enabled:
                    self._load_config_source(source)
            
            # Merge configurations
            self._merge_configurations()
            
            # Validate final configuration
            self._validate_configuration()
            
            # Update configuration hash
            self._update_config_hash()
            
            logger.info("Configuration loading completed")
            
        except Exception as e:
            logger.error(f"Error loading configurations: {e}")
    
    def _load_config_source(self, source: ConfigSource):
        """Load configuration from a specific source"""
        try:
            logger.debug(f"Loading configuration from: {source.path}")
            
            if source.format == ConfigFormat.JSON:
                config_data = self._load_json_config(source.path)
            elif source.format == ConfigFormat.YAML:
                config_data = self._load_yaml_config(source.path)
            elif source.format == ConfigFormat.TOML:
                config_data = self._load_toml_config(source.path)
            elif source.format == ConfigFormat.INI:
                config_data = self._load_ini_config(source.path)
            else:
                logger.warning(f"Unsupported format: {source.format}")
                return
            
            # Store configuration data
            source_name = Path(source.path).stem
            self._config[source_name] = config_data
            
            logger.debug(f"Loaded configuration from {source.path}")
            
        except Exception as e:
            logger.error(f"Error loading configuration from {source.path}: {e}")
    
    def _load_json_config(self, file_path: str) -> Dict:
        """Load JSON configuration"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _load_yaml_config(self, file_path: str) -> Dict:
        """Load YAML configuration"""
        try:
            import yaml
            with open(file_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except ImportError:
            logger.error("PyYAML not available for YAML configuration")
            return {}
    
    def _load_toml_config(self, file_path: str) -> Dict:
        """Load TOML configuration"""
        try:
            import toml
            with open(file_path, 'r', encoding='utf-8') as f:
                return toml.load(f)
        except ImportError:
            logger.error("TOML not available for TOML configuration")
            return {}
    
    def _load_ini_config(self, file_path: str) -> Dict:
        """Load INI configuration"""
        config = configparser.ConfigParser()
        config.read(file_path)
        
        # Convert to dictionary
        result = {}
        for section in config.sections():
            result[section] = dict(config.items(section))
        
        return result
    
    def _merge_configurations(self):
        """Merge configurations from multiple sources"""
        try:
            logger.debug("Merging configurations...")
            
            # Start with empty configuration
            merged_config = {}
            
            # Merge in priority order (highest priority first)
            for source in sorted(self._config_sources, key=lambda x: x.priority, reverse=True):
                if source.enabled and Path(source.path).stem in self._config:
                    source_name = Path(source.path).stem
                    source_config = self._config[source_name]
                    
                    # Deep merge configuration
                    merged_config = self._deep_merge(merged_config, source_config)
            
            # Update main configuration
            self._config = merged_config
            
            logger.debug("Configuration merging completed")
            
        except Exception as e:
            logger.error(f"Error merging configurations: {e}")
    
    def _deep_merge(self, base: Dict, update: Dict) -> Dict:
        """Deep merge two dictionaries"""
        result = base.copy()
        
        for key, value in update.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _validate_configuration(self):
        """Validate the merged configuration"""
        try:
            logger.debug("Validating configuration...")
            
            # Validate against schemas
            for schema_name, schema in self._config_schemas.items():
                if schema_name in self._config:
                    validation = self._validate_against_schema(
                        self._config[schema_name], schema
                    )
                    
                    if not validation.valid:
                        logger.warning(f"Configuration validation failed for {schema_name}")
                        for error in validation.errors:
                            logger.warning(f"  - {error}")
            
            logger.debug("Configuration validation completed")
            
        except Exception as e:
            logger.error(f"Error validating configuration: {e}")
    
    def _validate_against_schema(self, config: Dict, schema: Dict) -> ConfigValidation:
        """Validate configuration against a schema"""
        validation = ConfigValidation(valid=True)
        
        try:
            # Basic schema validation
            required_fields = schema.get('required', [])
            for field in required_fields:
                if field not in config:
                    validation.valid = False
                    validation.errors.append(f"Required field missing: {field}")
            
            # Type validation
            properties = schema.get('properties', {})
            for field, value in config.items():
                if field in properties:
                    field_schema = properties[field]
                    expected_type = field_schema.get('type')
                    
                    if expected_type == 'string' and not isinstance(value, str):
                        validation.valid = False
                        validation.errors.append(f"Field {field} should be string")
                    elif expected_type == 'integer' and not isinstance(value, int):
                        validation.valid = False
                        validation.errors.append(f"Field {field} should be integer")
                    elif expected_type == 'boolean' and not isinstance(value, bool):
                        validation.valid = False
                        validation.errors.append(f"Field {field} should be boolean")
                    elif expected_type == 'array' and not isinstance(value, list):
                        validation.valid = False
                        validation.errors.append(f"Field {field} should be array")
                    elif expected_type == 'object' and not isinstance(value, dict):
                        validation.valid = False
                        validation.errors.append(f"Field {field} should be object")
            
        except Exception as e:
            validation.valid = False
            validation.errors.append(f"Schema validation error: {e}")
        
        return validation
    
    def _update_config_hash(self):
        """Update configuration hash for change detection"""
        try:
            config_str = json.dumps(self._config, sort_keys=True)
            self._config_hash = hashlib.md5(config_str.encode()).hexdigest()
        except Exception as e:
            logger.warning(f"Error updating config hash: {e}")
    
    def get_config(self, key: str = None, default: Any = None) -> Any:
        """Get configuration value"""
        try:
            if key is None:
                return self._config.copy()
            
            # Navigate nested keys (e.g., "database.host")
            keys = key.split('.')
            value = self._config
            
            for k in keys:
                if isinstance(value, dict) and k in value:
                    value = value[k]
                else:
                    return default
            
            return value
            
        except Exception as e:
            logger.error(f"Error getting configuration {key}: {e}")
            return default
    
    def set_config(self, key: str, value: Any):
        """Set configuration value"""
        try:
            # Navigate nested keys
            keys = key.split('.')
            config = self._config
            
            # Navigate to parent of target key
            for k in keys[:-1]:
                if k not in config:
                    config[k] = {}
                config = config[k]
            
            # Set the value
            config[keys[-1]] = value
            
            # Update configuration hash
            self._update_config_hash()
            
            # Notify watchers
            self._notify_config_changed(key, value)
            
            logger.debug(f"Configuration updated: {key} = {value}")
            
        except Exception as e:
            logger.error(f"Error setting configuration {key}: {e}")
    
    def update_config(self, updates: Dict[str, Any]):
        """Update multiple configuration values"""
        try:
            for key, value in updates.items():
                self.set_config(key, value)
            
            logger.info(f"Updated {len(updates)} configuration values")
            
        except Exception as e:
            logger.error(f"Error updating configurations: {e}")
    
    def add_config_source(self, file_path: str, priority: int = 50) -> bool:
        """Add a new configuration source"""
        try:
            config_path = Path(file_path)
            if not config_path.exists():
                logger.error(f"Configuration file not found: {file_path}")
                return False
            
            # Determine format
            if config_path.suffix in ['.json']:
                format_type = ConfigFormat.JSON
            elif config_path.suffix in ['.yaml', '.yml']:
                format_type = ConfigFormat.YAML
            elif config_path.suffix in ['.toml']:
                format_type = ConfigFormat.TOML
            elif config_path.suffix in ['.ini', '.cfg']:
                format_type = ConfigFormat.INI
            else:
                logger.error(f"Unsupported configuration format: {file_path}")
                return False
            
            # Create and add source
            source = ConfigSource(
                path=file_path,
                format=format_type,
                priority=priority
            )
            
            self._update_source_info(source)
            self._config_sources.append(source)
            
            # Reload configurations
            self._load_configurations()
            
            logger.info(f"Added configuration source: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding configuration source {file_path}: {e}")
            return False
    
    def remove_config_source(self, file_path: str) -> bool:
        """Remove a configuration source"""
        try:
            # Find and remove source
            for i, source in enumerate(self._config_sources):
                if source.path == file_path:
                    removed_source = self._config_sources.pop(i)
                    
                    # Remove from configuration
                    source_name = Path(removed_source.path).stem
                    if source_name in self._config:
                        del self._config[source_name]
                    
                    # Reload configurations
                    self._load_configurations()
                    
                    logger.info(f"Removed configuration source: {file_path}")
                    return True
            
            logger.warning(f"Configuration source not found: {file_path}")
            return False
            
        except Exception as e:
            logger.error(f"Error removing configuration source {file_path}: {e}")
            return False
    
    def add_schema(self, name: str, schema: Dict):
        """Add a configuration schema for validation"""
        try:
            self._config_schemas[name] = schema
            logger.info(f"Added configuration schema: {name}")
            
        except Exception as e:
            logger.error(f"Error adding configuration schema {name}: {e}")
    
    def add_validator(self, name: str, validator: Callable):
        """Add a custom configuration validator"""
        try:
            self._config_validators[name] = validator
            logger.info(f"Added configuration validator: {name}")
            
        except Exception as e:
            logger.error(f"Error adding configuration validator {name}: {e}")
    
    def add_config_watcher(self, callback: Callable):
        """Add configuration change watcher"""
        if callback not in self._config_watchers:
            self._config_watchers.append(callback)
            logger.debug("Added configuration watcher")
    
    def remove_config_watcher(self, callback: Callable):
        """Remove configuration change watcher"""
        if callback in self._config_watchers:
            self._config_watchers.remove(callback)
            logger.debug("Removed configuration watcher")
    
    def _notify_config_changed(self, key: str, value: Any):
        """Notify configuration watchers of changes"""
        for callback in self._config_watchers:
            try:
                callback(key, value, self._config)
            except Exception as e:
                logger.error(f"Error in configuration watcher: {e}")
    
    def start_file_watcher(self):
        """Start configuration file watcher"""
        if self._watch_enabled:
            return
        
        try:
            self._watch_enabled = True
            self._file_watcher = threading.Thread(target=self._watch_config_files, daemon=True)
            self._file_watcher.start()
            logger.info("Configuration file watcher started")
            
        except Exception as e:
            logger.error(f"Error starting configuration file watcher: {e}")
    
    def stop_file_watcher(self):
        """Stop configuration file watcher"""
        self._watch_enabled = False
        if self._file_watcher:
            self._file_watcher.join(timeout=1.0)
        logger.info("Configuration file watcher stopped")
    
    def _watch_config_files(self):
        """Watch for configuration file changes"""
        while self._watch_enabled:
            try:
                # Check for file changes
                config_changed = False
                
                for source in self._config_sources:
                    if source.enabled:
                        old_checksum = source.checksum
                        self._update_source_info(source)
                        
                        if source.checksum != old_checksum:
                            logger.info(f"Configuration file changed: {source.path}")
                            config_changed = True
                
                # Reload if changed
                if config_changed:
                    self._load_configurations()
                
                # Sleep for a while
                import time
                time.sleep(5)
                
            except Exception as e:
                logger.error(f"Error in configuration file watcher: {e}")
                import time
                time.sleep(30)
    
    def export_config(self, format: str = 'json', file_path: str = None) -> str:
        """Export configuration to file or string"""
        try:
            if format.lower() == 'json':
                config_str = json.dumps(self._config, indent=2, default=str)
            elif format.lower() == 'yaml':
                import yaml
                config_str = yaml.dump(self._config, default_flow_style=False)
            elif format.lower() == 'toml':
                import toml
                config_str = toml.dumps(self._config)
            else:
                return f"Unsupported export format: {format}"
            
            # Write to file if specified
            if file_path:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(config_str)
                logger.info(f"Configuration exported to: {file_path}")
            
            return config_str
            
        except Exception as e:
            logger.error(f"Error exporting configuration: {e}")
            return f"Error exporting configuration: {e}"
    
    def create_default_config(self, file_path: str, format: str = 'json'):
        """Create default configuration file"""
        try:
            default_config = {
                "system": {
                    "name": "Ultra-Modular AI System",
                    "version": "3.7.0",
                    "debug": False,
                    "log_level": "INFO"
                },
                "modules": {
                    "enabled": True,
                    "auto_discovery": True,
                    "plugin_support": True
                },
                "plugins": {
                    "enabled": True,
                    "auto_load": False,
                    "hot_reload": True
                },
                "monitoring": {
                    "enabled": True,
                    "interval": 5.0,
                    "metrics_retention": 1000
                }
            }
            
            self.export_config(format, file_path)
            logger.info(f"Default configuration created: {file_path}")
            
        except Exception as e:
            logger.error(f"Error creating default configuration: {e}")
    
    def get_config_info(self) -> Dict[str, Any]:
        """Get configuration information"""
        return {
            'config_sources': [
                {
                    'path': source.path,
                    'format': source.format.value,
                    'priority': source.priority,
                    'enabled': source.enabled,
                    'last_modified': source.last_modified.isoformat() if source.last_modified else None,
                    'checksum': source.checksum
                }
                for source in self._config_sources
            ],
            'schemas': list(self._config_schemas.keys()),
            'validators': list(self._config_validators.keys()),
            'watchers': len(self._config_watchers),
            'file_watcher_enabled': self._watch_enabled,
            'config_hash': self._config_hash,
            'config_size': len(json.dumps(self._config))
        }
    
    def reload_config(self):
        """Reload all configurations"""
        try:
            logger.info("Reloading configurations...")
            self._load_configurations()
            logger.info("Configuration reload completed")
            
        except Exception as e:
            logger.error(f"Error reloading configurations: {e}")
    
    def shutdown(self):
        """Shutdown configuration manager"""
        try:
            logger.info("Shutting down configuration manager...")
            
            # Stop file watcher
            self.stop_file_watcher()
            
            # Clear watchers
            self._config_watchers.clear()
            
            logger.info("Configuration manager shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.shutdown()
