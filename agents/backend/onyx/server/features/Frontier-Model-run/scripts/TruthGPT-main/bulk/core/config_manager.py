#!/usr/bin/env python3
"""
Config Manager - Centralized configuration management
Provides flexible configuration loading, validation, and hot-reloading
"""

import os
import json
import yaml
import logging
from typing import Dict, Any, List, Optional, Union, Callable
from dataclasses import dataclass, field, asdict
from pathlib import Path
import threading
import time
from datetime import datetime, timezone
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import hashlib
import base64
from cryptography.fernet import Fernet

@dataclass
class ConfigSchema:
    """Configuration schema definition."""
    fields: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    required_fields: List[str] = field(default_factory=list)
    validation_rules: Dict[str, Callable] = field(default_factory=dict)
    default_values: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ConfigValidationResult:
    """Configuration validation result."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    validated_config: Dict[str, Any] = field(default_factory=dict)

class ConfigFileHandler(FileSystemEventHandler):
    """File system event handler for config file changes."""
    
    def __init__(self, config_manager):
        self.config_manager = config_manager
        self.logger = logging.getLogger(__name__)
    
    def on_modified(self, event):
        """Handle file modification events."""
        if not event.is_directory and event.src_path.endswith(('.yaml', '.yml', '.json')):
            self.logger.info(f"Config file modified: {event.src_path}")
            self.config_manager.reload_config()

class ConfigManager:
    """Centralized configuration manager with advanced features."""
    
    def __init__(self, config_path: str = "config.yaml", enable_hot_reload: bool = True, 
                 enable_encryption: bool = False, encryption_key: Optional[str] = None):
        self.config_path = Path(config_path)
        self.enable_hot_reload = enable_hot_reload
        self.enable_encryption = enable_encryption
        self.logger = logging.getLogger(__name__)
        
        # Configuration state
        self.config = {}
        self.config_schema = ConfigSchema()
        self.config_lock = threading.RLock()
        self.last_modified = None
        self.config_hash = None
        
        # Hot reload setup
        self.observer = None
        self.file_handler = None
        self.hot_reload_callbacks = []
        
        # Encryption setup
        self.encryption_key = encryption_key
        self.cipher = None
        if enable_encryption:
            self._setup_encryption()
        
        # Load initial configuration
        self._load_config()
        
        # Setup hot reload if enabled
        if enable_hot_reload:
            self._setup_hot_reload()
    
    def _setup_encryption(self):
        """Setup encryption for sensitive configuration."""
        try:
            if self.encryption_key:
                key = self.encryption_key.encode()
            else:
                # Generate key from environment or create new one
                key = os.environ.get('CONFIG_ENCRYPTION_KEY', Fernet.generate_key())
            
            self.cipher = Fernet(key)
            self.logger.info("Configuration encryption enabled")
            
        except Exception as e:
            self.logger.error(f"Encryption setup failed: {e}")
            self.enable_encryption = False
    
    def _setup_hot_reload(self):
        """Setup hot reload monitoring."""
        try:
            self.file_handler = ConfigFileHandler(self)
            self.observer = Observer()
            self.observer.schedule(self.file_handler, str(self.config_path.parent), recursive=False)
            self.observer.start()
            
            self.logger.info("Hot reload monitoring enabled")
            
        except Exception as e:
            self.logger.error(f"Hot reload setup failed: {e}")
            self.enable_hot_reload = False
    
    def _load_config(self):
        """Load configuration from file."""
        try:
            with self.config_lock:
                if not self.config_path.exists():
                    self.logger.warning(f"Config file not found: {self.config_path}")
                    self.config = {}
                    return
                
                # Check if file was modified
                current_modified = self.config_path.stat().st_mtime
                if self.last_modified and current_modified <= self.last_modified:
                    return  # No changes
                
                # Load configuration based on file extension
                if self.config_path.suffix in ['.yaml', '.yml']:
                    with open(self.config_path, 'r') as f:
                        raw_config = yaml.safe_load(f)
                elif self.config_path.suffix == '.json':
                    with open(self.config_path, 'r') as f:
                        raw_config = json.load(f)
                else:
                    self.logger.error(f"Unsupported config file format: {self.config_path.suffix}")
                    return
                
                # Decrypt if encryption is enabled
                if self.enable_encryption and self.cipher:
                    raw_config = self._decrypt_config(raw_config)
                
                # Validate configuration
                validation_result = self._validate_config(raw_config)
                if not validation_result.is_valid:
                    self.logger.error(f"Config validation failed: {validation_result.errors}")
                    return
                
                # Update configuration
                self.config = validation_result.validated_config
                self.last_modified = current_modified
                self.config_hash = self._calculate_config_hash()
                
                self.logger.info("Configuration loaded successfully")
                
        except Exception as e:
            self.logger.error(f"Config loading failed: {e}")
    
    def _encrypt_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Encrypt sensitive configuration values."""
        try:
            if not self.cipher:
                return config
            
            encrypted_config = {}
            sensitive_keys = ['api_keys', 'passwords', 'secrets', 'tokens']
            
            for key, value in config.items():
                if any(sensitive_key in key.lower() for sensitive_key in sensitive_keys):
                    if isinstance(value, str):
                        encrypted_value = self.cipher.encrypt(value.encode()).decode()
                        encrypted_config[key] = f"encrypted:{encrypted_value}"
                    else:
                        encrypted_config[key] = value
                else:
                    encrypted_config[key] = value
            
            return encrypted_config
            
        except Exception as e:
            self.logger.error(f"Config encryption failed: {e}")
            return config
    
    def _decrypt_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Decrypt sensitive configuration values."""
        try:
            if not self.cipher:
                return config
            
            decrypted_config = {}
            
            for key, value in config.items():
                if isinstance(value, str) and value.startswith("encrypted:"):
                    try:
                        encrypted_value = value[10:]  # Remove "encrypted:" prefix
                        decrypted_value = self.cipher.decrypt(encrypted_value.encode()).decode()
                        decrypted_config[key] = decrypted_value
                    except Exception as e:
                        self.logger.warning(f"Failed to decrypt {key}: {e}")
                        decrypted_config[key] = value
                else:
                    decrypted_config[key] = value
            
            return decrypted_config
            
        except Exception as e:
            self.logger.error(f"Config decryption failed: {e}")
            return config
    
    def _validate_config(self, config: Dict[str, Any]) -> ConfigValidationResult:
        """Validate configuration against schema."""
        try:
            errors = []
            warnings = []
            validated_config = {}
            
            # Check required fields
            for field in self.config_schema.required_fields:
                if field not in config:
                    errors.append(f"Required field missing: {field}")
                else:
                    validated_config[field] = config[field]
            
            # Validate field types and values
            for field, field_config in self.config_schema.fields.items():
                if field in config:
                    value = config[field]
                    field_type = field_config.get('type', str)
                    
                    # Type validation
                    if not isinstance(value, field_type):
                        errors.append(f"Field {field} has wrong type: expected {field_type}, got {type(value)}")
                        continue
                    
                    # Custom validation
                    if field in self.config_schema.validation_rules:
                        validation_func = self.config_schema.validation_rules[field]
                        if not validation_func(value):
                            errors.append(f"Field {field} failed custom validation")
                            continue
                    
                    validated_config[field] = value
                else:
                    # Use default value if available
                    if field in self.config_schema.default_values:
                        validated_config[field] = self.config_schema.default_values[field]
                    else:
                        warnings.append(f"Field {field} not provided, using default")
            
            # Add any additional fields not in schema
            for key, value in config.items():
                if key not in validated_config:
                    validated_config[key] = value
                    warnings.append(f"Unknown field {key} added to config")
            
            return ConfigValidationResult(
                is_valid=len(errors) == 0,
                errors=errors,
                warnings=warnings,
                validated_config=validated_config
            )
            
        except Exception as e:
            return ConfigValidationResult(
                is_valid=False,
                errors=[f"Validation error: {str(e)}"],
                validated_config={}
            )
    
    def _calculate_config_hash(self) -> str:
        """Calculate hash of current configuration."""
        try:
            config_str = json.dumps(self.config, sort_keys=True)
            return hashlib.md5(config_str.encode()).hexdigest()
        except Exception as e:
            self.logger.error(f"Hash calculation failed: {e}")
            return ""
    
    def reload_config(self):
        """Manually reload configuration."""
        try:
            self._load_config()
            
            # Notify callbacks
            for callback in self.hot_reload_callbacks:
                try:
                    callback(self.config)
                except Exception as e:
                    self.logger.error(f"Hot reload callback failed: {e}")
            
            self.logger.info("Configuration reloaded")
            
        except Exception as e:
            self.logger.error(f"Config reload failed: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key."""
        try:
            with self.config_lock:
                keys = key.split('.')
                value = self.config
                
                for k in keys:
                    if isinstance(value, dict) and k in value:
                        value = value[k]
                    else:
                        return default
                
                return value
                
        except Exception as e:
            self.logger.error(f"Config get failed for key {key}: {e}")
            return default
    
    def set(self, key: str, value: Any, save_to_file: bool = True):
        """Set configuration value."""
        try:
            with self.config_lock:
                keys = key.split('.')
                config = self.config
                
                # Navigate to the parent of the target key
                for k in keys[:-1]:
                    if k not in config:
                        config[k] = {}
                    config = config[k]
                
                # Set the value
                config[keys[-1]] = value
                
                # Save to file if requested
                if save_to_file:
                    self.save_config()
                
        except Exception as e:
            self.logger.error(f"Config set failed for key {key}: {e}")
    
    def save_config(self, filepath: Optional[str] = None):
        """Save configuration to file."""
        try:
            save_path = Path(filepath) if filepath else self.config_path
            
            # Encrypt if encryption is enabled
            config_to_save = self.config
            if self.enable_encryption and self.cipher:
                config_to_save = self._encrypt_config(self.config)
            
            # Save based on file extension
            if save_path.suffix in ['.yaml', '.yml']:
                with open(save_path, 'w') as f:
                    yaml.dump(config_to_save, f, default_flow_style=False, indent=2)
            elif save_path.suffix == '.json':
                with open(save_path, 'w') as f:
                    json.dump(config_to_save, f, indent=2)
            else:
                self.logger.error(f"Unsupported file format: {save_path.suffix}")
                return False
            
            self.logger.info(f"Configuration saved to {save_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Config save failed: {e}")
            return False
    
    def add_hot_reload_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Add callback for hot reload events."""
        self.hot_reload_callbacks.append(callback)
    
    def remove_hot_reload_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Remove hot reload callback."""
        if callback in self.hot_reload_callbacks:
            self.hot_reload_callbacks.remove(callback)
    
    def set_schema(self, schema: ConfigSchema):
        """Set configuration schema."""
        self.config_schema = schema
        self.logger.info("Configuration schema updated")
    
    def validate_current_config(self) -> ConfigValidationResult:
        """Validate current configuration."""
        return self._validate_config(self.config)
    
    def get_all_config(self) -> Dict[str, Any]:
        """Get all configuration."""
        with self.config_lock:
            return self.config.copy()
    
    def export_config(self, filepath: str, format: str = 'yaml') -> bool:
        """Export configuration to file."""
        try:
            export_path = Path(filepath)
            
            if format.lower() == 'yaml':
                with open(export_path, 'w') as f:
                    yaml.dump(self.config, f, default_flow_style=False, indent=2)
            elif format.lower() == 'json':
                with open(export_path, 'w') as f:
                    json.dump(self.config, f, indent=2)
            else:
                self.logger.error(f"Unsupported export format: {format}")
                return False
            
            self.logger.info(f"Configuration exported to {export_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Config export failed: {e}")
            return False
    
    def import_config(self, filepath: str) -> bool:
        """Import configuration from file."""
        try:
            import_path = Path(filepath)
            
            if not import_path.exists():
                self.logger.error(f"Import file not found: {import_path}")
                return False
            
            # Load configuration
            if import_path.suffix in ['.yaml', '.yml']:
                with open(import_path, 'r') as f:
                    imported_config = yaml.safe_load(f)
            elif import_path.suffix == '.json':
                with open(import_path, 'r') as f:
                    imported_config = json.load(f)
            else:
                self.logger.error(f"Unsupported import format: {import_path.suffix}")
                return False
            
            # Validate imported configuration
            validation_result = self._validate_config(imported_config)
            if not validation_result.is_valid:
                self.logger.error(f"Imported config validation failed: {validation_result.errors}")
                return False
            
            # Update configuration
            with self.config_lock:
                self.config = validation_result.validated_config
            
            self.logger.info(f"Configuration imported from {import_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Config import failed: {e}")
            return False
    
    def cleanup(self):
        """Cleanup resources."""
        try:
            if self.observer:
                self.observer.stop()
                self.observer.join()
            
            self.logger.info("Config manager cleaned up")
            
        except Exception as e:
            self.logger.error(f"Config manager cleanup failed: {e}")
    
    def __del__(self):
        """Destructor."""
        self.cleanup()
