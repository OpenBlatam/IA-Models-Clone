"""
Advanced Configuration System
==============================

Advanced configuration management with validation and transformation.
"""

import logging
import os
import json
from typing import Dict, Any, Optional, List, Type, Callable
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class ConfigSource(Enum):
    """Configuration source."""
    ENVIRONMENT = "environment"
    FILE = "file"
    DEFAULT = "default"
    CLI = "cli"


@dataclass
class ConfigField:
    """Configuration field definition."""
    name: str
    field_type: Type
    default: Any = None
    required: bool = False
    validator: Optional[Callable[[Any], bool]] = None
    transformer: Optional[Callable[[Any], Any]] = None
    description: str = ""
    env_var: Optional[str] = None


class AdvancedConfig:
    """Advanced configuration manager."""
    
    def __init__(
        self,
        config_file: Optional[Path] = None,
        env_prefix: str = ""
    ):
        """
        Initialize advanced config.
        
        Args:
            config_file: Optional config file path
            env_prefix: Environment variable prefix
        """
        self.config_file = config_file
        self.env_prefix = env_prefix
        self.fields: Dict[str, ConfigField] = {}
        self.values: Dict[str, Any] = {}
        self.sources: Dict[str, ConfigSource] = {}
        
        if config_file and config_file.exists():
            self._load_from_file()
    
    def register_field(self, field: ConfigField):
        """
        Register a configuration field.
        
        Args:
            field: Configuration field
        """
        self.fields[field.name] = field
        
        # Set default value
        if field.default is not None:
            self.values[field.name] = field.default
            self.sources[field.name] = ConfigSource.DEFAULT
    
    def get(self, name: str, default: Any = None) -> Any:
        """
        Get configuration value.
        
        Args:
            name: Field name
            default: Default value if not found
            
        Returns:
            Configuration value
        """
        # Check if field is registered
        if name in self.fields:
            field = self.fields[name]
            
            # Check environment variable
            if field.env_var:
                env_value = os.getenv(field.env_var)
                if env_value is not None:
                    value = self._transform_value(field, env_value)
                    if self._validate_value(field, value):
                        self.values[name] = value
                        self.sources[name] = ConfigSource.ENVIRONMENT
                        return value
            
            # Check loaded values
            if name in self.values:
                return self.values[name]
            
            # Return field default
            if field.default is not None:
                return field.default
        
        # Return provided default
        return default
    
    def set(self, name: str, value: Any, source: ConfigSource = ConfigSource.CLI):
        """
        Set configuration value.
        
        Args:
            name: Field name
            value: Value to set
            source: Configuration source
        """
        if name in self.fields:
            field = self.fields[name]
            value = self._transform_value(field, value)
            
            if self._validate_value(field, value):
                self.values[name] = value
                self.sources[name] = source
            else:
                raise ValueError(f"Invalid value for field {name}: {value}")
        else:
            # Allow setting unregistered fields
            self.values[name] = value
            self.sources[name] = source
    
    def _transform_value(self, field: ConfigField, value: Any) -> Any:
        """Transform value using field transformer."""
        if field.transformer:
            return field.transformer(value)
        
        # Default type conversion
        if field.field_type == bool:
            if isinstance(value, str):
                return value.lower() in ('true', '1', 'yes', 'on')
            return bool(value)
        elif field.field_type == int:
            return int(value)
        elif field.field_type == float:
            return float(value)
        elif field.field_type == list:
            if isinstance(value, str):
                return [item.strip() for item in value.split(',')]
            return list(value) if value else []
        
        return value
    
    def _validate_value(self, field: ConfigField, value: Any) -> bool:
        """Validate value using field validator."""
        # Type check
        if not isinstance(value, field.field_type):
            # Allow None for optional fields
            if value is None and not field.required:
                return True
            return False
        
        # Custom validator
        if field.validator:
            return field.validator(value)
        
        return True
    
    def _load_from_file(self):
        """Load configuration from file."""
        if not self.config_file or not self.config_file.exists():
            return
        
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for key, value in data.items():
                if key in self.fields:
                    self.set(key, value, ConfigSource.FILE)
                else:
                    # Allow unregistered fields from file
                    self.values[key] = value
                    self.sources[key] = ConfigSource.FILE
            
            logger.info(f"Loaded configuration from {self.config_file}")
        except Exception as e:
            logger.error(f"Error loading config file: {e}")
    
    def save_to_file(self, file_path: Optional[Path] = None):
        """
        Save configuration to file.
        
        Args:
            file_path: Optional file path (uses default if not provided)
        """
        path = file_path or self.config_file
        if not path:
            logger.warning("No config file path specified")
            return
        
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(self.values, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved configuration to {path}")
        except Exception as e:
            logger.error(f"Error saving config file: {e}")
    
    def validate(self) -> tuple[bool, List[str]]:
        """
        Validate all required fields.
        
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        for name, field in self.fields.items():
            if field.required:
                value = self.get(name)
                if value is None:
                    errors.append(f"Required field '{name}' is not set")
                elif not self._validate_value(field, value):
                    errors.append(f"Field '{name}' has invalid value")
        
        return len(errors) == 0, errors
    
    def to_dict(self) -> Dict[str, Any]:
        """Get all configuration as dictionary."""
        result = {}
        for name in self.fields.keys():
            result[name] = self.get(name)
        return result
    
    def get_source(self, name: str) -> Optional[ConfigSource]:
        """Get configuration source for a field."""
        return self.sources.get(name)




