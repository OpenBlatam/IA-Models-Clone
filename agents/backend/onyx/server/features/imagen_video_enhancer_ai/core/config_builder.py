"""
Configuration Builder
=====================

Advanced configuration builder with validation and merging.
"""

import logging
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from pathlib import Path
import json
import yaml

logger = logging.getLogger(__name__)


@dataclass
class ConfigSection:
    """Configuration section."""
    name: str
    config: Dict[str, Any]
    required: bool = True
    validator: Optional[Callable[[Dict[str, Any]], tuple[bool, Optional[str]]]] = None


class ConfigBuilder:
    """Configuration builder."""
    
    def __init__(self):
        """Initialize config builder."""
        self.sections: Dict[str, ConfigSection] = {}
        self.defaults: Dict[str, Any] = {}
        self.validators: Dict[str, Callable[[Dict[str, Any]], tuple[bool, Optional[str]]]] = {}
    
    def add_section(
        self,
        name: str,
        config: Dict[str, Any],
        required: bool = True,
        validator: Optional[Callable[[Dict[str, Any]], tuple[bool, Optional[str]]]] = None
    ):
        """
        Add configuration section.
        
        Args:
            name: Section name
            config: Configuration dictionary
            required: Whether section is required
            validator: Optional validator function
        """
        section = ConfigSection(
            name=name,
            config=config,
            required=required,
            validator=validator
        )
        self.sections[name] = section
        logger.debug(f"Added config section: {name}")
    
    def set_default(self, key: str, value: Any):
        """
        Set default value.
        
        Args:
            key: Configuration key (supports dot notation)
            value: Default value
        """
        self.defaults[key] = value
    
    def register_validator(
        self,
        name: str,
        validator: Callable[[Dict[str, Any]], tuple[bool, Optional[str]]]
    ):
        """
        Register validator function.
        
        Args:
            name: Validator name
            validator: Validator function
        """
        self.validators[name] = validator
        logger.debug(f"Registered validator: {name}")
    
    def build(self) -> Dict[str, Any]:
        """
        Build configuration from sections.
        
        Returns:
            Complete configuration dictionary
        """
        config = {}
        
        # Add sections
        for name, section in self.sections.items():
            if section.required and not section.config:
                raise ValueError(f"Required section {name} is empty")
            
            # Validate if validator exists
            if section.validator:
                is_valid, error = section.validator(section.config)
                if not is_valid:
                    raise ValueError(f"Section {name} validation failed: {error}")
            
            config[name] = section.config
        
        # Apply defaults
        for key, value in self.defaults.items():
            keys = key.split('.')
            current = config
            
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]
            
            if keys[-1] not in current:
                current[keys[-1]] = value
        
        return config
    
    def load_from_file(self, file_path: Path) -> Dict[str, Any]:
        """
        Load configuration from file.
        
        Args:
            file_path: Configuration file path
            
        Returns:
            Configuration dictionary
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Config file not found: {file_path}")
        
        try:
            if file_path.suffix in ['.yaml', '.yml']:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return yaml.safe_load(f) or {}
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            raise ValueError(f"Error loading config from {file_path}: {e}")
    
    def save_to_file(self, config: Dict[str, Any], file_path: Path, format: str = "json"):
        """
        Save configuration to file.
        
        Args:
            config: Configuration dictionary
            file_path: Output file path
            format: File format (json, yaml)
        """
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            if format == "yaml":
                with open(file_path, 'w', encoding='utf-8') as f:
                    yaml.dump(config, f, default_flow_style=False)
            else:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(config, f, indent=2)
            
            logger.info(f"Saved configuration to {file_path}")
        except Exception as e:
            raise ValueError(f"Error saving config to {file_path}: {e}")
    
    def merge(self, *configs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge multiple configurations.
        
        Args:
            *configs: Configuration dictionaries to merge
            
        Returns:
            Merged configuration
        """
        merged = {}
        
        for config in configs:
            merged = self._deep_merge(merged, config)
        
        return merged
    
    def _deep_merge(self, base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries."""
        result = base.copy()
        
        for key, value in update.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result




