"""
Config Base
===========

Base classes and utilities for configuration management.
"""

import os
import json
import yaml
from typing import Dict, Any, Optional, Type, TypeVar
from pathlib import Path
from dataclasses import dataclass, field, asdict
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)

T = TypeVar('T')


class ConfigBase(ABC):
    """Base class for configurations."""
    
    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        """
        Create config from dictionary.
        
        Args:
            data: Configuration dictionary
            
        Returns:
            Config instance
        """
        return cls(**data)
    
    @classmethod
    def from_file(cls: Type[T], file_path: Path) -> T:
        """
        Load config from file.
        
        Args:
            file_path: Configuration file path
            
        Returns:
            Config instance
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Config file not found: {file_path}")
        
        if file_path.suffix in ['.yaml', '.yml']:
            with open(file_path, 'r') as f:
                data = yaml.safe_load(f)
        elif file_path.suffix == '.json':
            with open(file_path, 'r') as f:
                data = json.load(f)
        else:
            raise ValueError(f"Unsupported config file format: {file_path.suffix}")
        
        return cls.from_dict(data)
    
    @classmethod
    def from_env(cls: Type[T], prefix: str = "") -> T:
        """
        Load config from environment variables.
        
        Args:
            prefix: Environment variable prefix
            
        Returns:
            Config instance
        """
        data = {}
        prefix = prefix.upper() + "_" if prefix else ""
        
        # Get all environment variables with prefix
        for key, value in os.environ.items():
            if key.startswith(prefix):
                config_key = key[len(prefix):].lower()
                # Convert to nested dict if needed
                parts = config_key.split('_')
                current = data
                for part in parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                current[parts[-1]] = value
        
        return cls.from_dict(data)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)
    
    def to_file(self, file_path: Path):
        """
        Save config to file.
        
        Args:
            file_path: Output file path
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = self.to_dict()
        
        if file_path.suffix in ['.yaml', '.yml']:
            with open(file_path, 'w') as f:
                yaml.dump(data, f, default_flow_style=False)
        elif file_path.suffix == '.json':
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
        else:
            raise ValueError(f"Unsupported config file format: {file_path.suffix}")
    
    def validate(self) -> bool:
        """
        Validate configuration.
        
        Returns:
            True if valid
        """
        return True
    
    def merge(self, other: 'ConfigBase') -> 'ConfigBase':
        """
        Merge with another config.
        
        Args:
            other: Other config instance
            
        Returns:
            Merged config
        """
        self_dict = self.to_dict()
        other_dict = other.to_dict()
        
        # Deep merge
        merged = self._deep_merge(self_dict, other_dict)
        return self.from_dict(merged)
    
    @staticmethod
    def _deep_merge(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries."""
        result = base.copy()
        for key, value in update.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = ConfigBase._deep_merge(result[key], value)
            else:
                result[key] = value
        return result


class ConfigLoader:
    """Configuration loader with multiple sources."""
    
    @staticmethod
    def load(
        config_class: Type[T],
        file_path: Optional[Path] = None,
        env_prefix: Optional[str] = None,
        default: Optional[Dict[str, Any]] = None
    ) -> T:
        """
        Load configuration from multiple sources.
        
        Priority: file > environment > default
        
        Args:
            config_class: Configuration class
            file_path: Optional config file path
            env_prefix: Optional environment variable prefix
            default: Optional default values
            
        Returns:
            Configuration instance
        """
        config_data = default or {}
        
        # Load from environment
        if env_prefix:
            env_data = {}
            prefix = env_prefix.upper() + "_"
            for key, value in os.environ.items():
                if key.startswith(prefix):
                    config_key = key[len(prefix):].lower()
                    parts = config_key.split('_')
                    current = env_data
                    for part in parts[:-1]:
                        if part not in current:
                            current[part] = {}
                        current = current[part]
                    current[parts[-1]] = value
            
            # Merge environment data
            config_data = ConfigBase._deep_merge(config_data, env_data)
        
        # Load from file (highest priority)
        if file_path and Path(file_path).exists():
            file_data = config_class.from_file(file_path).to_dict()
            config_data = ConfigBase._deep_merge(config_data, file_data)
        
        return config_class.from_dict(config_data)
    
    @staticmethod
    def validate_config(config: ConfigBase) -> tuple[bool, list[str]]:
        """
        Validate configuration.
        
        Args:
            config: Configuration instance
            
        Returns:
            Tuple of (is_valid, errors)
        """
        errors = []
        
        try:
            if not config.validate():
                errors.append("Configuration validation failed")
        except Exception as e:
            errors.append(f"Validation error: {e}")
        
        return len(errors) == 0, errors




