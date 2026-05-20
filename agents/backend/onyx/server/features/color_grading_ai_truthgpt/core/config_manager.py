"""
Configuration Manager for Color Grading AI
===========================================

Unified configuration management with validation, environment variables, and defaults.
"""

import logging
import os
import json
from typing import Dict, Any, Optional, Type, TypeVar, List
from pathlib import Path
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)

T = TypeVar('T')


class ConfigManager:
    """
    Unified configuration manager.
    
    Features:
    - Environment variable loading
    - File-based configuration
    - Validation
    - Default values
    - Type conversion
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_file: Optional configuration file path
        """
        self.config_file = Path(config_file) if config_file else None
        self._config: Dict[str, Any] = {}
        self._load_config()
    
    def _load_config(self):
        """Load configuration from file and environment."""
        # Load from file if exists
        if self.config_file and self.config_file.exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    self._config = json.load(f)
                logger.info(f"Loaded configuration from {self.config_file}")
            except Exception as e:
                logger.warning(f"Error loading config file: {e}")
        
        # Override with environment variables
        self._load_from_env()
    
    def _load_from_env(self):
        """Load configuration from environment variables."""
        # Common environment variables
        env_mappings = {
            "OPENROUTER_API_KEY": "openrouter.api_key",
            "TRUTHGPT_ENDPOINT": "truthgpt.endpoint",
            "TRUTHGPT_ENABLED": "truthgpt.enabled",
            "FFMPEG_PATH": "video_processing.ffmpeg_path",
            "MAX_PARALLEL_TASKS": "max_parallel_tasks",
            "OUTPUT_DIR": "output_dir",
            "DEBUG": "debug",
            "CACHE_TTL": "cache_ttl",
            "ENABLE_CACHE": "enable_cache",
        }
        
        for env_key, config_path in env_mappings.items():
            value = os.getenv(env_key)
            if value is not None:
                self._set_nested(config_path, self._convert_value(value))
    
    def _set_nested(self, path: str, value: Any):
        """Set nested configuration value."""
        keys = path.split('.')
        config = self._config
        
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        config[keys[-1]] = value
    
    def _get_nested(self, path: str, default: Any = None) -> Any:
        """Get nested configuration value."""
        keys = path.split('.')
        config = self._config
        
        for key in keys:
            if not isinstance(config, dict) or key not in config:
                return default
            config = config[key]
        
        return config
    
    def _convert_value(self, value: str) -> Any:
        """Convert string value to appropriate type."""
        # Boolean
        if value.lower() in ('true', '1', 'yes', 'on'):
            return True
        if value.lower() in ('false', '0', 'no', 'off'):
            return False
        
        # Integer
        try:
            return int(value)
        except ValueError:
            pass
        
        # Float
        try:
            return float(value)
        except ValueError:
            pass
        
        # String
        return value
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value.
        
        Args:
            key: Configuration key (supports dot notation)
            default: Default value
            
        Returns:
            Configuration value
        """
        return self._get_nested(key, default)
    
    def set(self, key: str, value: Any):
        """
        Set configuration value.
        
        Args:
            key: Configuration key (supports dot notation)
            value: Configuration value
        """
        self._set_nested(key, value)
    
    def update(self, config: Dict[str, Any]):
        """
        Update configuration with dictionary.
        
        Args:
            config: Configuration dictionary
        """
        self._config.update(config)
    
    def save(self, file_path: Optional[str] = None):
        """
        Save configuration to file.
        
        Args:
            file_path: Optional file path (uses default if not provided)
        """
        target_file = Path(file_path) if file_path else self.config_file
        if not target_file:
            raise ValueError("No file path provided")
        
        target_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(target_file, 'w', encoding='utf-8') as f:
            json.dump(self._config, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved configuration to {target_file}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Get configuration as dictionary."""
        return self._config.copy()
    
    def validate(self, required_keys: Optional[List[str]] = None) -> bool:
        """
        Validate configuration.
        
        Args:
            required_keys: List of required keys
            
        Returns:
            True if valid
            
        Raises:
            ValueError: If validation fails
        """
        if required_keys:
            missing = []
            for key in required_keys:
                if self._get_nested(key) is None:
                    missing.append(key)
            
            if missing:
                raise ValueError(f"Missing required configuration keys: {missing}")
        
        return True

