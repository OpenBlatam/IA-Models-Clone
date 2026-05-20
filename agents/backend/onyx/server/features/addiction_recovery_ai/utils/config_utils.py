"""
Configuration utilities
Configuration management functions
"""

from typing import Any, Optional, Dict, List
import os
import json
from pathlib import Path


class Config:
    """
    Configuration manager
    """
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """
        Initialize config
        
        Args:
            config_dict: Optional initial config dictionary
        """
        self._config = config_dict or {}
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get config value
        
        Args:
            key: Config key (supports dot notation)
            default: Default value
        
        Returns:
            Config value
        """
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """
        Set config value
        
        Args:
            key: Config key (supports dot notation)
            value: Config value
        """
        keys = key.split('.')
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def has(self, key: str) -> bool:
        """
        Check if config key exists
        
        Args:
            key: Config key (supports dot notation)
        
        Returns:
            True if key exists
        """
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return False
        
        return True
    
    def remove(self, key: str) -> bool:
        """
        Remove config key
        
        Args:
            key: Config key (supports dot notation)
        
        Returns:
            True if removed
        """
        keys = key.split('.')
        config = self._config
        
        for k in keys[:-1]:
            if isinstance(config, dict) and k in config:
                config = config[k]
            else:
                return False
        
        if isinstance(config, dict) and keys[-1] in config:
            del config[keys[-1]]
            return True
        
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Get config as dictionary
        
        Returns:
            Config dictionary
        """
        return self._config.copy()
    
    def update(self, config_dict: Dict[str, Any]) -> None:
        """
        Update config with dictionary
        
        Args:
            config_dict: Config dictionary
        """
        self._config.update(config_dict)
    
    def clear(self) -> None:
        """
        Clear all config
        """
        self._config = {}


def load_config_from_file(file_path: str) -> Config:
    """
    Load config from JSON file
    
    Args:
        file_path: Path to config file
    
    Returns:
        Config instance
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        config_dict = json.load(f)
    
    return Config(config_dict)


def save_config_to_file(config: Config, file_path: str) -> None:
    """
    Save config to JSON file
    
    Args:
        config: Config instance
        file_path: Path to config file
    """
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(config.to_dict(), f, indent=2, ensure_ascii=False)


def load_config_from_env(prefix: str = "") -> Config:
    """
    Load config from environment variables
    
    Args:
        prefix: Optional prefix for env vars
    
    Returns:
        Config instance
    """
    config_dict = {}
    
    for key, value in os.environ.items():
        if prefix and not key.startswith(prefix):
            continue
        
        config_key = key[len(prefix):] if prefix else key
        config_key = config_key.lower().replace('_', '.')
        
        # Try to parse as JSON
        try:
            config_dict[config_key] = json.loads(value)
        except (json.JSONDecodeError, ValueError):
            # Use as string
            config_dict[config_key] = value
    
    return Config(config_dict)


def merge_configs(*configs: Config) -> Config:
    """
    Merge multiple configs
    
    Args:
        *configs: Config instances
    
    Returns:
        Merged config
    """
    merged = Config()
    
    for config in configs:
        merged.update(config.to_dict())
    
    return merged


def get_env_var(key: str, default: Any = None, required: bool = False) -> Any:
    """
    Get environment variable
    
    Args:
        key: Environment variable key
        default: Default value
        required: Raise error if not found
    
    Returns:
        Environment variable value
    """
    value = os.getenv(key, default)
    
    if required and value is None:
        raise ValueError(f"Required environment variable {key} not set")
    
    return value


def get_env_bool(key: str, default: bool = False) -> bool:
    """
    Get boolean environment variable
    
    Args:
        key: Environment variable key
        default: Default value
    
    Returns:
        Boolean value
    """
    value = os.getenv(key, str(default)).lower()
    return value in ('true', '1', 'yes', 'on')


def get_env_int(key: str, default: int = 0) -> int:
    """
    Get integer environment variable
    
    Args:
        key: Environment variable key
        default: Default value
    
    Returns:
        Integer value
    """
    try:
        return int(os.getenv(key, str(default)))
    except ValueError:
        return default


def get_env_float(key: str, default: float = 0.0) -> float:
    """
    Get float environment variable
    
    Args:
        key: Environment variable key
        default: Default value
    
    Returns:
        Float value
    """
    try:
        return float(os.getenv(key, str(default)))
    except ValueError:
        return default

