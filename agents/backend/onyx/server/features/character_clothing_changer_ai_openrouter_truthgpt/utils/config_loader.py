"""
Configuration Loader
===================

Utilities for loading and managing configuration.
"""

import os
import json
import logging
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


def load_env_file(filepath: str = ".env") -> Dict[str, str]:
    """
    Load environment variables from .env file.
    
    Args:
        filepath: Path to .env file
        
    Returns:
        Dictionary of environment variables
    """
    env_vars = {}
    env_path = Path(filepath)
    
    if not env_path.exists():
        logger.warning(f".env file not found: {filepath}")
        return env_vars
    
    try:
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    env_vars[key] = value
                    os.environ[key] = value
        
        logger.info(f"Loaded {len(env_vars)} environment variables from {filepath}")
    except Exception as e:
        logger.error(f"Error loading .env file: {e}")
    
    return env_vars


def load_json_config(filepath: str) -> Dict[str, Any]:
    """
    Load configuration from JSON file.
    
    Args:
        filepath: Path to JSON config file
        
    Returns:
        Dictionary with configuration
    """
    config_path = Path(filepath)
    
    if not config_path.exists():
        logger.warning(f"Config file not found: {filepath}")
        return {}
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        logger.info(f"Loaded configuration from {filepath}")
        return config
    except Exception as e:
        logger.error(f"Error loading config file: {e}")
        return {}


def get_env_var(key: str, default: Any = None, required: bool = False) -> Any:
    """
    Get environment variable with optional default and validation.
    
    Args:
        key: Environment variable key
        default: Default value if not found
        required: Whether variable is required
        
    Returns:
        Environment variable value
        
    Raises:
        ValueError: If required variable is missing
    """
    value = os.getenv(key, default)
    
    if required and value is None:
        raise ValueError(f"Required environment variable {key} is not set")
    
    return value


def get_env_bool(key: str, default: bool = False) -> bool:
    """
    Get boolean environment variable.
    
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
    Get integer environment variable.
    
    Args:
        key: Environment variable key
        default: Default value
        
    Returns:
        Integer value
    """
    try:
        return int(os.getenv(key, str(default)))
    except ValueError:
        logger.warning(f"Invalid integer value for {key}, using default: {default}")
        return default


def get_env_float(key: str, default: float = 0.0) -> float:
    """
    Get float environment variable.
    
    Args:
        key: Environment variable key
        default: Default value
        
    Returns:
        Float value
    """
    try:
        return float(os.getenv(key, str(default)))
    except ValueError:
        logger.warning(f"Invalid float value for {key}, using default: {default}")
        return default

