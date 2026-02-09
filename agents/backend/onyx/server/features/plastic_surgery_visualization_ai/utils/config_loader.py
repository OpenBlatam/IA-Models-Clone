"""Configuration loading utilities."""

import os
from pathlib import Path
from typing import Optional, Dict, Any
import json
import yaml
from functools import lru_cache

from utils.logger import get_logger

logger = get_logger(__name__)


def load_env_file(env_path: Optional[Path] = None) -> Dict[str, str]:
    """
    Load environment variables from .env file.
    
    Args:
        env_path: Path to .env file
        
    Returns:
        Dictionary of environment variables
    """
    if env_path is None:
        env_path = Path(".env")
    
    env_vars = {}
    
    if not env_path.exists():
        logger.warning(f".env file not found at {env_path}")
        return env_vars
    
    try:
        with open(env_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                
                if "=" in line:
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    env_vars[key] = value
    except Exception as e:
        logger.error(f"Error loading .env file: {e}")
    
    return env_vars


def load_json_config(config_path: Path) -> Optional[Dict[str, Any]]:
    """
    Load configuration from JSON file.
    
    Args:
        config_path: Path to JSON config file
        
    Returns:
        Configuration dictionary or None
    """
    if not config_path.exists():
        logger.warning(f"Config file not found: {config_path}")
        return None
    
    try:
        with open(config_path, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading config file: {e}")
        return None


def get_env_var(key: str, default: Optional[str] = None) -> Optional[str]:
    """
    Get environment variable with optional default.
    
    Args:
        key: Environment variable key
        default: Default value if not found
        
    Returns:
        Environment variable value or default
    """
    return os.getenv(key, default)


def require_env_var(key: str) -> str:
    """
    Require environment variable (raises if missing).
    
    Args:
        key: Environment variable key
        
    Returns:
        Environment variable value
        
    Raises:
        ValueError: If variable is not set
    """
    value = os.getenv(key)
    if not value:
        raise ValueError(f"Required environment variable {key} is not set")
    return value


# Additional functions from config_helpers.py
class ConfigLoader:
    """Helper for loading configurations."""
    
    @staticmethod
    def load_json(path: Path) -> Dict[str, Any]:
        """Load JSON configuration."""
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading JSON config: {e}")
            raise
    
    @staticmethod
    def load_yaml(path: Path) -> Dict[str, Any]:
        """Load YAML configuration."""
        try:
            with open(path, 'r') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logger.error(f"Error loading YAML config: {e}")
            raise
    
    @staticmethod
    def load_env_file(path: Path) -> Dict[str, str]:
        """Load .env file."""
        return load_env_file(path)
    
    @staticmethod
    def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
        """Merge multiple configuration dictionaries."""
        merged = {}
        for config in configs:
            merged.update(config)
        return merged


class EnvironmentHelper:
    """Helper for environment detection."""
    
    @staticmethod
    def is_development() -> bool:
        """Check if running in development."""
        env = os.getenv("ENVIRONMENT", "").lower()
        return env in ["dev", "development", "local"]
    
    @staticmethod
    def is_production() -> bool:
        """Check if running in production."""
        env = os.getenv("ENVIRONMENT", "").lower()
        return env in ["prod", "production"]
    
    @staticmethod
    def is_testing() -> bool:
        """Check if running in test environment."""
        env = os.getenv("ENVIRONMENT", "").lower()
        return env in ["test", "testing"]
    
    @staticmethod
    def get_environment() -> str:
        """Get current environment."""
        return os.getenv("ENVIRONMENT", "development").lower()


@lru_cache(maxsize=1)
def get_project_root() -> Path:
    """Get project root directory."""
    current = Path(__file__).resolve()
    markers = [".git", "pyproject.toml", "setup.py", "requirements.txt"]
    
    for parent in current.parents:
        if any((parent / marker).exists() for marker in markers):
            return parent
    
    return current.parent.parent


def get_config_path(filename: str) -> Path:
    """Get path to configuration file."""
    project_root = get_project_root()
    return project_root / filename


def load_config_file(filename: str) -> Dict[str, Any]:
    """Load configuration file."""
    path = get_config_path(filename)
    
    if not path.exists():
        logger.warning(f"Config file not found: {path}")
        return {}
    
    if filename.endswith('.json'):
        return ConfigLoader.load_json(path)
    elif filename.endswith(('.yaml', '.yml')):
        return ConfigLoader.load_yaml(path)
    elif filename.endswith('.env'):
        return ConfigLoader.load_env_file(path)
    else:
        logger.warning(f"Unknown config file type: {filename}")
        return {}

