"""
Configuration loader with support for YAML files.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


def load_yaml_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML config file. If None, looks for config.yaml in current directory.
    
    Returns:
        Dictionary with configuration
    """
    if not YAML_AVAILABLE:
        logger.warning("PyYAML not available. Install with: pip install pyyaml")
        return {}
    
    if config_path is None:
        config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    else:
        config_path = Path(config_path)
    
    if not config_path.exists():
        logger.warning(f"Config file not found: {config_path}")
        return {}
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        config = _expand_env_vars(config)
        logger.info(f"Configuration loaded from {config_path}")
        return config or {}
    except Exception as e:
        logger.error(f"Error loading config file: {e}")
        return {}


def _expand_env_vars(obj: Any) -> Any:
    """
    Recursively expand environment variables in configuration.
    
    Args:
        obj: Configuration object (dict, list, or string)
    
    Returns:
        Object with environment variables expanded
    """
    if isinstance(obj, dict):
        return {key: _expand_env_vars(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [_expand_env_vars(item) for item in obj]
    elif isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
        env_var = obj[2:-1]
        return os.getenv(env_var, obj)
    else:
        return obj






