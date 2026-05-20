"""
Common Utilities
================

Shared utilities and constants used across the core module.
"""

from typing import Dict, Any, Optional
from pathlib import Path

# Common message creation
def create_message(role: str, content: Any) -> Dict[str, Any]:
    """
    Create a message dictionary for API requests.
    
    Args:
        role: Message role (user, assistant, system)
        content: Message content
        
    Returns:
        Message dictionary
    """
    return {"role": role, "content": content}


# Common path utilities
def normalize_path(path: str | Path) -> Path:
    """
    Normalize a path to Path object.
    
    Args:
        path: Path string or Path object
        
    Returns:
        Normalized Path object
    """
    return Path(path) if isinstance(path, str) else path


def ensure_path_exists(path: str | Path) -> Path:
    """
    Ensure a path exists, creating directories if needed.
    
    Args:
        path: Path to ensure
        
    Returns:
        Path object
    """
    path_obj = normalize_path(path)
    if path_obj.is_file():
        path_obj.parent.mkdir(parents=True, exist_ok=True)
    else:
        path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


# Common dictionary utilities
def safe_get(data: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    """
    Safely get nested dictionary values.
    
    Args:
        data: Dictionary to search
        *keys: Keys to traverse
        default: Default value if key not found
        
    Returns:
        Value or default
    """
    current = data
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    return current


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two dictionaries.
    
    Args:
        base: Base dictionary
        override: Dictionary to merge in
        
    Returns:
        Merged dictionary
    """
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    
    return result




