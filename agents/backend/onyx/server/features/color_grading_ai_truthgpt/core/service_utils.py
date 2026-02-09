"""
Service Utilities for Color Grading AI
=======================================

Common utilities for services.
"""

import logging
import hashlib
import json
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


def generate_id(prefix: str = "", length: int = 8) -> str:
    """
    Generate unique ID.
    
    Args:
        prefix: Optional prefix
        length: ID length
        
    Returns:
        Unique ID
    """
    import secrets
    import string
    
    chars = string.ascii_letters + string.digits
    random_part = ''.join(secrets.choice(chars) for _ in range(length))
    
    if prefix:
        return f"{prefix}_{random_part}"
    return random_part


def hash_data(data: Any) -> str:
    """
    Generate hash for data.
    
    Args:
        data: Data to hash
        
    Returns:
        Hash string
    """
    if isinstance(data, dict):
        data_str = json.dumps(data, sort_keys=True)
    else:
        data_str = str(data)
    
    return hashlib.md5(data_str.encode()).hexdigest()


def safe_json_load(file_path: Path, default: Any = None) -> Any:
    """
    Safely load JSON file.
    
    Args:
        file_path: File path
        default: Default value if error
        
    Returns:
        Loaded data or default
    """
    try:
        if not file_path.exists():
            return default
        
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading JSON from {file_path}: {e}")
        return default


def safe_json_save(file_path: Path, data: Any) -> bool:
    """
    Safely save JSON file.
    
    Args:
        file_path: File path
        data: Data to save
        
    Returns:
        True if successful
    """
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        
        return True
    except Exception as e:
        logger.error(f"Error saving JSON to {file_path}: {e}")
        return False


def normalize_path(path: str, base_dir: Optional[Path] = None) -> Path:
    """
    Normalize and validate file path.
    
    Args:
        path: File path
        base_dir: Base directory for relative paths
        
    Returns:
        Normalized path
    """
    if base_dir:
        normalized = (base_dir / path).resolve()
    else:
        normalized = Path(path).resolve()
    
    return normalized


def filter_dict(data: Dict[str, Any], keys: List[str]) -> Dict[str, Any]:
    """
    Filter dictionary to include only specified keys.
    
    Args:
        data: Dictionary to filter
        keys: Keys to include
        
    Returns:
        Filtered dictionary
    """
    return {k: v for k, v in data.items() if k in keys}


def merge_dicts(*dicts: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge multiple dictionaries.
    
    Args:
        *dicts: Dictionaries to merge
        
    Returns:
        Merged dictionary
    """
    result = {}
    for d in dicts:
        result.update(d)
    return result


def format_duration(seconds: float) -> str:
    """
    Format duration in human-readable format.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted string
    """
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def get_timestamp() -> str:
    """Get current timestamp as ISO string."""
    return datetime.now().isoformat()


def parse_timestamp(timestamp: str) -> datetime:
    """Parse ISO timestamp string."""
    return datetime.fromisoformat(timestamp)




