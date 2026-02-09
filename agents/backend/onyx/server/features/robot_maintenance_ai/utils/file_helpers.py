"""
File I/O and date/time helper functions for common operations.
Consolidates repetitive file handling and date/time patterns across the codebase.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


def ensure_directory_exists(file_path: str) -> Path:
    """
    Ensure the directory for a file path exists.
    
    Args:
        file_path: Path to file (directory will be created)
    
    Returns:
        Path object for the file
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def get_timestamp_filename(prefix: str, extension: str, directory: Optional[str] = None) -> str:
    """
    Generate a filename with timestamp.
    
    Args:
        prefix: Filename prefix
        extension: File extension (with or without dot)
        directory: Optional directory path
    
    Returns:
        Full path to the generated filename
    """
    if not extension.startswith('.'):
        extension = f'.{extension}'
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{timestamp}{extension}"
    
    if directory:
        path = Path(directory) / filename
        ensure_directory_exists(str(path))
        return str(path)
    
    return filename


def write_json_file(data: Dict[str, Any], file_path: str, indent: int = 2) -> str:
    """
    Write data to a JSON file with consistent formatting.
    
    Args:
        data: Data to write
        file_path: Path to JSON file
        indent: JSON indentation (default: 2)
    
    Returns:
        Path to the written file
    """
    path = ensure_directory_exists(file_path)
    
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)
    
    logger.debug(f"JSON file written: {path}")
    return str(path)


def read_json_file(file_path: str) -> Dict[str, Any]:
    """
    Read data from a JSON file with error handling.
    
    Args:
        file_path: Path to JSON file
    
    Returns:
        Dictionary with file contents
    
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file is not valid JSON
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.debug(f"JSON file read: {path}")
        return data
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in file {file_path}: {e}") from e


def get_iso_timestamp() -> str:
    """
    Get current timestamp in ISO format.
    
    Returns:
        ISO format timestamp string
    """
    return datetime.now().isoformat()


def get_timestamp_string(format_str: str = "%Y%m%d_%H%M%S") -> str:
    """
    Get current timestamp as formatted string.
    
    Args:
        format_str: Timestamp format string (default: "%Y%m%d_%H%M%S")
    
    Returns:
        Formatted timestamp string
    """
    return datetime.now().strftime(format_str)


def parse_iso_date(date_str: Optional[str], default: Optional[datetime] = None) -> Optional[datetime]:
    """
    Safely parse ISO format date string.
    
    Args:
        date_str: ISO format date string (e.g., "2024-01-15T10:30:00")
        default: Default datetime if parsing fails or string is None
    
    Returns:
        Parsed datetime object or default
    """
    if not date_str:
        return default
    
    try:
        return datetime.fromisoformat(date_str)
    except (ValueError, TypeError) as e:
        logger.warning(f"Failed to parse ISO date '{date_str}': {e}")
        return default


def get_date_range(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    default_days: int = 30
) -> Tuple[datetime, datetime]:
    """
    Get date range from optional ISO date strings with defaults.
    
    Args:
        start_date: Optional start date in ISO format
        end_date: Optional end date in ISO format (defaults to now)
        default_days: Default number of days before end_date if start_date not provided
    
    Returns:
        Tuple of (start_date, end_date) as datetime objects
    """
    if end_date:
        end = parse_iso_date(end_date, default=datetime.now())
    else:
        end = datetime.now()
    
    if start_date:
        start = parse_iso_date(start_date, default=end - timedelta(days=default_days))
    else:
        start = end - timedelta(days=default_days)
    
    return start, end


def datetime_to_iso(dt: datetime) -> str:
    """
    Convert datetime object to ISO format string.
    
    Args:
        dt: Datetime object
    
    Returns:
        ISO format string
    """
    return dt.isoformat()


def extract_date_from_iso(iso_string: Optional[str]) -> Optional[str]:
    """
    Extract date part (YYYY-MM-DD) from ISO format timestamp string.
    
    Args:
        iso_string: ISO format timestamp string (e.g., "2024-01-15T10:30:00")
    
    Returns:
        Date string (YYYY-MM-DD) or None if invalid
    """
    if not iso_string or len(iso_string) < 10:
        return None
    return iso_string[:10]


def get_timestamp_id(prefix: str = "") -> str:
    """
    Generate a unique ID based on current timestamp.
    
    Args:
        prefix: Optional prefix for the ID (e.g., "audit_", "alert_")
    
    Returns:
        ID string with format: "{prefix}{timestamp}" or "{timestamp}" if no prefix
    """
    timestamp = datetime.now().timestamp()
    if prefix:
        return f"{prefix}{timestamp}"
    return str(timestamp)


def create_resource(
    data: Dict[str, Any],
    id_prefix: str = "",
    include_timestamps: bool = True
) -> Dict[str, Any]:
    """
    Create a resource dictionary with common fields (id, created_at, updated_at).
    
    Args:
        data: Dictionary with resource data (will be merged with common fields)
        id_prefix: Prefix for the generated ID (e.g., "incident_", "template_")
        include_timestamps: Whether to include created_at and updated_at fields
    
    Returns:
        Dictionary with resource data including common fields
    """
    resource = data.copy()
    resource["id"] = get_timestamp_id(id_prefix)
    
    if include_timestamps:
        timestamp = get_iso_timestamp()
        resource["created_at"] = timestamp
        resource["updated_at"] = timestamp
    
    return resource


def update_resource(
    resource: Dict[str, Any],
    updates: Dict[str, Any],
    update_timestamp: bool = True
) -> Dict[str, Any]:
    """
    Update a resource dictionary with new data and optionally update the updated_at field.
    
    Args:
        resource: Existing resource dictionary to update
        updates: Dictionary with fields to update
        update_timestamp: Whether to update the updated_at field
    
    Returns:
        Updated resource dictionary
    """
    updated_resource = resource.copy()
    updated_resource.update(updates)
    
    if update_timestamp:
        updated_resource["updated_at"] = get_iso_timestamp()
    
    return updated_resource

