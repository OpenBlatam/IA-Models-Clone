"""
Export Utilities

Utility functions for exporting data in various formats.
"""

import json
import csv
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime


def export_to_json(
    data: Any,
    file_path: str,
    indent: int = 2,
    ensure_ascii: bool = False
) -> bool:
    """
    Export data to JSON file.
    
    Args:
        data: Data to export
        file_path: Output file path
        indent: JSON indentation
        ensure_ascii: Whether to ensure ASCII encoding
    
    Returns:
        True if successful
    """
    try:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii, default=str)
        
        return True
    except Exception as e:
        print(f"Failed to export JSON: {str(e)}")
        return False


def export_to_csv(
    data: List[Dict[str, Any]],
    file_path: str,
    fieldnames: Optional[List[str]] = None
) -> bool:
    """
    Export data to CSV file.
    
    Args:
        data: List of dictionaries to export
        file_path: Output file path
        fieldnames: Optional list of field names
    
    Returns:
        True if successful
    """
    try:
        if not data:
            return False
        
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Get fieldnames from first item if not provided
        if fieldnames is None:
            fieldnames = list(data[0].keys())
        
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
        
        return True
    except Exception as e:
        print(f"Failed to export CSV: {str(e)}")
        return False


def export_to_dict(
    data: Any,
    include_none: bool = False
) -> Dict[str, Any]:
    """
    Export data to dictionary format.
    
    Args:
        data: Data to export
        include_none: Whether to include None values
    
    Returns:
        Dictionary representation
    """
    if hasattr(data, 'to_dict'):
        return data.to_dict()
    elif isinstance(data, dict):
        result = {}
        for k, v in data.items():
            if v is None and not include_none:
                continue
            if hasattr(v, 'to_dict'):
                result[k] = v.to_dict()
            elif isinstance(v, (list, tuple)):
                result[k] = [export_to_dict(item, include_none) for item in v]
            elif isinstance(v, dict):
                result[k] = export_to_dict(v, include_none)
            else:
                result[k] = v
        return result
    elif isinstance(data, (list, tuple)):
        return [export_to_dict(item, include_none) for item in data]
    else:
        return data



