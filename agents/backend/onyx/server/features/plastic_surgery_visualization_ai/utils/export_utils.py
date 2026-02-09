"""Export utilities."""

from typing import List, Dict, Any
from pathlib import Path
import csv
import json


def export_to_csv(
    data: List[Dict],
    file_path: Path,
    fieldnames: Optional[List[str]] = None
) -> None:
    """
    Export data to CSV file.
    
    Args:
        data: List of dictionaries
        file_path: Output file path
        fieldnames: Optional field names (uses all keys if not provided)
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not data:
        return
    
    if not fieldnames:
        fieldnames = list(data[0].keys())
    
    with open(file_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)


def export_to_json(
    data: Any,
    file_path: Path,
    indent: int = 2
) -> None:
    """
    Export data to JSON file.
    
    Args:
        data: Data to export
        file_path: Output file path
        indent: JSON indentation
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def export_to_text(
    data: List[str],
    file_path: Path,
    separator: str = '\n'
) -> None:
    """
    Export data to text file.
    
    Args:
        data: List of strings
        file_path: Output file path
        separator: Line separator
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(separator.join(data))


def import_from_csv(file_path: Path) -> List[Dict]:
    """
    Import data from CSV file.
    
    Args:
        file_path: CSV file path
        
    Returns:
        List of dictionaries
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return list(reader)


def import_from_json(file_path: Path) -> Any:
    """
    Import data from JSON file.
    
    Args:
        file_path: JSON file path
        
    Returns:
        Parsed data
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

