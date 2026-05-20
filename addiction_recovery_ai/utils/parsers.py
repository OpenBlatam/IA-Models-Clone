"""
Parsing utilities
Data parsing functions
"""

from typing import Any, Optional, List, Dict
import json

try:
    from utils.date_helpers import parse_iso_date
    from utils.type_converters import to_int, to_float, to_bool, to_datetime
except ImportError:
    from ..date_helpers import parse_iso_date
    from ..type_converters import to_int, to_float, to_bool, to_datetime


def parse_json(json_str: str) -> Any:
    """
    Parse JSON string
    
    Args:
        json_str: JSON string
    
    Returns:
        Parsed data
    
    Raises:
        ValueError if JSON is invalid
    """
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}")


def parse_query_string(query_str: str) -> Dict[str, str]:
    """
    Parse query string
    
    Args:
        query_str: Query string (e.g., "key1=value1&key2=value2")
    
    Returns:
        Dictionary of parameters
    """
    from urllib.parse import parse_qs, unquote
    
    params = {}
    for key, values in parse_qs(query_str).items():
        params[unquote(key)] = unquote(values[0]) if values else ""
    
    return params


def parse_csv_line(line: str, delimiter: str = ",") -> List[str]:
    """
    Parse CSV line
    
    Args:
        line: CSV line
        delimiter: Field delimiter
    
    Returns:
        List of fields
    """
    import csv
    from io import StringIO
    
    reader = csv.reader(StringIO(line), delimiter=delimiter)
    return next(reader)


def parse_key_value_pairs(
    text: str,
    separator: str = "=",
    line_separator: str = "\n"
) -> Dict[str, str]:
    """
    Parse key-value pairs from text
    
    Args:
        text: Text with key-value pairs
        separator: Key-value separator
        line_separator: Line separator
    
    Returns:
        Dictionary of key-value pairs
    """
    result = {}
    
    for line in text.split(line_separator):
        line = line.strip()
        if not line or separator not in line:
            continue
        
        key, value = line.split(separator, 1)
        result[key.strip()] = value.strip()
    
    return result

