"""
Serialization utilities for API responses
"""

from typing import Any, Dict, List
from datetime import datetime, date
from decimal import Decimal
import json


def serialize_datetime(obj: datetime) -> str:
    """Serialize datetime to ISO format string"""
    return obj.isoformat()


def serialize_date(obj: date) -> str:
    """Serialize date to ISO format string"""
    return obj.isoformat()


def serialize_decimal(obj: Decimal) -> float:
    """Serialize Decimal to float"""
    return float(obj)


def serialize_for_json(obj: Any) -> Any:
    """
    Serialize object for JSON encoding
    
    Handles datetime, date, Decimal, and other non-serializable types
    """
    if isinstance(obj, datetime):
        return serialize_datetime(obj)
    
    if isinstance(obj, date):
        return serialize_date(obj)
    
    if isinstance(obj, Decimal):
        return serialize_decimal(obj)
    
    if isinstance(obj, dict):
        return {key: serialize_for_json(value) for key, value in obj.items()}
    
    if isinstance(obj, list):
        return [serialize_for_json(item) for item in obj]
    
    return obj


def to_json_string(obj: Any, indent: int | None = None) -> str:
    """
    Convert object to JSON string with proper serialization
    
    Args:
        obj: Object to serialize
        indent: Optional indentation for pretty printing
    
    Returns:
        JSON string
    """
    serialized = serialize_for_json(obj)
    return json.dumps(serialized, indent=indent, ensure_ascii=False)


def prepare_response_data(data: Any) -> Dict[str, Any] | List[Any]:
    """
    Prepare data for API response
    
    Ensures all data is JSON-serializable
    
    Args:
        data: Data to prepare
    
    Returns:
        Prepared data ready for JSON serialization
    """
    return serialize_for_json(data)

