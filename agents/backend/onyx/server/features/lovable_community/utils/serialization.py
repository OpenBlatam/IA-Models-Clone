"""
Fast Serialization Utilities

Optimized serialization for better performance.
"""

import json
import orjson
from typing import Any, Dict, List
from datetime import datetime
from decimal import Decimal


def fast_json_dumps(obj: Any) -> str:
    """
    Fast JSON serialization using orjson.
    
    Args:
        obj: Object to serialize
        
    Returns:
        JSON string
    """
    return orjson.dumps(
        obj,
        option=orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_SERIALIZE_DATACLASS
    ).decode('utf-8')


def fast_json_loads(s: str) -> Any:
    """
    Fast JSON deserialization using orjson.
    
    Args:
        s: JSON string
        
    Returns:
        Deserialized object
    """
    return orjson.loads(s)


def serialize_datetime(dt: datetime) -> str:
    """
    Fast datetime serialization.
    
    Args:
        dt: Datetime object
        
    Returns:
        ISO format string
    """
    return dt.isoformat()


def serialize_model_fast(model: Any) -> Dict[str, Any]:
    """
    Fast model serialization avoiding SQLAlchemy overhead.
    
    Args:
        model: SQLAlchemy model instance
        
    Returns:
        Dictionary representation
    """
    result = {}
    for column in model.__table__.columns:
        value = getattr(model, column.name)
        if isinstance(value, datetime):
            result[column.name] = serialize_datetime(value)
        elif isinstance(value, Decimal):
            result[column.name] = float(value)
        else:
            result[column.name] = value
    return result


def serialize_models_batch(models: List[Any]) -> List[Dict[str, Any]]:
    """
    Batch serialize multiple models efficiently.
    
    Args:
        models: List of SQLAlchemy model instances
        
    Returns:
        List of dictionaries
    """
    return [serialize_model_fast(model) for model in models]
