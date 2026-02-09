"""
Serialization utilities for converting models to dictionaries.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import json

from sqlalchemy.orm import DeclarativeBase


def serialize_model(model: DeclarativeBase, exclude: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Serialize a SQLAlchemy model to dictionary.
    
    Args:
        model: SQLAlchemy model instance
        exclude: Optional list of fields to exclude
        
    Returns:
        Dictionary representation of the model
    """
    if model is None:
        return {}
    
    exclude = exclude or []
    
    result = {}
    for column in model.__table__.columns:
        if column.name in exclude:
            continue
        
        value = getattr(model, column.name, None)
        
        # Handle datetime serialization
        if isinstance(value, datetime):
            value = value.isoformat()
        
        result[column.name] = value
    
    return result


def serialize_list(
    models: List[DeclarativeBase],
    exclude: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    Serialize a list of models to dictionaries.
    
    Args:
        models: List of SQLAlchemy model instances
        exclude: Optional list of fields to exclude
        
    Returns:
        List of dictionary representations
    """
    return [serialize_model(model, exclude=exclude) for model in models]


def serialize_with_relations(
    model: DeclarativeBase,
    relations: Optional[List[str]] = None,
    exclude: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Serialize a model with its relations.
    
    Args:
        model: SQLAlchemy model instance
        relations: Optional list of relation names to include
        exclude: Optional list of fields to exclude
        
    Returns:
        Dictionary with model and relations
    """
    result = serialize_model(model, exclude=exclude)
    
    if relations:
        for relation_name in relations:
            if hasattr(model, relation_name):
                relation = getattr(model, relation_name)
                if relation is not None:
                    if isinstance(relation, list):
                        result[relation_name] = serialize_list(relation)
                    else:
                        result[relation_name] = serialize_model(relation)
    
    return result






