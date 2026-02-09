"""
Pydantic helper utilities for optimization
"""

from typing import Any, Dict, List
from pydantic import BaseModel


def model_to_dict(model: BaseModel, exclude_none: bool = True) -> Dict[str, Any]:
    """
    Convert Pydantic model to dict with optimization
    
    Args:
        model: Pydantic model instance
        exclude_none: Whether to exclude None values
    
    Returns:
        Dictionary representation
    """
    return model.model_dump(exclude_none=exclude_none, mode='json')


def models_to_dicts(
    models: List[BaseModel],
    exclude_none: bool = True
) -> List[Dict[str, Any]]:
    """
    Convert list of Pydantic models to list of dicts
    
    Args:
        models: List of Pydantic model instances
        exclude_none: Whether to exclude None values
    
    Returns:
        List of dictionary representations
    """
    return [model_to_dict(model, exclude_none) for model in models]


def validate_and_parse(
    data: Dict[str, Any],
    model_class: type[BaseModel]
) -> BaseModel:
    """
    Validate and parse data into Pydantic model
    
    Args:
        data: Dictionary data
        model_class: Pydantic model class
    
    Returns:
        Validated model instance
    
    Raises:
        ValidationError if data is invalid
    """
    return model_class.model_validate(data)


def partial_update_model(
    model: BaseModel,
    updates: Dict[str, Any]
) -> BaseModel:
    """
    Partially update a Pydantic model
    
    Args:
        model: Existing model instance
        updates: Dictionary of updates
    
    Returns:
        Updated model instance
    """
    current_data = model.model_dump(exclude_none=True)
    current_data.update(updates)
    return model.__class__.model_validate(current_data)

