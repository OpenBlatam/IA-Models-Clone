"""
Helper functions for serializing Pydantic models and data structures.
Eliminates repetitive model_dump() patterns.
"""

from typing import Any, List, Optional, Dict, Union
from pydantic import BaseModel


def serialize_model(model: BaseModel, exclude_none: bool = False, **kwargs) -> Dict[str, Any]:
    """
    Serializa un modelo Pydantic a diccionario.
    
    Args:
        model: Modelo Pydantic a serializar
        exclude_none: Si excluir campos None (default: False)
        **kwargs: Argumentos adicionales para model_dump()
        
    Returns:
        Diccionario serializado
        
    Examples:
        >>> serialize_model(identity)
        {"profile_id": "...", "username": "..."}
        
        >>> serialize_model(identity, exclude_none=True)
        {"profile_id": "...", "username": "..."}  # sin campos None
    """
    return model.model_dump(exclude_none=exclude_none, **kwargs)


def serialize_models(
    models: List[BaseModel],
    exclude_none: bool = False,
    **kwargs
) -> List[Dict[str, Any]]:
    """
    Serializa una lista de modelos Pydantic.
    
    Args:
        models: Lista de modelos a serializar
        exclude_none: Si excluir campos None
        **kwargs: Argumentos adicionales para model_dump()
        
    Returns:
        Lista de diccionarios serializados
        
    Examples:
        >>> serialize_models([identity1, identity2])
        [{"profile_id": "..."}, {"profile_id": "..."}]
    """
    return [serialize_model(model, exclude_none=exclude_none, **kwargs) for model in models]


def serialize_optional_model(
    model: Optional[BaseModel],
    exclude_none: bool = False,
    **kwargs
) -> Optional[Dict[str, Any]]:
    """
    Serializa un modelo opcional (puede ser None).
    
    Args:
        model: Modelo opcional a serializar
        exclude_none: Si excluir campos None
        **kwargs: Argumentos adicionales
        
    Returns:
        Diccionario serializado o None
    """
    if model is None:
        return None
    return serialize_model(model, exclude_none=exclude_none, **kwargs)


def serialize_nested_models(
    data: Union[Dict, List, BaseModel, Any],
    exclude_none: bool = False,
    **kwargs
) -> Union[Dict, List, Any]:
    """
    Serializa recursivamente estructuras que contienen modelos Pydantic.
    
    Args:
        data: Datos que pueden contener modelos Pydantic
        exclude_none: Si excluir campos None
        **kwargs: Argumentos adicionales
        
    Returns:
        Estructura serializada
        
    Examples:
        >>> serialize_nested_models({
        ...     "identity": identity_model,
        ...     "profiles": [profile1, profile2]
        ... })
        {
            "identity": {"profile_id": "..."},
            "profiles": [{"username": "..."}, {"username": "..."}]
        }
    """
    if isinstance(data, BaseModel):
        return serialize_model(data, exclude_none=exclude_none, **kwargs)
    elif isinstance(data, dict):
        return {
            key: serialize_nested_models(value, exclude_none=exclude_none, **kwargs)
            for key, value in data.items()
        }
    elif isinstance(data, list):
        return [
            serialize_nested_models(item, exclude_none=exclude_none, **kwargs)
            for item in data
        ]
    else:
        return data








