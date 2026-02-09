"""
Helper functions for common validation patterns.
Eliminates repetitive validation code throughout the codebase.
"""

from typing import Any, Optional, List, Type, Union
from enum import Enum
from fastapi import HTTPException, status


def validate_not_none(value: Any, field_name: str) -> None:
    """
    Valida que un valor no sea None.
    
    Args:
        value: Valor a validar
        field_name: Nombre del campo para mensaje de error
        
    Raises:
        ValueError: Si el valor es None
    """
    if value is None:
        raise ValueError(f"{field_name} cannot be None")


def validate_not_empty(value: Union[str, List, dict], field_name: str) -> None:
    """
    Valida que un valor no esté vacío.
    
    Args:
        value: Valor a validar (string, lista, o dict)
        field_name: Nombre del campo
        
    Raises:
        ValueError: Si el valor está vacío
    """
    if not value:
        raise ValueError(f"{field_name} cannot be empty")


def validate_at_least_one(
    *values: Any,
    field_names: List[str],
    message: str = None
) -> None:
    """
    Valida que al menos uno de los valores no sea None/vacío.
    
    Args:
        *values: Valores a validar
        field_names: Nombres de los campos
        message: Mensaje de error personalizado
        
    Raises:
        HTTPException: Si todos los valores son None/vacíos
    """
    if not any(v for v in values):
        if message:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=message
            )
        else:
            field_list = ", ".join(field_names)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Al menos uno de los siguientes debe ser proporcionado: {field_list}"
            )


def validate_enum(
    value: str,
    enum_class: Type[Enum],
    field_name: str = "value"
) -> Enum:
    """
    Valida y convierte un string a un valor de Enum.
    
    Args:
        value: String a validar
        enum_class: Clase Enum
        field_name: Nombre del campo para mensaje de error
        
    Returns:
        Valor del Enum
        
    Raises:
        HTTPException: Si el valor no es válido
    """
    try:
        return enum_class(value)
    except ValueError:
        valid_values = [e.value for e in enum_class]
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"{field_name} debe ser uno de: {', '.join(valid_values)}"
        )


def validate_platform(platform_str: str):
    """
    Valida y convierte string a Platform enum.
    
    Args:
        platform_str: String de plataforma
        
    Returns:
        Platform enum
        
    Raises:
        HTTPException: Si la plataforma no es válida
    """
    from ..core.models import Platform
    return validate_enum(platform_str, Platform, "platform")


def validate_content_type(content_type_str: str):
    """
    Valida y convierte string a ContentType enum.
    
    Args:
        content_type_str: String de tipo de contenido
        
    Returns:
        ContentType enum
        
    Raises:
        HTTPException: Si el tipo no es válido
    """
    from ..core.models import ContentType
    return validate_enum(content_type_str, ContentType, "content_type")








