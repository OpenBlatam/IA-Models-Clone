"""
Validators - Common validation utilities
========================================

Utilidades de validación compartidas para servicios y handlers MCP.
"""

import logging
from typing import Any, Dict, Optional

from ..exceptions import MCPValidationError

logger = logging.getLogger(__name__)


def validate_resource_id(resource_id: Any, max_length: int = 255) -> str:
    """
    Validar que resource_id sea un string no vacío.
    
    Args:
        resource_id: ID del recurso a validar
        max_length: Longitud máxima permitida (default: 255)
        
    Returns:
        resource_id validado como string (normalizado con strip)
        
    Raises:
        ValueError: Si resource_id es inválido
        TypeError: Si resource_id no es string
    """
    if resource_id is None:
        raise ValueError("resource_id cannot be None")
    if not isinstance(resource_id, str):
        raise TypeError(f"resource_id must be a string, got {type(resource_id)}")
    if not resource_id.strip():
        raise ValueError("resource_id cannot be empty or only whitespace")
    
    validated = resource_id.strip()
    if len(validated) > max_length:
        raise ValueError(f"resource_id exceeds maximum length of {max_length} characters")
    
    return validated


def validate_operation(operation: Any, max_length: int = 100) -> str:
    """
    Validar que operation sea un string no vacío.
    
    Args:
        operation: Operación a validar
        max_length: Longitud máxima permitida (default: 100)
        
    Returns:
        operation validado como string (normalizado a lowercase y strip)
        
    Raises:
        ValueError: Si operation es inválido
        TypeError: Si operation no es string
    """
    if operation is None:
        raise ValueError("operation cannot be None")
    if not isinstance(operation, str):
        raise TypeError(f"operation must be a string, got {type(operation)}")
    if not operation.strip():
        raise ValueError("operation cannot be empty or only whitespace")
    
    validated = operation.strip().lower()
    if len(validated) > max_length:
        raise ValueError(f"operation exceeds maximum length of {max_length} characters")
    
    return validated


def validate_user(user: Any, require_sub: bool = False) -> Dict[str, Any]:
    """
    Validar que user sea un diccionario válido.
    
    Args:
        user: Información del usuario a validar
        require_sub: Si True, requiere que 'sub' esté presente (default: False)
        
    Returns:
        user validado como diccionario
        
    Raises:
        ValueError: Si user es inválido o falta 'sub' cuando es requerido
        TypeError: Si user no es un diccionario
    """
    if user is None:
        raise ValueError("user cannot be None")
    if not isinstance(user, dict):
        raise TypeError(f"user must be a dictionary, got {type(user)}")
    if not user:
        raise ValueError("user dictionary cannot be empty")
    
    # Validar 'sub' si es requerido
    if require_sub:
        if "sub" not in user:
            raise ValueError("user dictionary must contain 'sub' key (user_id)")
        if not user.get("sub") or not isinstance(user.get("sub"), str):
            raise ValueError("user 'sub' must be a non-empty string")
    elif "sub" not in user:
        logger.warning("User dictionary missing 'sub' key (user_id)")
    
    return user


def validate_parameters(parameters: Any, max_size: int = 1000) -> Dict[str, Any]:
    """
    Validar que parameters sea un diccionario.
    
    Args:
        parameters: Parámetros a validar
        max_size: Tamaño máximo del diccionario (número de claves, default: 1000)
        
    Returns:
        parameters validado como diccionario (vacío si None)
        
    Raises:
        ValueError: Si parameters no es un diccionario o excede max_size
        TypeError: Si parameters no es None ni dict
    """
    if parameters is None:
        return {}
    
    if not isinstance(parameters, dict):
        raise TypeError(f"parameters must be a dictionary or None, got {type(parameters)}")
    
    if len(parameters) > max_size:
        raise ValueError(f"parameters dictionary exceeds maximum size of {max_size} keys")
    
    return parameters


def validate_request_data(
    resource_id: Any,
    operation: Any,
    user: Any,
    parameters: Optional[Any] = None,
    require_user_sub: bool = False
) -> tuple[str, str, Dict[str, Any], Dict[str, Any]]:
    """
    Validar todos los datos de una request de una vez.
    
    Args:
        resource_id: ID del recurso
        operation: Operación a ejecutar
        user: Información del usuario
        parameters: Parámetros opcionales
        require_user_sub: Si True, requiere que user tenga 'sub' (default: False)
        
    Returns:
        Tupla con (resource_id, operation, user, parameters) validados
        
    Raises:
        MCPValidationError: Si la validación falla (envuelve ValueError/TypeError)
    """
    errors = []
    
    try:
        validated_resource_id = validate_resource_id(resource_id)
    except (ValueError, TypeError) as e:
        errors.append(f"resource_id: {e}")
    
    try:
        validated_operation = validate_operation(operation)
    except (ValueError, TypeError) as e:
        errors.append(f"operation: {e}")
    
    try:
        validated_user = validate_user(user, require_sub=require_user_sub)
    except (ValueError, TypeError) as e:
        errors.append(f"user: {e}")
    
    try:
        validated_parameters = validate_parameters(parameters)
    except (ValueError, TypeError) as e:
        errors.append(f"parameters: {e}")
    
    if errors:
        error_message = "Invalid request data: " + "; ".join(errors)
        raise MCPValidationError(error_message)
    
    return (
        validated_resource_id,
        validated_operation,
        validated_user,
        validated_parameters
    )

