"""
Validation Utilities - Utilidades de validación
===============================================

Utilidades para validación común y consistente en todo el código.
"""

from typing import Optional, Callable, Any
import re


def validate_not_none(value: Any, name: str = "value") -> None:
    """
    Validar que un valor no sea None.
    
    Args:
        value: Valor a validar.
        name: Nombre del valor (para mensaje de error).
    
    Raises:
        ValueError: Si el valor es None.
    """
    if value is None:
        raise ValueError(f"{name} cannot be None")


def validate_not_empty(value: str, name: str = "value") -> None:
    """
    Validar que un string no esté vacío.
    
    Args:
        value: String a validar.
        name: Nombre del valor (para mensaje de error).
    
    Raises:
        ValueError: Si el string está vacío o es None.
    """
    if not value or not value.strip():
        raise ValueError(f"{name} cannot be empty")


def validate_port(port: int, name: str = "port") -> None:
    """
    Validar que un puerto esté en el rango válido.
    
    Args:
        port: Puerto a validar.
        name: Nombre del puerto (para mensaje de error).
    
    Raises:
        ValueError: Si el puerto está fuera del rango válido.
    """
    if not (1 <= port <= 65535):
        raise ValueError(f"{name} must be between 1 and 65535, got {port}")


def validate_positive(value: int | float, name: str = "value") -> None:
    """
    Validar que un valor numérico sea positivo.
    
    Args:
        value: Valor a validar.
        name: Nombre del valor (para mensaje de error).
    
    Raises:
        ValueError: Si el valor no es positivo.
    """
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def validate_non_negative(value: int | float, name: str = "value") -> None:
    """
    Validar que un valor numérico sea no negativo.
    
    Args:
        value: Valor a validar.
        name: Nombre del valor (para mensaje de error).
    
    Raises:
        ValueError: Si el valor es negativo.
    """
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")


def validate_in_range(
    value: int | float,
    min_value: int | float,
    max_value: int | float,
    name: str = "value"
) -> None:
    """
    Validar que un valor esté en un rango específico.
    
    Args:
        value: Valor a validar.
        min_value: Valor mínimo (inclusive).
        max_value: Valor máximo (inclusive).
        name: Nombre del valor (para mensaje de error).
    
    Raises:
        ValueError: Si el valor está fuera del rango.
    """
    if not (min_value <= value <= max_value):
        raise ValueError(
            f"{name} must be between {min_value} and {max_value}, got {value}"
        )


def validate_regex(
    value: str,
    pattern: str,
    name: str = "value",
    error_message: Optional[str] = None
) -> None:
    """
    Validar que un string coincida con un patrón regex.
    
    Args:
        value: String a validar.
        pattern: Patrón regex.
        name: Nombre del valor (para mensaje de error).
        error_message: Mensaje de error personalizado.
    
    Raises:
        ValueError: Si el string no coincide con el patrón.
    """
    if not re.match(pattern, value):
        if error_message:
            raise ValueError(error_message)
        raise ValueError(f"{name} does not match pattern {pattern}")


def validate_one_of(
    value: Any,
    allowed_values: list[Any],
    name: str = "value"
) -> None:
    """
    Validar que un valor esté en una lista de valores permitidos.
    
    Args:
        value: Valor a validar.
        allowed_values: Lista de valores permitidos.
        name: Nombre del valor (para mensaje de error).
    
    Raises:
        ValueError: Si el valor no está en la lista permitida.
    """
    if value not in allowed_values:
        raise ValueError(
            f"{name} must be one of {allowed_values}, got {value}"
        )


def validate_custom(
    value: Any,
    validator: Callable[[Any], bool],
    name: str = "value",
    error_message: Optional[str] = None
) -> None:
    """
    Validar usando una función personalizada.
    
    Args:
        value: Valor a validar.
        validator: Función que retorna True si el valor es válido.
        name: Nombre del valor (para mensaje de error).
        error_message: Mensaje de error personalizado.
    
    Raises:
        ValueError: Si la validación falla.
    """
    if not validator(value):
        if error_message:
            raise ValueError(error_message)
        raise ValueError(f"{name} failed custom validation")


def validate_all(
    value: Any,
    validators: list[Callable[[Any], None]],
    name: str = "value"
) -> None:
    """
    Aplicar múltiples validadores a un valor.
    
    Args:
        value: Valor a validar.
        validators: Lista de funciones validadoras.
        name: Nombre del valor (para mensaje de error).
    
    Raises:
        ValueError: Si alguna validación falla.
    """
    for validator in validators:
        validator(value)


def safe_validate(
    value: Any,
    validator: Callable[[Any], None],
    default: Optional[Any] = None,
    name: str = "value"
) -> Any:
    """
    Validar un valor de forma segura, retornando un valor por defecto si falla.
    
    Args:
        value: Valor a validar.
        validator: Función validadora.
        default: Valor por defecto si la validación falla.
        name: Nombre del valor (para logging).
    
    Returns:
        El valor si es válido, o el valor por defecto si la validación falla.
    """
    try:
        validator(value)
        return value
    except ValueError:
        return default




