"""
Validation Helpers
==================
Utilidades adicionales para validación.
"""

import re
from typing import Optional, List, Callable, Any
from email.utils import parseaddr


def is_valid_email(email: str) -> bool:
    """
    Validar formato de email.
    
    Args:
        email: Email a validar
        
    Returns:
        True si es válido
    """
    if not email:
        return False
    
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))


def is_valid_url(url: str) -> bool:
    """
    Validar formato de URL.
    
    Args:
        url: URL a validar
        
    Returns:
        True si es válido
    """
    if not url:
        return False
    
    pattern = r'^https?://(?:[-\w.])+(?:\:[0-9]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:\#(?:[\w.])*)?)?$'
    return bool(re.match(pattern, url))


def is_valid_phone(phone: str) -> bool:
    """
    Validar formato de teléfono básico.
    
    Args:
        phone: Teléfono a validar
        
    Returns:
        True si es válido
    """
    if not phone:
        return False
    
    # Remover espacios, guiones, paréntesis
    cleaned = re.sub(r'[\s\-\(\)]', '', phone)
    
    # Debe contener solo dígitos y opcionalmente +
    pattern = r'^\+?[1-9]\d{6,14}$'
    return bool(re.match(pattern, cleaned))


def validate_range(value: float, min_value: float, max_value: float) -> bool:
    """
    Validar que un valor esté en un rango.
    
    Args:
        value: Valor a validar
        min_value: Valor mínimo
        max_value: Valor máximo
        
    Returns:
        True si está en el rango
    """
    return min_value <= value <= max_value


def validate_length(text: str, min_length: int = 0, max_length: Optional[int] = None) -> bool:
    """
    Validar longitud de texto.
    
    Args:
        text: Texto a validar
        min_length: Longitud mínima
        max_length: Longitud máxima (opcional)
        
    Returns:
        True si la longitud es válida
    """
    if not text:
        return min_length == 0
    
    length = len(text)
    
    if length < min_length:
        return False
    
    if max_length and length > max_length:
        return False
    
    return True


def validate_required_fields(data: dict, required_fields: List[str]) -> List[str]:
    """
    Validar que campos requeridos estén presentes.
    
    Args:
        data: Diccionario a validar
        required_fields: Lista de campos requeridos
        
    Returns:
        Lista de campos faltantes
    """
    missing = []
    
    for field in required_fields:
        if field not in data or data[field] is None:
            missing.append(field)
    
    return missing


def validate_with_custom_validator(
    value: Any,
    validator: Callable[[Any], bool],
    error_message: str = "Validation failed"
) -> bool:
    """
    Validar con función personalizada.
    
    Args:
        value: Valor a validar
        validator: Función de validación
        error_message: Mensaje de error
        
    Returns:
        True si es válido
        
    Raises:
        ValueError: Si la validación falla
    """
    if not validator(value):
        raise ValueError(error_message)
    return True

