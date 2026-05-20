"""
Security Utilities
==================
Utilidades de seguridad.
"""

import re
from typing import Optional


def sanitize_html(text: str) -> str:
    """
    Remover HTML tags y caracteres peligrosos.
    
    Args:
        text: Texto a sanitizar
        
    Returns:
        Texto sanitizado
    """
    if not text:
        return ""
    
    # Remover tags HTML
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remover scripts y eventos
    text = re.sub(r'javascript:', '', text, flags=re.IGNORECASE)
    text = re.sub(r'on\w+\s*=', '', text, flags=re.IGNORECASE)
    
    return text


def validate_input_length(text: str, min_length: int = 1, max_length: int = 10000) -> bool:
    """
    Validar longitud de input.
    
    Args:
        text: Texto a validar
        min_length: Longitud mínima
        max_length: Longitud máxima
        
    Returns:
        True si es válido
    """
    if not text:
        return min_length == 0
    
    length = len(text)
    return min_length <= length <= max_length


def detect_sql_injection(text: str) -> bool:
    """
    Detectar posibles intentos de SQL injection.
    
    Args:
        text: Texto a analizar
        
    Returns:
        True si detecta patrón sospechoso
    """
    if not text:
        return False
    
    sql_patterns = [
        r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|EXECUTE)\b)",
        r"(--|#|/\*|\*/)",
        r"(\b(OR|AND)\s+\d+\s*=\s*\d+)",
        r"('|(\\')|(;)|(\|)|(\*))",
    ]
    
    text_upper = text.upper()
    for pattern in sql_patterns:
        if re.search(pattern, text_upper, re.IGNORECASE):
            return True
    
    return False


def validate_api_key_format(api_key: str) -> bool:
    """
    Validar formato de API key.
    
    Args:
        api_key: API key a validar
        
    Returns:
        True si el formato es válido
    """
    if not api_key:
        return False
    
    # Formato básico: al menos 20 caracteres alfanuméricos
    if len(api_key) < 20:
        return False
    
    # Debe contener letras y números
    if not re.search(r'[a-zA-Z]', api_key):
        return False
    
    if not re.search(r'[0-9]', api_key):
        return False
    
    return True

