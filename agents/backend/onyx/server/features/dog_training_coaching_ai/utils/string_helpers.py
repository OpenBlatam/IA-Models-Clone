"""
String Helpers
==============
Utilidades para manipulación de strings.
"""

import re
from typing import List, Optional


def camel_to_snake(name: str) -> str:
    """
    Convertir camelCase a snake_case.
    
    Args:
        name: String en camelCase
        
    Returns:
        String en snake_case
    """
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


def snake_to_camel(name: str) -> str:
    """
    Convertir snake_case a camelCase.
    
    Args:
        name: String en snake_case
        
    Returns:
        String en camelCase
    """
    components = name.split('_')
    return components[0] + ''.join(x.title() for x in components[1:])


def truncate_string(text: str, max_length: int, suffix: str = "...") -> str:
    """
    Truncar string a longitud máxima.
    
    Args:
        text: Texto a truncar
        max_length: Longitud máxima
        suffix: Sufijo a agregar si se trunca
        
    Returns:
        Texto truncado
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix


def extract_emails(text: str) -> List[str]:
    """
    Extraer emails de un texto.
    
    Args:
        text: Texto a analizar
        
    Returns:
        Lista de emails encontrados
    """
    pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    return re.findall(pattern, text)


def extract_urls(text: str) -> List[str]:
    """
    Extraer URLs de un texto.
    
    Args:
        text: Texto a analizar
        
    Returns:
        Lista de URLs encontradas
    """
    pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    return re.findall(pattern, text)


def normalize_whitespace(text: str) -> str:
    """
    Normalizar espacios en blanco.
    
    Args:
        text: Texto a normalizar
        
    Returns:
        Texto normalizado
    """
    # Reemplazar múltiples espacios con uno solo
    text = re.sub(r'\s+', ' ', text)
    # Trim
    return text.strip()


def remove_special_chars(text: str, keep: Optional[str] = None) -> str:
    """
    Remover caracteres especiales.
    
    Args:
        text: Texto a limpiar
        keep: Caracteres a mantener (opcional)
        
    Returns:
        Texto limpio
    """
    if keep:
        pattern = f'[^a-zA-Z0-9{re.escape(keep)}]'
    else:
        pattern = r'[^a-zA-Z0-9\s]'
    
    return re.sub(pattern, '', text)

