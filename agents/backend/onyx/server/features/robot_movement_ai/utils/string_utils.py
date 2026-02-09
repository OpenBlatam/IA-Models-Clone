"""
String Utilities - Utilidades de strings
========================================

Utilidades para manipulación y procesamiento de strings.
"""

import re
from typing import List, Optional, Callable
from urllib.parse import quote, unquote


def slugify(text: str, separator: str = '-') -> str:
    """
    Convertir texto a slug.
    
    Args:
        text: Texto a convertir
        separator: Separador (default: '-')
    
    Returns:
        Slug normalizado
    """
    text = text.lower()
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[-\s]+', separator, text)
    return text.strip(separator)


def camel_to_snake(text: str) -> str:
    """
    Convertir camelCase a snake_case.
    
    Args:
        text: Texto en camelCase
    
    Returns:
        Texto en snake_case
    """
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', text)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


def snake_to_camel(text: str, capitalize_first: bool = False) -> str:
    """
    Convertir snake_case a camelCase.
    
    Args:
        text: Texto en snake_case
        capitalize_first: Si True, capitaliza la primera letra
    
    Returns:
        Texto en camelCase
    """
    components = text.split('_')
    if capitalize_first:
        return ''.join(word.capitalize() for word in components)
    return components[0] + ''.join(word.capitalize() for word in components[1:])


def truncate(text: str, max_length: int, suffix: str = '...') -> str:
    """
    Truncar texto a longitud máxima.
    
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


def remove_whitespace(text: str, replace_with: str = ' ') -> str:
    """
    Remover espacios en blanco múltiples.
    
    Args:
        text: Texto
        replace_with: Reemplazar con (default: ' ')
    
    Returns:
        Texto sin espacios múltiples
    """
    return re.sub(r'\s+', replace_with, text).strip()


def extract_emails(text: str) -> List[str]:
    """
    Extraer emails de un texto.
    
    Args:
        text: Texto
    
    Returns:
        Lista de emails encontrados
    """
    pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    return re.findall(pattern, text)


def extract_urls(text: str) -> List[str]:
    """
    Extraer URLs de un texto.
    
    Args:
        text: Texto
    
    Returns:
        Lista de URLs encontradas
    """
    pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    return re.findall(pattern, text)


def mask_sensitive(text: str, visible_chars: int = 4, mask_char: str = '*') -> str:
    """
    Enmascarar texto sensible (ej: tarjetas, emails).
    
    Args:
        text: Texto a enmascarar
        visible_chars: Caracteres visibles al inicio
        mask_char: Carácter de máscara
    
    Returns:
        Texto enmascarado
    """
    if len(text) <= visible_chars:
        return mask_char * len(text)
    return text[:visible_chars] + mask_char * (len(text) - visible_chars)


def normalize_whitespace(text: str) -> str:
    """
    Normalizar espacios en blanco.
    
    Args:
        text: Texto
    
    Returns:
        Texto normalizado
    """
    return ' '.join(text.split())


def word_count(text: str) -> int:
    """
    Contar palabras en texto.
    
    Args:
        text: Texto
    
    Returns:
        Número de palabras
    """
    return len(text.split())


def contains_any(text: str, keywords: List[str], case_sensitive: bool = False) -> bool:
    """
    Verificar si texto contiene alguna keyword.
    
    Args:
        text: Texto
        keywords: Lista de keywords
        case_sensitive: Si True, case sensitive
    
    Returns:
        True si contiene alguna keyword
    """
    if not case_sensitive:
        text = text.lower()
        keywords = [k.lower() for k in keywords]
    return any(keyword in text for keyword in keywords)


def contains_all(text: str, keywords: List[str], case_sensitive: bool = False) -> bool:
    """
    Verificar si texto contiene todas las keywords.
    
    Args:
        text: Texto
        keywords: Lista de keywords
        case_sensitive: Si True, case sensitive
    
    Returns:
        True si contiene todas las keywords
    """
    if not case_sensitive:
        text = text.lower()
        keywords = [k.lower() for k in keywords]
    return all(keyword in text for keyword in keywords)


def sanitize_filename(filename: str, replacement: str = '_') -> str:
    """
    Sanitizar nombre de archivo.
    
    Args:
        filename: Nombre de archivo
        replacement: Carácter de reemplazo para caracteres inválidos
    
    Returns:
        Nombre sanitizado
    """
    invalid_chars = r'[<>:"/\\|?*]'
    sanitized = re.sub(invalid_chars, replacement, filename)
    sanitized = sanitized.strip('. ')
    return sanitized


def url_encode(text: str, safe: str = '') -> str:
    """
    Codificar texto para URL.
    
    Args:
        text: Texto
        safe: Caracteres seguros adicionales
    
    Returns:
        Texto codificado
    """
    return quote(text, safe=safe)


def url_decode(text: str) -> str:
    """
    Decodificar texto de URL.
    
    Args:
        text: Texto codificado
    
    Returns:
        Texto decodificado
    """
    return unquote(text)


def split_camel_case(text: str) -> List[str]:
    """
    Dividir camelCase en palabras.
    
    Args:
        text: Texto en camelCase
    
    Returns:
        Lista de palabras
    """
    matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', text)
    return [m.group(0) for m in matches]


def pluralize(word: str, count: int = 2) -> str:
    """
    Pluralizar palabra (básico).
    
    Args:
        word: Palabra
        count: Cantidad (si es 1, no pluraliza)
    
    Returns:
        Palabra pluralizada si count != 1
    """
    if count == 1:
        return word
    
    if word.endswith('y'):
        return word[:-1] + 'ies'
    elif word.endswith(('s', 'sh', 'ch', 'x', 'z')):
        return word + 'es'
    elif word.endswith('f'):
        return word[:-1] + 'ves'
    elif word.endswith('fe'):
        return word[:-2] + 'ves'
    else:
        return word + 's'

