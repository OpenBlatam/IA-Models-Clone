"""
Encoding Utils - Utilidades de Encoding/Decoding
=================================================

Utilidades para encoding y decoding de datos en diferentes formatos.
"""

import logging
import base64
import binascii
from typing import Union, Optional
from urllib.parse import quote, unquote, quote_plus, unquote_plus

logger = logging.getLogger(__name__)


def encode_base64(data: Union[str, bytes]) -> str:
    """
    Codificar datos a Base64.
    
    Args:
        data: Datos a codificar (string o bytes)
        
    Returns:
        String Base64
    """
    if isinstance(data, str):
        data = data.encode('utf-8')
    return base64.b64encode(data).decode('utf-8')


def decode_base64(encoded: str) -> bytes:
    """
    Decodificar datos de Base64.
    
    Args:
        encoded: String Base64
        
    Returns:
        Bytes decodificados
    """
    try:
        return base64.b64decode(encoded)
    except (binascii.Error, ValueError) as e:
        logger.error(f"Error decoding Base64: {e}")
        raise ValueError(f"Invalid Base64 string: {e}")


def encode_base64_to_string(encoded: str) -> str:
    """
    Decodificar Base64 a string.
    
    Args:
        encoded: String Base64
        
    Returns:
        String decodificado
    """
    return decode_base64(encoded).decode('utf-8')


def encode_url(text: str, safe: str = '/') -> str:
    """
    Codificar texto para URL.
    
    Args:
        text: Texto a codificar
        safe: Caracteres seguros (no codificar)
        
    Returns:
        Texto codificado
    """
    return quote(text, safe=safe)


def decode_url(encoded: str) -> str:
    """
    Decodificar texto de URL.
    
    Args:
        encoded: Texto codificado
        
    Returns:
        Texto decodificado
    """
    return unquote(encoded)


def encode_url_plus(text: str) -> str:
    """
    Codificar texto para URL (con + para espacios).
    
    Args:
        text: Texto a codificar
        
    Returns:
        Texto codificado
    """
    return quote_plus(text)


def decode_url_plus(encoded: str) -> str:
    """
    Decodificar texto de URL (con + para espacios).
    
    Args:
        encoded: Texto codificado
        
    Returns:
        Texto decodificado
    """
    return unquote_plus(encoded)


def encode_hex(data: Union[str, bytes]) -> str:
    """
    Codificar datos a hexadecimal.
    
    Args:
        data: Datos a codificar
        
    Returns:
        String hexadecimal
    """
    if isinstance(data, str):
        data = data.encode('utf-8')
    return data.hex()


def decode_hex(encoded: str) -> bytes:
    """
    Decodificar datos de hexadecimal.
    
    Args:
        encoded: String hexadecimal
        
    Returns:
        Bytes decodificados
    """
    try:
        return bytes.fromhex(encoded)
    except ValueError as e:
        logger.error(f"Error decoding hex: {e}")
        raise ValueError(f"Invalid hex string: {e}")


def encode_hex_to_string(encoded: str) -> str:
    """
    Decodificar hexadecimal a string.
    
    Args:
        encoded: String hexadecimal
        
    Returns:
        String decodificado
    """
    return decode_hex(encoded).decode('utf-8')


def encode_base64url(data: Union[str, bytes]) -> str:
    """
    Codificar datos a Base64URL (URL-safe).
    
    Args:
        data: Datos a codificar
        
    Returns:
        String Base64URL
    """
    if isinstance(data, str):
        data = data.encode('utf-8')
    encoded = base64.b64encode(data).decode('utf-8')
    # Reemplazar caracteres no seguros para URL
    return encoded.replace('+', '-').replace('/', '_').rstrip('=')


def decode_base64url(encoded: str) -> bytes:
    """
    Decodificar datos de Base64URL.
    
    Args:
        encoded: String Base64URL
        
    Returns:
        Bytes decodificados
    """
    try:
        # Restaurar caracteres
        encoded = encoded.replace('-', '+').replace('_', '/')
        # Agregar padding si es necesario
        padding = 4 - len(encoded) % 4
        if padding != 4:
            encoded += '=' * padding
        return base64.b64decode(encoded)
    except (binascii.Error, ValueError) as e:
        logger.error(f"Error decoding Base64URL: {e}")
        raise ValueError(f"Invalid Base64URL string: {e}")


def encode_html(text: str) -> str:
    """
    Codificar caracteres HTML especiales.
    
    Args:
        text: Texto a codificar
        
    Returns:
        Texto codificado
    """
    html_entities = {
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#x27;',
        '/': '&#x2F;'
    }
    
    result = text
    for char, entity in html_entities.items():
        result = result.replace(char, entity)
    
    return result


def decode_html(encoded: str) -> str:
    """
    Decodificar entidades HTML.
    
    Args:
        encoded: Texto con entidades HTML
        
    Returns:
        Texto decodificado
    """
    html_entities = {
        '&amp;': '&',
        '&lt;': '<',
        '&gt;': '>',
        '&quot;': '"',
        '&#x27;': "'",
        '&#x2F;': '/',
        '&nbsp;': ' '
    }
    
    result = encoded
    for entity, char in html_entities.items():
        result = result.replace(entity, char)
    
    return result


def encode_unicode_escape(text: str) -> str:
    """
    Codificar texto a Unicode escape.
    
    Args:
        text: Texto a codificar
        
    Returns:
        Texto con escapes Unicode
    """
    return text.encode('unicode_escape').decode('ascii')


def decode_unicode_escape(encoded: str) -> str:
    """
    Decodificar texto de Unicode escape.
    
    Args:
        encoded: Texto con escapes Unicode
        
    Returns:
        Texto decodificado
    """
    return encoded.encode('ascii').decode('unicode_escape')




