"""
Utilidades de seguridad (optimizado)

Incluye funciones para sanitización, validación de inputs, y protección contra ataques comunes.
"""

import re
import html
from typing import Optional, List
from urllib.parse import quote, unquote


def sanitize_html(text: str, max_length: Optional[int] = None) -> str:
    """
    Sanitiza HTML escapando caracteres peligrosos (optimizado).
    
    Args:
        text: Texto a sanitizar
        max_length: Longitud máxima (opcional)
        
    Returns:
        Texto sanitizado
    """
    if not text:
        return ""
    
    # Escapar HTML
    sanitized = html.escape(text)
    
    # Limitar longitud
    if max_length and len(sanitized) > max_length:
        sanitized = sanitized[:max_length]
    
    return sanitized


def sanitize_sql_input(text: str) -> str:
    """
    Sanitiza input para prevenir SQL injection (optimizado).
    
    Nota: Esto es una medida adicional. Siempre usar parámetros preparados.
    
    Args:
        text: Texto a sanitizar
        
    Returns:
        Texto sanitizado
    """
    if not text:
        return ""
    
    # Remover caracteres peligrosos
    dangerous_chars = ["'", '"', ';', '--', '/*', '*/', 'xp_', 'sp_']
    sanitized = text
    
    for char in dangerous_chars:
        sanitized = sanitized.replace(char, '')
    
    return sanitized.strip()


def validate_email(email: str) -> bool:
    """
    Valida formato de email (optimizado).
    
    Args:
        email: Email a validar
        
    Returns:
        True si es válido, False en caso contrario
    """
    if not email:
        return False
    
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))


def validate_url(url: str, allowed_schemes: Optional[List[str]] = None) -> bool:
    """
    Valida formato de URL (optimizado).
    
    Args:
        url: URL a validar
        allowed_schemes: Lista de esquemas permitidos (http, https, etc.)
        
    Returns:
        True si es válido, False en caso contrario
    """
    if not url:
        return False
    
    if allowed_schemes is None:
        allowed_schemes = ["http", "https"]
    
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        
        if not parsed.scheme:
            return False
        
        if parsed.scheme not in allowed_schemes:
            return False
        
        if not parsed.netloc:
            return False
        
        return True
    except Exception:
        return False


def sanitize_filename(filename: str) -> str:
    """
    Sanitiza un nombre de archivo (optimizado).
    
    Args:
        filename: Nombre de archivo a sanitizar
        
    Returns:
        Nombre de archivo sanitizado
    """
    if not filename:
        return "file"
    
    # Remover caracteres peligrosos
    dangerous_chars = ['/', '\\', '..', '<', '>', ':', '"', '|', '?', '*']
    sanitized = filename
    
    for char in dangerous_chars:
        sanitized = sanitized.replace(char, '_')
    
    # Limitar longitud
    if len(sanitized) > 255:
        sanitized = sanitized[:255]
    
    return sanitized.strip()


def rate_limit_key(user_id: str, endpoint: str) -> str:
    """
    Genera una clave de rate limiting (optimizado).
    
    Args:
        user_id: ID del usuario
        endpoint: Endpoint o acción
        
    Returns:
        Clave de rate limiting
    """
    return f"rate_limit:{user_id}:{endpoint}"


def generate_csrf_token() -> str:
    """
    Genera un token CSRF simple (optimizado).
    
    Nota: En producción, usar una librería especializada.
    
    Returns:
        Token CSRF
    """
    import secrets
    return secrets.token_urlsafe(32)


def validate_csrf_token(token: str, expected_token: str) -> bool:
    """
    Valida un token CSRF (optimizado).
    
    Args:
        token: Token a validar
        expected_token: Token esperado
        
    Returns:
        True si es válido, False en caso contrario
    """
    import secrets
    
    if not token or not expected_token:
        return False
    
    return secrets.compare_digest(token, expected_token)


def generate_hash(value: str, algorithm: str = "sha256") -> str:
    """
    Generates a hash of a string.
    
    Args:
        value: String to hash
        algorithm: Hash algorithm to use (md5, sha1, sha256, sha512)
        
    Returns:
        Hexadecimal hash string
    """
    import hashlib
    
    if not value:
        return ""
    
    hash_obj = hashlib.new(algorithm)
    hash_obj.update(value.encode('utf-8'))
    return hash_obj.hexdigest()


def mask_email(email: str) -> str:
    """
    Masks an email for privacy.
    
    Args:
        email: Email to mask
        
    Returns:
        Masked email (e.g., "u***@example.com")
    """
    if not email or '@' not in email:
        return email
    
    local, domain = email.split('@', 1)
    
    if len(local) <= 2:
        masked_local = local[0] + '*'
    else:
        masked_local = local[0] + '*' * (len(local) - 2) + local[-1]
    
    return f"{masked_local}@{domain}"
