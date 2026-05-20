"""
Security Helpers - Utilidades de seguridad
===========================================

Funciones helper para mejorar la seguridad del servidor MCP.
"""

import logging
import re
import hashlib
import hmac
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


def validate_origin(origin: str, allowed_origins: List[str]) -> bool:
    """
    Validar origen contra lista de orígenes permitidos.
    
    Args:
        origin: Origen a validar
        allowed_origins: Lista de orígenes permitidos
    
    Returns:
        True si el origen está permitido
    """
    if "*" in allowed_origins:
        return True
    
    if origin in allowed_origins:
        return True
    
    # Validar patrones wildcard
    for allowed in allowed_origins:
        if "*" in allowed:
            pattern = allowed.replace("*", ".*")
            if re.match(pattern, origin):
                return True
    
    return False


def sanitize_input(
    value: str,
    max_length: Optional[int] = None,
    allowed_chars: Optional[str] = None
) -> str:
    """
    Sanitizar input de usuario.
    
    Args:
        value: Valor a sanitizar
        max_length: Longitud máxima (opcional)
        allowed_chars: Caracteres permitidos (regex, opcional)
    
    Returns:
        String sanitizado
    """
    if not isinstance(value, str):
        value = str(value)
    
    # Limpiar espacios
    value = value.strip()
    
    # Aplicar longitud máxima
    if max_length and len(value) > max_length:
        value = value[:max_length]
    
    # Filtrar caracteres permitidos
    if allowed_chars:
        value = re.sub(f"[^{allowed_chars}]", "", value)
    
    return value


def validate_url(url: str, allowed_schemes: Optional[List[str]] = None) -> bool:
    """
    Validar URL.
    
    Args:
        url: URL a validar
        allowed_schemes: Esquemas permitidos (opcional)
    
    Returns:
        True si la URL es válida
    """
    try:
        parsed = urlparse(url)
        
        if not parsed.scheme:
            return False
        
        if allowed_schemes and parsed.scheme not in allowed_schemes:
            return False
        
        if not parsed.netloc:
            return False
        
        return True
    except Exception:
        return False


def generate_hmac_signature(
    data: str,
    secret: str,
    algorithm: str = "sha256"
) -> str:
    """
    Generar firma HMAC.
    
    Args:
        data: Datos a firmar
        secret: Secreto para la firma
        algorithm: Algoritmo (sha256, sha512, etc.)
    
    Returns:
        Firma HMAC hexadecimal
    """
    if algorithm == "sha256":
        return hmac.new(
            secret.encode('utf-8'),
            data.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
    elif algorithm == "sha512":
        return hmac.new(
            secret.encode('utf-8'),
            data.encode('utf-8'),
            hashlib.sha512
        ).hexdigest()
    else:
        raise ValueError(f"Unsupported HMAC algorithm: {algorithm}")


def verify_hmac_signature(
    data: str,
    signature: str,
    secret: str,
    algorithm: str = "sha256"
) -> bool:
    """
    Verificar firma HMAC.
    
    Args:
        data: Datos originales
        signature: Firma a verificar
        secret: Secreto
        algorithm: Algoritmo usado
    
    Returns:
        True si la firma es válida
    """
    expected = generate_hmac_signature(data, secret, algorithm)
    return hmac.compare_digest(expected, signature)


def mask_sensitive_data(
    data: Any,
    fields: Optional[List[str]] = None,
    mask_char: str = "*",
    visible_chars: int = 4
) -> Any:
    """
    Enmascarar datos sensibles.
    
    Args:
        data: Datos a enmascarar (dict, list, o string)
        fields: Lista de campos a enmascarar (opcional)
        mask_char: Carácter de enmascaramiento
        visible_chars: Número de caracteres visibles
    
    Returns:
        Datos con campos sensibles enmascarados
    """
    sensitive_fields = fields or [
        "password", "secret", "token", "key", "api_key",
        "access_token", "refresh_token", "authorization"
    ]
    
    if isinstance(data, dict):
        result = {}
        for key, value in data.items():
            if any(sensitive in key.lower() for sensitive in sensitive_fields):
                if isinstance(value, str) and len(value) > visible_chars:
                    result[key] = value[:visible_chars] + mask_char * (len(value) - visible_chars)
                else:
                    result[key] = mask_char * 8
            else:
                result[key] = mask_sensitive_data(value, fields, mask_char, visible_chars)
        return result
    
    if isinstance(data, list):
        return [mask_sensitive_data(item, fields, mask_char, visible_chars) for item in data]
    
    return data


def rate_limit_key(
    identifier: str,
    endpoint: Optional[str] = None,
    prefix: str = "rate_limit"
) -> str:
    """
    Generar key para rate limiting.
    
    Args:
        identifier: Identificador (user_id, IP, etc.)
        endpoint: Endpoint específico (opcional)
        prefix: Prefijo para la key
    
    Returns:
        Key para rate limiting
    """
    if endpoint:
        return f"{prefix}:{identifier}:{endpoint}"
    return f"{prefix}:{identifier}"


def validate_csrf_token(
    token: str,
    session_token: str,
    secret: str
) -> bool:
    """
    Validar token CSRF.
    
    Args:
        token: Token CSRF del request
        session_token: Token de sesión
        secret: Secreto compartido
    
    Returns:
        True si el token es válido
    """
    expected = generate_hmac_signature(session_token, secret)
    return hmac.compare_digest(expected, token)


def generate_csrf_token(session_token: str, secret: str) -> str:
    """
    Generar token CSRF.
    
    Args:
        session_token: Token de sesión
        secret: Secreto compartido
    
    Returns:
        Token CSRF
    """
    return generate_hmac_signature(session_token, secret)


def check_password_strength(password: str) -> Dict[str, Any]:
    """
    Verificar fortaleza de contraseña.
    
    Args:
        password: Contraseña a verificar
    
    Returns:
        Diccionario con análisis de fortaleza
    """
    score = 0
    feedback = []
    
    if len(password) >= 8:
        score += 1
    else:
        feedback.append("Password should be at least 8 characters")
    
    if re.search(r"[a-z]", password):
        score += 1
    else:
        feedback.append("Password should contain lowercase letters")
    
    if re.search(r"[A-Z]", password):
        score += 1
    else:
        feedback.append("Password should contain uppercase letters")
    
    if re.search(r"\d", password):
        score += 1
    else:
        feedback.append("Password should contain numbers")
    
    if re.search(r"[!@#$%^&*(),.?\":{}|<>]", password):
        score += 1
    else:
        feedback.append("Password should contain special characters")
    
    strength = "weak"
    if score >= 4:
        strength = "strong"
    elif score >= 3:
        strength = "medium"
    
    return {
        "score": score,
        "max_score": 5,
        "strength": strength,
        "feedback": feedback
    }

