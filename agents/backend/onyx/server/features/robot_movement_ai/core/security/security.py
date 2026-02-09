"""
Security Utilities
==================

Utilidades de seguridad para el sistema.
"""

import hashlib
import hmac
import secrets
from typing import Optional, Dict, Any
import base64
import logging

logger = logging.getLogger(__name__)


def generate_api_key(length: int = 32) -> str:
    """
    Generar API key segura.
    
    Args:
        length: Longitud de la key
        
    Returns:
        API key generada
    """
    return secrets.token_urlsafe(length)


def hash_password(password: str, salt: Optional[str] = None) -> tuple:
    """
    Hashear contraseña con salt.
    
    Args:
        password: Contraseña a hashear
        salt: Salt opcional (se genera si None)
        
    Returns:
        Tupla (hash, salt)
    """
    if salt is None:
        salt = secrets.token_hex(16)
    
    # Usar PBKDF2
    import hashlib
    key = hashlib.pbkdf2_hmac(
        'sha256',
        password.encode('utf-8'),
        salt.encode('utf-8'),
        100000  # Iteraciones
    )
    
    hash_str = base64.b64encode(key).decode('utf-8')
    return hash_str, salt


def verify_password(password: str, hash_str: str, salt: str) -> bool:
    """
    Verificar contraseña.
    
    Args:
        password: Contraseña a verificar
        hash_str: Hash almacenado
        salt: Salt usado
        
    Returns:
        True si la contraseña es correcta
    """
    computed_hash, _ = hash_password(password, salt)
    return hmac.compare_digest(computed_hash, hash_str)


def generate_token(data: Dict[str, Any], secret: str) -> str:
    """
    Generar token firmado.
    
    Args:
        data: Datos a incluir en el token
        secret: Secreto para firmar
        
    Returns:
        Token generado
    """
    import json
    payload = json.dumps(data, sort_keys=True)
    signature = hmac.new(
        secret.encode('utf-8'),
        payload.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    
    token_data = {
        "payload": payload,
        "signature": signature
    }
    
    return base64.b64encode(json.dumps(token_data).encode('utf-8')).decode('utf-8')


def verify_token(token: str, secret: str) -> Optional[Dict[str, Any]]:
    """
    Verificar token firmado.
    
    Args:
        token: Token a verificar
        secret: Secreto para verificar
        
    Returns:
        Datos del token si es válido, None en caso contrario
    """
    try:
        import json
        token_data = json.loads(base64.b64decode(token).decode('utf-8'))
        payload = token_data["payload"]
        signature = token_data["signature"]
        
        # Verificar firma
        computed_signature = hmac.new(
            secret.encode('utf-8'),
            payload.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        if not hmac.compare_digest(computed_signature, signature):
            return None
        
        return json.loads(payload)
    
    except Exception as e:
        logger.error(f"Error verifying token: {e}")
        return None


def sanitize_input(input_str: str, max_length: int = 1000) -> str:
    """
    Sanitizar input de usuario.
    
    Args:
        input_str: String a sanitizar
        max_length: Longitud máxima
        
    Returns:
        String sanitizado
    """
    # Limitar longitud
    if len(input_str) > max_length:
        input_str = input_str[:max_length]
    
    # Remover caracteres peligrosos básicos
    dangerous_chars = ['<', '>', '"', "'", '&']
    for char in dangerous_chars:
        input_str = input_str.replace(char, '')
    
    return input_str.strip()


class RateLimiter:
    """
    Rate limiter simple.
    
    Limita número de requests por tiempo.
    """
    
    def __init__(self, max_requests: int = 100, window_seconds: float = 60.0):
        """
        Inicializar rate limiter.
        
        Args:
            max_requests: Máximo de requests
            window_seconds: Ventana de tiempo en segundos
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: Dict[str, list] = {}  # {key: [timestamps]}
    
    def is_allowed(self, key: str) -> bool:
        """
        Verificar si request está permitido.
        
        Args:
            key: Clave única (ej: IP, user_id)
            
        Returns:
            True si está permitido
        """
        import time
        current_time = time.time()
        
        if key not in self.requests:
            self.requests[key] = []
        
        # Limpiar requests antiguos
        cutoff_time = current_time - self.window_seconds
        self.requests[key] = [
            t for t in self.requests[key]
            if t > cutoff_time
        ]
        
        # Verificar límite
        if len(self.requests[key]) >= self.max_requests:
            return False
        
        # Agregar request actual
        self.requests[key].append(current_time)
        return True
    
    def reset(self, key: Optional[str] = None) -> None:
        """
        Resetear rate limiter.
        
        Args:
            key: Clave específica (None = todas)
        """
        if key:
            self.requests.pop(key, None)
        else:
            self.requests.clear()






