"""
Validación de URLs
"""

import re
from urllib.parse import urlparse
from typing import Tuple, Optional


class URLValidator:
    """Validador de URLs con validaciones de seguridad"""
    
    ALLOWED_SCHEMES = {'http', 'https'}
    BLOCKED_DOMAINS = {
        'localhost',
        '127.0.0.1',
        '0.0.0.0',
        '::1'
    }
    
    @staticmethod
    def validate(url: str) -> Tuple[bool, Optional[str]]:
        """
        Validar URL
        
        Returns:
            Tuple[bool, Optional[str]]: (es_válida, mensaje_error)
        """
        if not url or not isinstance(url, str):
            return False, "URL no puede estar vacía"
        
        url = url.strip()
        
        # Validar formato básico
        try:
            parsed = urlparse(url)
        except Exception:
            return False, "URL con formato inválido"
        
        # Validar scheme
        if parsed.scheme not in URLValidator.ALLOWED_SCHEMES:
            return False, f"Scheme no permitido. Solo se permiten: {', '.join(URLValidator.ALLOWED_SCHEMES)}"
        
        # Validar que tenga netloc
        if not parsed.netloc:
            return False, "URL debe incluir dominio"
        
        # Validar que no sea localhost (seguridad)
        domain = parsed.netloc.split(':')[0].lower()
        if domain in URLValidator.BLOCKED_DOMAINS:
            return False, "No se permiten URLs locales por seguridad"
        
        # Validar longitud
        if len(url) > 2048:
            return False, "URL demasiado larga (máximo 2048 caracteres)"
        
        return True, None
    
    @staticmethod
    def normalize(url: str) -> str:
        """Normalizar URL"""
        url = url.strip()
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        return url








