"""
Security Utilities - Utilidades de Seguridad
=============================================

Utilidades para mejorar la seguridad del sistema.
"""

import re
import logging
from typing import Optional, List, Set, Tuple
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class SecurityValidator:
    """
    Validador de seguridad para comandos y URLs.
    """
    
    def __init__(
        self,
        allowed_domains: Optional[List[str]] = None,
        blocked_domains: Optional[List[str]] = None,
        max_url_length: Optional[int] = None
    ):
        """
        Inicializar validador de seguridad.
        
        Args:
            allowed_domains: Lista de dominios permitidos para API calls
            blocked_domains: Lista de dominios bloqueados
            max_url_length: Longitud máxima de URL
        """
        from .constants import MAX_URL_LENGTH
        
        self.allowed_domains = set(allowed_domains or [])
        self.blocked_domains = set(blocked_domains or [
            "localhost",
            "127.0.0.1",
            "0.0.0.0",
            "::1"
        ])
        self.max_url_length = max_url_length if max_url_length is not None else MAX_URL_LENGTH
        
        # Patrones peligrosos adicionales
        self.dangerous_patterns = [
            r'rm\s+-rf\s+/',
            r'dd\s+if=.*of=/dev/',
            r':\(\)\{.*\|\s*&\s*\};',
            r'mkfs\.',
            r'fdisk\s+/dev/',
            r'format\s+[c-z]:',
            r'del\s+/f\s+/s\s+/q\s+[c-z]:',
            r'__import__\s*\(',
            r'eval\s*\(',
            r'exec\s*\(',
            r'compile\s*\(',
            r'open\s*\([^)]*[\'"]/etc/',
            r'open\s*\([^)]*[\'"]/proc/',
            r'open\s*\([^)]*[\'"]/sys/',
        ]
    
    def validate_url(self, url: str) -> Tuple[bool, Optional[str]]:
        """
        Validar URL para API calls.
        
        Args:
            url: URL a validar
            
        Returns:
            Tupla (es_válida, mensaje_error)
        """
        if not url or not isinstance(url, str):
            return False, "URL must be a non-empty string"
        
        if len(url) > self.max_url_length:
            return False, f"URL exceeds maximum length of {self.max_url_length}"
        
        try:
            parsed = urlparse(url)
            
            # Solo permitir HTTP/HTTPS
            if parsed.scheme not in ['http', 'https']:
                return False, "Only HTTP and HTTPS URLs are allowed"
            
            # Validar dominio
            hostname = parsed.hostname
            if not hostname:
                return False, "URL must have a valid hostname"
            
            # Verificar dominios bloqueados
            if any(blocked in hostname.lower() for blocked in self.blocked_domains):
                return False, f"Domain {hostname} is not allowed"
            
            # Si hay dominios permitidos, verificar
            if self.allowed_domains:
                if not any(allowed in hostname.lower() for allowed in self.allowed_domains):
                    return False, f"Domain {hostname} is not in allowed list"
            
            return True, None
            
        except Exception as e:
            logger.debug(f"URL validation error: {e}")
            return False, "Invalid URL format"
    
    def sanitize_command(self, command: str) -> str:
        """
        Sanitizar comando removiendo caracteres peligrosos.
        
        Args:
            command: Comando a sanitizar
            
        Returns:
            Comando sanitizado
        """
        if not command:
            return ""
        
        # Remover caracteres de control
        sanitized = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]', '', command)
        
        # Remover caracteres peligrosos pero mantener funcionalidad básica
        sanitized = sanitized.strip()
        
        return sanitized
    
    def is_dangerous_command(self, command: str) -> Tuple[bool, Optional[str]]:
        """
        Verificar si un comando es peligroso.
        
        Args:
            command: Comando a verificar
            
        Returns:
            Tupla (es_peligroso, razón)
        """
        if not command:
            return False, None
        
        command_lower = command.lower()
        
        # Verificar patrones peligrosos
        for pattern in self.dangerous_patterns:
            if re.search(pattern, command_lower, re.IGNORECASE):
                return True, f"Command matches dangerous pattern: {pattern}"
        
        return False, None
    
    def sanitize_error_message(self, error_msg: str, max_length: int = 200) -> str:
        """
        Sanitizar mensaje de error para no exponer información sensible.
        
        Args:
            error_msg: Mensaje de error original
            max_length: Longitud máxima del mensaje
            
        Returns:
            Mensaje de error sanitizado
        """
        if not error_msg:
            return "An error occurred"
        
        # Remover paths absolutos
        sanitized = re.sub(r'/[a-zA-Z0-9_\-/\.]+', '[path]', error_msg)
        
        # Remover información de usuario
        sanitized = re.sub(r'/home/[a-zA-Z0-9_\-]+', '[home]', sanitized)
        sanitized = re.sub(r'C:\\Users\\[a-zA-Z0-9_\-]+', '[user]', sanitized)
        
        # Remover tokens y keys
        sanitized = re.sub(r'[a-zA-Z0-9]{32,}', '[token]', sanitized)
        
        # Limitar longitud
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length] + "..."
        
        return sanitized


def create_secure_temp_file(prefix: str = "temp_", suffix: str = ".py") -> str:
    """
    Crear nombre de archivo temporal seguro.
    
    Args:
        prefix: Prefijo del archivo
        suffix: Sufijo del archivo
        
    Returns:
        Nombre de archivo seguro
    """
    import tempfile
    import os
    
    # Crear directorio temporal seguro
    temp_dir = tempfile.gettempdir()
    
    # Generar nombre único
    import uuid
    unique_id = uuid.uuid4().hex[:12]
    
    filename = f"{prefix}{unique_id}{suffix}"
    filepath = os.path.join(temp_dir, filename)
    
    return filepath

