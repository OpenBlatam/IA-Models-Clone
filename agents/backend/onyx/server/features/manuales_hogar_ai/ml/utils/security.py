"""
Security
========

Utilidades de seguridad.
"""

import logging
import re
from typing import Optional, List

logger = logging.getLogger(__name__)


class SecurityUtils:
    """Utilidades de seguridad."""
    
    # Patrones peligrosos
    DANGEROUS_PATTERNS = [
        r'<script[^>]*>.*?</script>',  # XSS
        r'javascript:',  # JavaScript injection
        r'on\w+\s*=',  # Event handlers
        r'eval\s*\(',  # Eval
        r'exec\s*\(',  # Exec
        r'__import__',  # Import injection
    ]
    
    @staticmethod
    def sanitize_input(text: str) -> str:
        """
        Sanitizar input de usuario.
        
        Args:
            text: Texto a sanitizar
        
        Returns:
            Texto sanitizado
        """
        if not isinstance(text, str):
            return ""
        
        # Remover caracteres peligrosos
        sanitized = text
        
        # Remover patrones peligrosos
        for pattern in SecurityUtils.DANGEROUS_PATTERNS:
            sanitized = re.sub(pattern, '', sanitized, flags=re.IGNORECASE | re.DOTALL)
        
        # Limitar longitud
        sanitized = sanitized[:10000]
        
        return sanitized
    
    @staticmethod
    def validate_prompt(prompt: str, max_length: int = 5000) -> bool:
        """
        Validar prompt de forma segura.
        
        Args:
            prompt: Prompt a validar
            max_length: Longitud máxima
        
        Returns:
            True si seguro
        """
        if not isinstance(prompt, str):
            return False
        
        if len(prompt) > max_length:
            logger.warning(f"Prompt muy largo: {len(prompt)}")
            return False
        
        # Verificar patrones peligrosos
        for pattern in SecurityUtils.DANGEROUS_PATTERNS:
            if re.search(pattern, prompt, re.IGNORECASE):
                logger.warning(f"Prompt contiene patrón peligroso: {pattern}")
                return False
        
        return True
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """
        Sanitizar nombre de archivo.
        
        Args:
            filename: Nombre de archivo
        
        Returns:
            Nombre sanitizado
        """
        # Remover caracteres peligrosos
        sanitized = re.sub(r'[^\w\-_\.]', '', filename)
        # Limitar longitud
        sanitized = sanitized[:255]
        return sanitized

