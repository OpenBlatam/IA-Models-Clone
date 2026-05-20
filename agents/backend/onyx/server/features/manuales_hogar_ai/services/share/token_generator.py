"""
Token Generator
==============

Generador especializado de tokens para compartir.
"""

import secrets
from ...core.base.service_base import BaseService


class TokenGenerator(BaseService):
    """Generador de tokens únicos para compartir."""
    
    def __init__(self):
        """Inicializar generador."""
        super().__init__(logger_name=__name__)
    
    def generate(self) -> str:
        """
        Generar token único para compartir.
        
        Returns:
            Token único
        """
        return secrets.token_urlsafe(32)

