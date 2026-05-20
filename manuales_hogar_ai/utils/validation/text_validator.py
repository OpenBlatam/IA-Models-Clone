"""
Text Validator
=============

Validador especializado para texto.
"""

import re
from typing import Tuple, Optional
from ...core.base.service_base import BaseService


class TextValidator(BaseService):
    """Validador para texto."""
    
    DANGEROUS_PATTERNS = [
        r'<script',
        r'javascript:',
        r'onerror=',
        r'onload='
    ]
    
    def __init__(self):
        """Inicializar validador."""
        super().__init__(logger_name=__name__)
    
    def validate_problem_description(self, description: str) -> Tuple[bool, Optional[str]]:
        """
        Validar descripción del problema.
        
        Args:
            description: Descripción a validar
        
        Returns:
            Tuple de (es_válida, mensaje_error)
        """
        if not description or not description.strip():
            return False, "La descripción del problema no puede estar vacía"
        
        if len(description) < 10:
            return False, "La descripción debe tener al menos 10 caracteres"
        
        if len(description) > 5000:
            return False, "La descripción no puede exceder 5000 caracteres"
        
        for pattern in self.DANGEROUS_PATTERNS:
            if re.search(pattern, description, re.IGNORECASE):
                return False, "La descripción contiene contenido no permitido"
        
        return True, None
    
    def sanitize(self, text: str) -> str:
        """
        Sanitizar texto removiendo caracteres peligrosos.
        
        Args:
            text: Texto a sanitizar
        
        Returns:
            Texto sanitizado
        """
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(r'on\w+\s*=', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

