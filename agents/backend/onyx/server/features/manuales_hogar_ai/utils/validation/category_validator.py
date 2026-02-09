"""
Category Validator
=================

Validador especializado para categorías.
"""

from typing import Tuple, Optional
from ...core.base.service_base import BaseService


class CategoryValidator(BaseService):
    """Validador para categorías."""
    
    VALID_CATEGORIES = [
        "plomeria", "techos", "carpinteria", "electricidad",
        "albanileria", "pintura", "herreria", "jardineria", "general"
    ]
    
    def __init__(self):
        """Inicializar validador."""
        super().__init__(logger_name=__name__)
    
    def validate(self, category: str) -> Tuple[bool, Optional[str]]:
        """
        Validar categoría.
        
        Args:
            category: Categoría a validar
        
        Returns:
            Tuple de (es_válida, mensaje_error)
        """
        if category.lower() not in self.VALID_CATEGORIES:
            return False, f"Categoría no válida. Categorías válidas: {', '.join(self.VALID_CATEGORIES)}"
        
        return True, None

