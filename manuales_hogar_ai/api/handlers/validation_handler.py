"""
Validation Handler
=================

Handler para validaciones de requests.
"""

import logging
from typing import Optional
from fastapi import HTTPException

from ...config.settings import get_settings

logger = logging.getLogger(__name__)


class ValidationHandler:
    """Handler para validaciones."""
    
    def __init__(self):
        """Inicializar handler."""
        self.settings = get_settings()
        self._logger = logger
    
    def validate_category(self, category: str) -> None:
        """
        Validar categoría.
        
        Args:
            category: Categoría a validar
        
        Raises:
            HTTPException: Si la categoría no es válida
        """
        if category not in self.settings.supported_categories:
            raise HTTPException(
                status_code=400,
                detail=f"Categoría no soportada. Categorías válidas: {', '.join(self.settings.supported_categories)}"
            )

