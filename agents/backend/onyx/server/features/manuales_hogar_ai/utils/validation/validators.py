"""
Validators
==========

Validador principal que compone todos los validadores especializados.
"""

from typing import Tuple, Optional
from datetime import datetime

from ...core.base.service_base import BaseService
from .category_validator import CategoryValidator
from .text_validator import TextValidator
from .user_validator import UserValidator
from .date_validator import DateValidator


class Validators(BaseService):
    """Validador principal que compone validadores especializados."""
    
    def __init__(self):
        """Inicializar validadores."""
        super().__init__(logger_name=__name__)
        self.category_validator = CategoryValidator()
        self.text_validator = TextValidator()
        self.user_validator = UserValidator()
        self.date_validator = DateValidator()
    
    def validate_category(self, category: str) -> Tuple[bool, Optional[str]]:
        """Validar categoría."""
        return self.category_validator.validate(category)
    
    def validate_difficulty(self, difficulty: Optional[str]) -> Tuple[bool, Optional[str]]:
        """
        Validar dificultad.
        
        Args:
            difficulty: Dificultad a validar
        
        Returns:
            Tuple de (es_válida, mensaje_error)
        """
        if difficulty is None:
            return True, None
        
        valid_difficulties = ["Fácil", "Media", "Difícil"]
        
        if difficulty not in valid_difficulties:
            return False, f"Dificultad no válida. Valores válidos: {', '.join(valid_difficulties)}"
        
        return True, None
    
    def validate_rating(self, rating: int) -> Tuple[bool, Optional[str]]:
        """
        Validar rating.
        
        Args:
            rating: Rating a validar (1-5)
        
        Returns:
            Tuple de (es_válida, mensaje_error)
        """
        if not isinstance(rating, int):
            return False, "Rating debe ser un número entero"
        
        if rating < 1 or rating > 5:
            return False, "Rating debe estar entre 1 y 5"
        
        return True, None
    
    def validate_problem_description(self, description: str) -> Tuple[bool, Optional[str]]:
        """Validar descripción del problema."""
        return self.text_validator.validate_problem_description(description)
    
    def validate_user_id(self, user_id: Optional[str]) -> Tuple[bool, Optional[str]]:
        """Validar ID de usuario."""
        return self.user_validator.validate_user_id(user_id)
    
    def validate_date_range(
        self,
        date_from: Optional[datetime],
        date_to: Optional[datetime]
    ) -> Tuple[bool, Optional[str]]:
        """Validar rango de fechas."""
        return self.date_validator.validate_date_range(date_from, date_to)
    
    def sanitize_text(self, text: str) -> str:
        """Sanitizar texto."""
        return self.text_validator.sanitize(text)

