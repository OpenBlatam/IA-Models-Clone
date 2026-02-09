"""
Advanced Request Validator - Validador avanzado de requests
==========================================================
"""

import logging
import re
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ValidationRule:
    """Regla de validación"""
    field: str
    validators: List[Callable]
    required: bool = False
    error_message: Optional[str] = None


class AdvancedRequestValidator:
    """Validador avanzado de requests"""
    
    def __init__(self):
        self.rules: Dict[str, List[ValidationRule]] = {}  # endpoint -> rules
    
    def add_rule(self, endpoint: str, rule: ValidationRule):
        """Agrega una regla de validación"""
        if endpoint not in self.rules:
            self.rules[endpoint] = []
        self.rules[endpoint].append(rule)
    
    def validate(
        self,
        endpoint: str,
        data: Dict[str, Any]
    ) -> tuple[bool, List[str]]:
        """Valida datos según reglas"""
        if endpoint not in self.rules:
            return True, []
        
        errors = []
        
        for rule in self.rules[endpoint]:
            value = data.get(rule.field)
            
            # Verificar requerido
            if rule.required and (value is None or value == ""):
                error = rule.error_message or f"Field {rule.field} is required"
                errors.append(error)
                continue
            
            # Si no es requerido y está vacío, saltar validaciones
            if not rule.required and (value is None or value == ""):
                continue
            
            # Ejecutar validadores
            for validator in rule.validators:
                try:
                    if not validator(value):
                        error = rule.error_message or f"Field {rule.field} validation failed"
                        errors.append(error)
                        break
                except Exception as e:
                    errors.append(f"Validation error for {rule.field}: {str(e)}")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def is_email(value: str) -> bool:
        """Valida email"""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, value))
    
    @staticmethod
    def is_url(value: str) -> bool:
        """Valida URL"""
        pattern = r'^https?://.+'
        return bool(re.match(pattern, value))
    
    @staticmethod
    def min_length(min_len: int) -> Callable:
        """Validador de longitud mínima"""
        return lambda v: len(str(v)) >= min_len
    
    @staticmethod
    def max_length(max_len: int) -> Callable:
        """Validador de longitud máxima"""
        return lambda v: len(str(v)) <= max_len




