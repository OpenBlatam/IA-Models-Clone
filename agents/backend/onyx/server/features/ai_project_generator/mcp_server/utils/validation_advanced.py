"""
Advanced Validation - Sistema de validación avanzado
=====================================================

Sistema de validación composable con esquemas, reglas personalizadas,
y transformaciones de datos.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Callable, Union, Tuple
from datetime import datetime
from decimal import Decimal

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Excepción de validación."""
    
    def __init__(self, message: str, field: Optional[str] = None, value: Any = None):
        """
        Inicializar error de validación.
        
        Args:
            message: Mensaje de error
            field: Campo que falló (opcional)
            value: Valor que falló (opcional)
        """
        super().__init__(message)
        self.message = message
        self.field = field
        self.value = value


class Validator:
    """
    Validador base composable.
    
    Permite crear validadores personalizados y combinarlos.
    """
    
    def __init__(self, name: Optional[str] = None):
        """
        Inicializar validador.
        
        Args:
            name: Nombre del validador (opcional)
        """
        self.name = name or self.__class__.__name__
        self._rules: List[Callable[[Any], Optional[str]]] = []
    
    def add_rule(self, rule: Callable[[Any], Optional[str]]) -> 'Validator':
        """
        Agregar regla de validación.
        
        Args:
            rule: Función que retorna None si es válido, mensaje de error si no
        
        Returns:
            Self para chaining
        
        Example:
            validator = Validator().add_rule(lambda x: None if x > 0 else "Must be positive")
        """
        self._rules.append(rule)
        return self
    
    def validate(self, value: Any) -> Tuple[bool, Optional[str]]:
        """
        Validar valor.
        
        Args:
            value: Valor a validar
        
        Returns:
            Tupla (is_valid, error_message)
        """
        for rule in self._rules:
            error = rule(value)
            if error:
                return False, error
        return True, None
    
    def __call__(self, value: Any) -> Any:
        """
        Validar valor (permite usar como función).
        
        Args:
            value: Valor a validar
        
        Returns:
            Valor validado
        
        Raises:
            ValidationError: Si la validación falla
        """
        is_valid, error = self.validate(value)
        if not is_valid:
            raise ValidationError(error or "Validation failed", value=value)
        return value


class SchemaValidator:
    """
    Validador de esquemas.
    
    Valida diccionarios contra esquemas definidos.
    """
    
    def __init__(self, schema: Dict[str, Any]):
        """
        Inicializar validador de esquema.
        
        Args:
            schema: Esquema de validación
        
        Example:
            schema = {
                "name": {"type": str, "required": True, "min_length": 3},
                "age": {"type": int, "required": True, "min": 0, "max": 120}
            }
            validator = SchemaValidator(schema)
        """
        self.schema = schema
    
    def validate(self, data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validar datos contra esquema.
        
        Args:
            data: Datos a validar
        
        Returns:
            Tupla (is_valid, errors)
        """
        errors = []
        
        for field, rules in self.schema.items():
            value = data.get(field)
            
            # Verificar requerido
            if rules.get('required', False) and value is None:
                errors.append(f"{field} is required")
                continue
            
            if value is None:
                continue
            
            # Validar tipo
            expected_type = rules.get('type')
            if expected_type and not isinstance(value, expected_type):
                errors.append(f"{field} must be {expected_type.__name__}")
                continue
            
            # Validar longitud (strings)
            if isinstance(value, str):
                if 'min_length' in rules and len(value) < rules['min_length']:
                    errors.append(f"{field} must be at least {rules['min_length']} characters")
                if 'max_length' in rules and len(value) > rules['max_length']:
                    errors.append(f"{field} must be at most {rules['max_length']} characters")
                if 'pattern' in rules:
                    if not re.match(rules['pattern'], value):
                        errors.append(f"{field} does not match required pattern")
            
            # Validar rango (números)
            if isinstance(value, (int, float, Decimal)):
                if 'min' in rules and value < rules['min']:
                    errors.append(f"{field} must be >= {rules['min']}")
                if 'max' in rules and value > rules['max']:
                    errors.append(f"{field} must be <= {rules['max']}")
            
            # Validar valores permitidos
            if 'choices' in rules and value not in rules['choices']:
                errors.append(f"{field} must be one of {rules['choices']}")
            
            # Validar con función personalizada
            if 'validator' in rules:
                custom_error = rules['validator'](value)
                if custom_error:
                    errors.append(f"{field}: {custom_error}")
        
        return len(errors) == 0, errors


# Validadores predefinidos

def required() -> Validator:
    """Validador para campos requeridos."""
    return Validator("required").add_rule(
        lambda x: None if x is not None else "Field is required"
    )


def not_empty() -> Validator:
    """Validador para campos no vacíos."""
    return Validator("not_empty").add_rule(
        lambda x: None if x and (not isinstance(x, str) or x.strip()) else "Field cannot be empty"
    )


def min_length(length: int) -> Validator:
    """Validador para longitud mínima."""
    return Validator(f"min_length_{length}").add_rule(
        lambda x: None if not isinstance(x, str) or len(x) >= length else f"Must be at least {length} characters"
    )


def max_length(length: int) -> Validator:
    """Validador para longitud máxima."""
    return Validator(f"max_length_{length}").add_rule(
        lambda x: None if not isinstance(x, str) or len(x) <= length else f"Must be at most {length} characters"
    )


def min_value(min_val: Union[int, float]) -> Validator:
    """Validador para valor mínimo."""
    return Validator(f"min_value_{min_val}").add_rule(
        lambda x: None if not isinstance(x, (int, float)) or x >= min_val else f"Must be >= {min_val}"
    )


def max_value(max_val: Union[int, float]) -> Validator:
    """Validador para valor máximo."""
    return Validator(f"max_value_{max_val}").add_rule(
        lambda x: None if not isinstance(x, (int, float)) or x <= max_val else f"Must be <= {max_val}"
    )


def pattern(regex: str) -> Validator:
    """Validador para patrón regex."""
    compiled = re.compile(regex)
    return Validator(f"pattern_{regex}").add_rule(
        lambda x: None if not isinstance(x, str) or compiled.match(x) else f"Does not match pattern {regex}"
    )


def email() -> Validator:
    """Validador para email."""
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return pattern(email_pattern).add_rule(
        lambda x: None if not isinstance(x, str) or re.match(email_pattern, x) else "Invalid email format"
    )


def url() -> Validator:
    """Validador para URL."""
    url_pattern = r'^https?://[^\s/$.?#].[^\s]*$'
    return pattern(url_pattern).add_rule(
        lambda x: None if not isinstance(x, str) or re.match(url_pattern, x) else "Invalid URL format"
    )


def one_of(choices: List[Any]) -> Validator:
    """Validador para valores permitidos."""
    return Validator(f"one_of_{choices}").add_rule(
        lambda x: None if x in choices else f"Must be one of {choices}"
    )


def custom(validator_func: Callable[[Any], Optional[str]]) -> Validator:
    """
    Crear validador personalizado.
    
    Args:
        validator_func: Función que retorna None si es válido, mensaje de error si no
    
    Returns:
        Validator
    
    Example:
        def is_positive(x):
            return None if x > 0 else "Must be positive"
        
        validator = custom(is_positive)
    """
    return Validator("custom").add_rule(validator_func)


def combine(*validators: Validator) -> Validator:
    """
    Combinar múltiples validadores.
    
    Args:
        *validators: Validadores a combinar
    
    Returns:
        Validator combinado
    
    Example:
        validator = combine(required(), min_length(3), max_length(50))
    """
    combined = Validator("combined")
    for validator in validators:
        for rule in validator._rules:
            combined.add_rule(rule)
    return combined


__all__ = [
    "ValidationError",
    "Validator",
    "SchemaValidator",
    "required",
    "not_empty",
    "min_length",
    "max_length",
    "min_value",
    "max_value",
    "pattern",
    "email",
    "url",
    "one_of",
    "custom",
    "combine",
]

