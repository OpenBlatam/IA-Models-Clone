"""
Sistema de Validación de Datos Avanzado
========================================
Validación robusta de datos de entrada
"""

from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
from enum import Enum
import structlog
import re
from pydantic import BaseModel, Field, field_validator

logger = structlog.get_logger()


class ValidationRule:
    """Regla de validación"""
    
    def __init__(
        self,
        name: str,
        validator: Callable[[Any], bool],
        error_message: str
    ):
        self.name = name
        self.validator = validator
        self.error_message = error_message
    
    def validate(self, value: Any) -> tuple[bool, Optional[str]]:
        """
        Validar valor
        
        Returns:
            (is_valid, error_message)
        """
        try:
            if self.validator(value):
                return True, None
            else:
                return False, self.error_message
        except Exception as e:
            return False, f"Validation error: {str(e)}"


class DataValidator:
    """Validador de datos"""
    
    def __init__(self):
        """Inicializar validador"""
        self._rules: Dict[str, List[ValidationRule]] = {}
        self._load_default_rules()
        logger.info("DataValidator initialized")
    
    def _load_default_rules(self) -> None:
        """Cargar reglas por defecto"""
        # Reglas para email
        self._rules["email"] = [
            ValidationRule(
                name="email_format",
                validator=lambda x: bool(re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', x)),
                error_message="Invalid email format"
            ),
            ValidationRule(
                name="email_length",
                validator=lambda x: len(x) <= 255,
                error_message="Email too long"
            )
        ]
        
        # Reglas para URL
        self._rules["url"] = [
            ValidationRule(
                name="url_format",
                validator=lambda x: bool(re.match(r'^https?://', x)),
                error_message="Invalid URL format"
            )
        ]
        
        # Reglas para UUID
        self._rules["uuid"] = [
            ValidationRule(
                name="uuid_format",
                validator=lambda x: bool(re.match(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', x, re.I)),
                error_message="Invalid UUID format"
            )
        ]
    
    def add_rule(
        self,
        field_name: str,
        rule: ValidationRule
    ) -> None:
        """
        Agregar regla de validación
        
        Args:
            field_name: Nombre del campo
            rule: Regla de validación
        """
        if field_name not in self._rules:
            self._rules[field_name] = []
        
        self._rules[field_name].append(rule)
        logger.info("Validation rule added", field=field_name, rule=rule.name)
    
    def validate_field(
        self,
        field_name: str,
        value: Any
    ) -> tuple[bool, List[str]]:
        """
        Validar campo
        
        Args:
            field_name: Nombre del campo
            value: Valor a validar
            
        Returns:
            (is_valid, errors)
        """
        if field_name not in self._rules:
            return True, []
        
        errors = []
        for rule in self._rules[field_name]:
            is_valid, error = rule.validate(value)
            if not is_valid:
                errors.append(error)
        
        return len(errors) == 0, errors
    
    def validate_data(
        self,
        data: Dict[str, Any],
        schema: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Validar datos completos
        
        Args:
            data: Datos a validar
            schema: Esquema de validación (opcional)
            
        Returns:
            Resultado de validación
        """
        errors = {}
        valid = True
        
        for field_name, value in data.items():
            # Determinar tipo de validación
            validation_type = schema.get(field_name) if schema else None
            
            if validation_type:
                is_valid, field_errors = self.validate_field(validation_type, value)
            else:
                # Intentar inferir tipo
                if field_name.endswith("_email") or field_name == "email":
                    is_valid, field_errors = self.validate_field("email", value)
                elif field_name.endswith("_url") or field_name == "url":
                    is_valid, field_errors = self.validate_field("url", value)
                elif field_name.endswith("_id") and isinstance(value, str):
                    is_valid, field_errors = self.validate_field("uuid", value)
                else:
                    is_valid, field_errors = True, []
            
            if not is_valid:
                errors[field_name] = field_errors
                valid = False
        
        return {
            "valid": valid,
            "errors": errors
        }


# Instancia global del validador
data_validator = DataValidator()




