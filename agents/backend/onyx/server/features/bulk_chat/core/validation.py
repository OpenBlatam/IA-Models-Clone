"""
Validation - Sistema de Validación Avanzada
===========================================

Sistema de validación con reglas personalizadas y validadores.
"""

import asyncio
import logging
import re
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Error de validación."""
    pass


@dataclass
class ValidationRule:
    """Regla de validación."""
    rule_id: str
    field_name: str
    validator: Callable
    error_message: str
    required: bool = True


class ValidationEngine:
    """Motor de validación."""
    
    def __init__(self):
        self.rules: Dict[str, List[ValidationRule]] = {}
        self.validators: Dict[str, Callable] = {}
        self._register_default_validators()
    
    def _register_default_validators(self):
        """Registrar validadores por defecto."""
        self.validators["email"] = self._validate_email
        self.validators["url"] = self._validate_url
        self.validators["phone"] = self._validate_phone
        self.validators["min_length"] = self._validate_min_length
        self.validators["max_length"] = self._validate_max_length
        self.validators["regex"] = self._validate_regex
    
    def register_rule(self, context: str, rule: ValidationRule):
        """Registrar regla de validación."""
        if context not in self.rules:
            self.rules[context] = []
        self.rules[context].append(rule)
        logger.debug(f"Registered validation rule: {rule.rule_id} for {context}")
    
    async def validate(
        self,
        context: str,
        data: Dict[str, Any],
    ) -> Dict[str, List[str]]:
        """
        Validar datos.
        
        Args:
            context: Contexto de validación
            data: Datos a validar
        
        Returns:
            Diccionario con errores por campo
        """
        errors: Dict[str, List[str]] = {}
        
        rules = self.rules.get(context, [])
        
        for rule in rules:
            field_value = data.get(rule.field_name)
            
            # Verificar si es requerido
            if rule.required and (field_value is None or field_value == ""):
                if rule.field_name not in errors:
                    errors[rule.field_name] = []
                errors[rule.field_name].append(f"{rule.field_name} is required")
                continue
            
            # Validar si tiene valor
            if field_value is not None and field_value != "":
                try:
                    if asyncio.iscoroutinefunction(rule.validator):
                        result = await rule.validator(field_value, data)
                    else:
                        result = rule.validator(field_value, data)
                    
                    if not result:
                        if rule.field_name not in errors:
                            errors[rule.field_name] = []
                        errors[rule.field_name].append(rule.error_message)
                        
                except Exception as e:
                    if rule.field_name not in errors:
                        errors[rule.field_name] = []
                    errors[rule.field_name].append(f"Validation error: {str(e)}")
        
        return errors
    
    def _validate_email(self, value: str, data: Dict[str, Any]) -> bool:
        """Validar email."""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, value))
    
    def _validate_url(self, value: str, data: Dict[str, Any]) -> bool:
        """Validar URL."""
        pattern = r'^https?://.+'
        return bool(re.match(pattern, value))
    
    def _validate_phone(self, value: str, data: Dict[str, Any]) -> bool:
        """Validar teléfono."""
        pattern = r'^\+?[\d\s-()]+$'
        return bool(re.match(pattern, value))
    
    def _validate_min_length(self, value: str, min_length: int, data: Dict[str, Any]) -> bool:
        """Validar longitud mínima."""
        return len(value) >= min_length
    
    def _validate_max_length(self, value: str, max_length: int, data: Dict[str, Any]) -> bool:
        """Validar longitud máxima."""
        return len(value) <= max_length
    
    def _validate_regex(self, value: str, pattern: str, data: Dict[str, Any]) -> bool:
        """Validar con regex."""
        return bool(re.match(pattern, value))
    
    def create_validator(self, validator_name: str, validator_func: Callable):
        """Crear validador personalizado."""
        self.validators[validator_name] = validator_func
        logger.info(f"Created custom validator: {validator_name}")
















