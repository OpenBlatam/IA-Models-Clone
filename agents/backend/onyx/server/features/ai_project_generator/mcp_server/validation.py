"""
MCP Advanced Validation - Validación avanzada
==============================================
"""

import logging
from typing import Any, Dict, Optional, List, Callable
from pydantic import BaseModel, Field, validator
import re

from .exceptions import MCPValidationError

logger = logging.getLogger(__name__)


class ValidationRule(BaseModel):
    """Regla de validación"""
    field: str = Field(..., description="Campo a validar")
    rule_type: str = Field(..., description="Tipo de regla")
    value: Any = Field(None, description="Valor de la regla")
    message: Optional[str] = Field(None, description="Mensaje de error")


class AdvancedValidator:
    """
    Validador avanzado
    
    Permite definir reglas de validación complejas.
    """
    
    def __init__(self):
        self._rules: Dict[str, List[ValidationRule]] = {}
        self._custom_validators: Dict[str, Callable] = {}
    
    def add_rule(self, resource_id: str, rule: ValidationRule):
        """
        Agrega regla de validación
        
        Args:
            resource_id: ID del recurso
            rule: Regla de validación
        """
        if resource_id not in self._rules:
            self._rules[resource_id] = []
        
        self._rules[resource_id].append(rule)
        logger.info(f"Added validation rule for {resource_id}: {rule.rule_type}")
    
    def register_validator(self, name: str, validator_func: Callable):
        """
        Registra validador personalizado
        
        Args:
            name: Nombre del validador
            validator_func: Función de validación
        """
        self._custom_validators[name] = validator_func
        logger.info(f"Registered custom validator: {name}")
    
    def validate(
        self,
        resource_id: str,
        data: Dict[str, Any],
    ) -> List[str]:
        """
        Valida datos según reglas
        
        Args:
            resource_id: ID del recurso
            data: Datos a validar
            
        Returns:
            Lista de errores (vacía si válido)
        """
        errors = []
        rules = self._rules.get(resource_id, [])
        
        for rule in rules:
            field_value = data.get(rule.field)
            
            try:
                if not self._check_rule(rule, field_value):
                    error_msg = rule.message or f"Validation failed for {rule.field}: {rule.rule_type}"
                    errors.append(error_msg)
            except Exception as e:
                errors.append(f"Validation error for {rule.field}: {e}")
        
        return errors
    
    def _check_rule(self, rule: ValidationRule, value: Any) -> bool:
        """
        Verifica una regla
        
        Args:
            rule: Regla a verificar
            value: Valor a validar
            
        Returns:
            True si pasa la validación
        """
        if rule.rule_type == "required":
            return value is not None and value != ""
        
        if rule.rule_type == "min_length":
            return len(str(value)) >= rule.value
        
        if rule.rule_type == "max_length":
            return len(str(value)) <= rule.value
        
        if rule.rule_type == "min":
            return float(value) >= rule.value
        
        if rule.rule_type == "max":
            return float(value) <= rule.value
        
        if rule.rule_type == "pattern":
            return bool(re.match(rule.value, str(value)))
        
        if rule.rule_type == "email":
            email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            return bool(re.match(email_pattern, str(value)))
        
        if rule.rule_type == "url":
            url_pattern = r'^https?://.+'
            return bool(re.match(url_pattern, str(value)))
        
        if rule.rule_type == "custom":
            validator_func = self._custom_validators.get(rule.value)
            if validator_func:
                return validator_func(value)
        
        return True


def validate_request(
    validator: AdvancedValidator,
    resource_id: str,
    data: Dict[str, Any],
) -> None:
    """
    Valida request y lanza excepción si falla
    
    Args:
        validator: Instancia de AdvancedValidator
        resource_id: ID del recurso
        data: Datos a validar
        
    Raises:
        MCPValidationError: Si la validación falla
    """
    errors = validator.validate(resource_id, data)
    if errors:
        raise MCPValidationError(f"Validation failed: {', '.join(errors)}")

