"""
Advanced Validation Service - Validación avanzada
==================================================

Sistema de validación avanzado con reglas personalizadas.
"""

import logging
import re
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class ValidationRule(str, Enum):
    """Tipos de reglas de validación"""
    REQUIRED = "required"
    MIN_LENGTH = "min_length"
    MAX_LENGTH = "max_length"
    PATTERN = "pattern"
    EMAIL = "email"
    URL = "url"
    NUMBER = "number"
    MIN_VALUE = "min_value"
    MAX_VALUE = "max_value"
    CUSTOM = "custom"


@dataclass
class ValidationError:
    """Error de validación"""
    field: str
    rule: str
    message: str
    value: Any = None


@dataclass
class ValidationResult:
    """Resultado de validación"""
    valid: bool
    errors: List[ValidationError] = field(default_factory=list)


class AdvancedValidationService:
    """Servicio de validación avanzado"""
    
    def __init__(self):
        """Inicializar servicio"""
        self.validators: Dict[str, Callable] = {}
        logger.info("AdvancedValidationService initialized")
    
    def register_validator(self, rule_name: str, validator: Callable):
        """Registrar validador personalizado"""
        self.validators[rule_name] = validator
        logger.info(f"Validator registered: {rule_name}")
    
    def validate(
        self,
        data: Dict[str, Any],
        rules: Dict[str, List[Dict[str, Any]]]
    ) -> ValidationResult:
        """Validar datos según reglas"""
        errors = []
        
        for field, field_rules in rules.items():
            value = data.get(field)
            
            for rule_config in field_rules:
                rule_type = rule_config.get("type")
                rule_value = rule_config.get("value")
                error_message = rule_config.get("message")
                
                error = self._validate_field(field, value, rule_type, rule_value, error_message)
                if error:
                    errors.append(error)
        
        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
        )
    
    def _validate_field(
        self,
        field: str,
        value: Any,
        rule_type: str,
        rule_value: Any,
        error_message: Optional[str]
    ) -> Optional[ValidationError]:
        """Validar campo individual"""
        if rule_type == ValidationRule.REQUIRED:
            if value is None or value == "":
                return ValidationError(
                    field=field,
                    rule=rule_type,
                    message=error_message or f"{field} is required",
                    value=value,
                )
        
        elif rule_type == ValidationRule.MIN_LENGTH:
            if value and len(str(value)) < rule_value:
                return ValidationError(
                    field=field,
                    rule=rule_type,
                    message=error_message or f"{field} must be at least {rule_value} characters",
                    value=value,
                )
        
        elif rule_type == ValidationRule.MAX_LENGTH:
            if value and len(str(value)) > rule_value:
                return ValidationError(
                    field=field,
                    rule=rule_type,
                    message=error_message or f"{field} must be at most {rule_value} characters",
                    value=value,
                )
        
        elif rule_type == ValidationRule.PATTERN:
            if value and not re.match(rule_value, str(value)):
                return ValidationError(
                    field=field,
                    rule=rule_type,
                    message=error_message or f"{field} does not match required pattern",
                    value=value,
                )
        
        elif rule_type == ValidationRule.EMAIL:
            email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            if value and not re.match(email_pattern, str(value)):
                return ValidationError(
                    field=field,
                    rule=rule_type,
                    message=error_message or f"{field} must be a valid email",
                    value=value,
                )
        
        elif rule_type == ValidationRule.URL:
            url_pattern = r'^https?://.+'
            if value and not re.match(url_pattern, str(value)):
                return ValidationError(
                    field=field,
                    rule=rule_type,
                    message=error_message or f"{field} must be a valid URL",
                    value=value,
                )
        
        elif rule_type == ValidationRule.NUMBER:
            try:
                float(value)
            except (ValueError, TypeError):
                return ValidationError(
                    field=field,
                    rule=rule_type,
                    message=error_message or f"{field} must be a number",
                    value=value,
                )
        
        elif rule_type == ValidationRule.MIN_VALUE:
            try:
                num_value = float(value)
                if num_value < rule_value:
                    return ValidationError(
                        field=field,
                        rule=rule_type,
                        message=error_message or f"{field} must be at least {rule_value}",
                        value=value,
                    )
            except (ValueError, TypeError):
                pass
        
        elif rule_type == ValidationRule.MAX_VALUE:
            try:
                num_value = float(value)
                if num_value > rule_value:
                    return ValidationError(
                        field=field,
                        rule=rule_type,
                        message=error_message or f"{field} must be at most {rule_value}",
                        value=value,
                    )
            except (ValueError, TypeError):
                pass
        
        elif rule_type == ValidationRule.CUSTOM:
            validator = self.validators.get(rule_value)
            if validator:
                if not validator(value):
                    return ValidationError(
                        field=field,
                        rule=rule_type,
                        message=error_message or f"{field} failed custom validation",
                        value=value,
                    )
        
        return None
    
    def validate_schema(
        self,
        data: Dict[str, Any],
        schema: Dict[str, Any]
    ) -> ValidationResult:
        """Validar datos contra schema"""
        rules = {}
        
        for field, field_schema in schema.items():
            field_rules = []
            
            if field_schema.get("required"):
                field_rules.append({"type": ValidationRule.REQUIRED})
            
            if "min_length" in field_schema:
                field_rules.append({
                    "type": ValidationRule.MIN_LENGTH,
                    "value": field_schema["min_length"],
                })
            
            if "max_length" in field_schema:
                field_rules.append({
                    "type": ValidationRule.MAX_LENGTH,
                    "value": field_schema["max_length"],
                })
            
            if "pattern" in field_schema:
                field_rules.append({
                    "type": ValidationRule.PATTERN,
                    "value": field_schema["pattern"],
                })
            
            if field_schema.get("type") == "email":
                field_rules.append({"type": ValidationRule.EMAIL})
            
            if field_schema.get("type") == "url":
                field_rules.append({"type": ValidationRule.URL})
            
            if field_schema.get("type") == "number":
                field_rules.append({"type": ValidationRule.NUMBER})
                if "min" in field_schema:
                    field_rules.append({
                        "type": ValidationRule.MIN_VALUE,
                        "value": field_schema["min"],
                    })
                if "max" in field_schema:
                    field_rules.append({
                        "type": ValidationRule.MAX_VALUE,
                        "value": field_schema["max"],
                    })
            
            if field_rules:
                rules[field] = field_rules
        
        return self.validate(data, rules)




