"""
Advanced Request Validation - Validación Avanzada de Requests
============================================================

Validación avanzada de requests:
- Request schema validation
- Parameter validation
- Header validation
- Body validation
- Custom validators
- Validation rules engine
"""

from typing import Optional, Dict, Any, List, Callable
from enum import Enum

from .shared_utils import get_logger

logger = get_logger(__name__)


class ValidationRule(str, Enum):
    """Reglas de validación"""
    REQUIRED = "required"
    TYPE = "type"
    MIN_LENGTH = "min_length"
    MAX_LENGTH = "max_length"
    MIN_VALUE = "min_value"
    MAX_VALUE = "max_value"
    PATTERN = "pattern"
    ENUM = "enum"
    CUSTOM = "custom"


class ValidationError:
    """Error de validación"""
    
    def __init__(
        self,
        field: str,
        rule: str,
        message: str,
        value: Any = None
    ) -> None:
        self.field = field
        self.rule = rule
        self.message = message
        self.value = value
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario"""
        return {
            "field": self.field,
            "rule": self.rule,
            "message": self.message,
            "value": self.value
        }


class AdvancedRequestValidator:
    """
    Validador avanzado de requests.
    """
    
    def __init__(self) -> None:
        self.validation_schemas: Dict[str, Dict[str, Any]] = {}
        self.custom_validators: Dict[str, List[Callable]] = {}
    
    def register_schema(
        self,
        endpoint: str,
        schema: Dict[str, Any]
    ) -> None:
        """Registra schema de validación"""
        self.validation_schemas[endpoint] = schema
        logger.info(f"Validation schema registered for {endpoint}")
    
    def register_custom_validator(
        self,
        field: str,
        validator: Callable[[Any], bool],
        message: Optional[str] = None
    ) -> None:
        """Registra validador personalizado"""
        if field not in self.custom_validators:
            self.custom_validators[field] = []
        self.custom_validators[field].append((validator, message))
        logger.info(f"Custom validator registered for {field}")
    
    def validate(
        self,
        endpoint: str,
        request: Dict[str, Any]
    ) -> tuple[bool, List[ValidationError]]:
        """Valida request"""
        schema = self.validation_schemas.get(endpoint)
        if not schema:
            return True, []  # Sin schema, no validar
        
        errors = []
        
        # Validar parámetros
        if "params" in schema:
            errors.extend(self._validate_params(request.get("params", {}), schema["params"]))
        
        # Validar headers
        if "headers" in schema:
            errors.extend(self._validate_headers(request.get("headers", {}), schema["headers"]))
        
        # Validar body
        if "body" in schema:
            errors.extend(self._validate_body(request.get("body", {}), schema["body"]))
        
        return len(errors) == 0, errors
    
    def _validate_params(
        self,
        params: Dict[str, Any],
        schema: Dict[str, Any]
    ) -> List[ValidationError]:
        """Valida parámetros"""
        errors = []
        
        for field, rules in schema.items():
            value = params.get(field)
            
            # Required
            if rules.get(ValidationRule.REQUIRED) and value is None:
                errors.append(ValidationError(
                    field,
                    ValidationRule.REQUIRED,
                    f"Field {field} is required"
                ))
                continue
            
            if value is None:
                continue  # Campo opcional sin valor
            
            # Type validation
            if ValidationRule.TYPE in rules:
                expected_type = rules[ValidationRule.TYPE]
                if not self._check_type(value, expected_type):
                    errors.append(ValidationError(
                        field,
                        ValidationRule.TYPE,
                        f"Field {field} must be of type {expected_type}",
                        value
                    ))
            
            # String validations
            if isinstance(value, str):
                if ValidationRule.MIN_LENGTH in rules:
                    if len(value) < rules[ValidationRule.MIN_LENGTH]:
                        errors.append(ValidationError(
                            field,
                            ValidationRule.MIN_LENGTH,
                            f"Field {field} must be at least {rules[ValidationRule.MIN_LENGTH]} characters",
                            value
                        ))
                
                if ValidationRule.MAX_LENGTH in rules:
                    if len(value) > rules[ValidationRule.MAX_LENGTH]:
                        errors.append(ValidationError(
                            field,
                            ValidationRule.MAX_LENGTH,
                            f"Field {field} must be at most {rules[ValidationRule.MAX_LENGTH]} characters",
                            value
                        ))
                
                if ValidationRule.PATTERN in rules:
                    import re
                    if not re.match(rules[ValidationRule.PATTERN], value):
                        errors.append(ValidationError(
                            field,
                            ValidationRule.PATTERN,
                            f"Field {field} does not match required pattern",
                            value
                        ))
            
            # Number validations
            if isinstance(value, (int, float)):
                if ValidationRule.MIN_VALUE in rules:
                    if value < rules[ValidationRule.MIN_VALUE]:
                        errors.append(ValidationError(
                            field,
                            ValidationRule.MIN_VALUE,
                            f"Field {field} must be at least {rules[ValidationRule.MIN_VALUE]}",
                            value
                        ))
                
                if ValidationRule.MAX_VALUE in rules:
                    if value > rules[ValidationRule.MAX_VALUE]:
                        errors.append(ValidationError(
                            field,
                            ValidationRule.MAX_VALUE,
                            f"Field {field} must be at most {rules[ValidationRule.MAX_VALUE]}",
                            value
                        ))
            
            # Enum validation
            if ValidationRule.ENUM in rules:
                if value not in rules[ValidationRule.ENUM]:
                    errors.append(ValidationError(
                        field,
                        ValidationRule.ENUM,
                        f"Field {field} must be one of {rules[ValidationRule.ENUM]}",
                        value
                    ))
            
            # Custom validators
            if field in self.custom_validators:
                for validator, message in self.custom_validators[field]:
                    try:
                        if not validator(value):
                            errors.append(ValidationError(
                                field,
                                ValidationRule.CUSTOM,
                                message or f"Custom validation failed for {field}",
                                value
                            ))
                    except Exception as e:
                        logger.error(f"Custom validator error for {field}: {e}")
        
        return errors
    
    def _validate_headers(
        self,
        headers: Dict[str, Any],
        schema: Dict[str, Any]
    ) -> List[ValidationError]:
        """Valida headers"""
        return self._validate_params(headers, schema)
    
    def _validate_body(
        self,
        body: Dict[str, Any],
        schema: Dict[str, Any]
    ) -> List[ValidationError]:
        """Valida body"""
        return self._validate_params(body, schema)
    
    def _check_type(self, value: Any, expected_type: str) -> bool:
        """Verifica tipo"""
        type_map = {
            "string": str,
            "integer": int,
            "number": float,
            "boolean": bool,
            "array": list,
            "object": dict
        }
        
        expected = type_map.get(expected_type)
        if not expected:
            return True  # Tipo desconocido, no validar
        
        # Verificar tipo
        if expected_type == "number":
            return isinstance(value, (int, float))
        
        return isinstance(value, expected)


def get_advanced_request_validator() -> AdvancedRequestValidator:
    """Obtiene validador avanzado de requests"""
    return AdvancedRequestValidator()




