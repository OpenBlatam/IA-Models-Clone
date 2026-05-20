"""
Schema Validation - Validación Avanzada de Schemas
==================================================

Validación avanzada de schemas:
- JSON Schema validation
- Pydantic integration
- Custom validators
- Validation rules
- Schema versioning
"""

import logging
from typing import Optional, Dict, Any, List, Callable
from enum import Enum

logger = logging.getLogger(__name__)


class ValidationLevel(str, Enum):
    """Niveles de validación"""
    STRICT = "strict"
    MODERATE = "moderate"
    LENIENT = "lenient"


class SchemaValidator:
    """
    Validador de schemas.
    """
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.MODERATE) -> None:
        self.validation_level = validation_level
        self.schemas: Dict[str, Dict[str, Any]] = {}
        self.custom_validators: Dict[str, List[Callable]] = {}
    
    def register_schema(
        self,
        schema_name: str,
        schema: Dict[str, Any],
        version: Optional[str] = None
    ) -> None:
        """Registra schema"""
        key = f"{schema_name}:{version}" if version else schema_name
        self.schemas[key] = schema
        logger.info(f"Schema registered: {key}")
    
    def register_custom_validator(
        self,
        schema_name: str,
        validator: Callable[[Any], bool]
    ) -> None:
        """Registra validador personalizado"""
        if schema_name not in self.custom_validators:
            self.custom_validators[schema_name] = []
        self.custom_validators[schema_name].append(validator)
        logger.info(f"Custom validator registered for {schema_name}")
    
    def validate(
        self,
        schema_name: str,
        data: Any,
        version: Optional[str] = None
    ) -> tuple[bool, Optional[List[str]]]:
        """Valida datos contra schema"""
        key = f"{schema_name}:{version}" if version else schema_name
        schema = self.schemas.get(key)
        
        if not schema:
            if self.validation_level == ValidationLevel.STRICT:
                return False, [f"Schema {key} not found"]
            return True, None
        
        errors = []
        
        # Validación básica con Pydantic si está disponible
        try:
            from pydantic import ValidationError, BaseModel
            
            # Intentar validar con Pydantic
            if isinstance(schema, dict) and "type" in schema:
                # Crear modelo Pydantic dinámico
                model = self._create_pydantic_model(schema)
                try:
                    model(**data if isinstance(data, dict) else {"value": data})
                except ValidationError as e:
                    errors.extend([str(err) for err in e.errors()])
        except ImportError:
            # Fallback: validación manual básica
            errors.extend(self._validate_manual(schema, data))
        
        # Validadores personalizados
        if schema_name in self.custom_validators:
            for validator in self.custom_validators[schema_name]:
                try:
                    if not validator(data):
                        errors.append(f"Custom validation failed for {schema_name}")
                except Exception as e:
                    errors.append(f"Validator error: {e}")
        
        return len(errors) == 0, errors if errors else None
    
    def _create_pydantic_model(self, schema: Dict[str, Any]) -> Any:
        """Crea modelo Pydantic dinámico"""
        from pydantic import BaseModel, create_model
        
        fields = {}
        for field_name, field_schema in schema.get("properties", {}).items():
            field_type = self._get_python_type(field_schema.get("type", "string"))
            fields[field_name] = (field_type, ...)
        
        return create_model("DynamicModel", **fields)
    
    def _get_python_type(self, json_type: str) -> type:
        """Convierte tipo JSON a tipo Python"""
        type_map = {
            "string": str,
            "integer": int,
            "number": float,
            "boolean": bool,
            "array": list,
            "object": dict
        }
        return type_map.get(json_type, str)
    
    def _validate_manual(
        self,
        schema: Dict[str, Any],
        data: Any
    ) -> List[str]:
        """Validación manual básica"""
        errors = []
        
        if "type" in schema:
            expected_type = schema["type"]
            if expected_type == "object" and not isinstance(data, dict):
                errors.append(f"Expected object, got {type(data).__name__}")
            elif expected_type == "array" and not isinstance(data, list):
                errors.append(f"Expected array, got {type(data).__name__}")
            elif expected_type == "string" and not isinstance(data, str):
                errors.append(f"Expected string, got {type(data).__name__}")
            elif expected_type == "integer" and not isinstance(data, int):
                errors.append(f"Expected integer, got {type(data).__name__}")
            elif expected_type == "number" and not isinstance(data, (int, float)):
                errors.append(f"Expected number, got {type(data).__name__}")
            elif expected_type == "boolean" and not isinstance(data, bool):
                errors.append(f"Expected boolean, got {type(data).__name__}")
        
        # Validar propiedades requeridas
        if isinstance(data, dict) and "required" in schema:
            for field in schema["required"]:
                if field not in data:
                    errors.append(f"Required field missing: {field}")
        
        return errors


def get_schema_validator(
    validation_level: ValidationLevel = ValidationLevel.MODERATE
) -> SchemaValidator:
    """Obtiene validador de schemas"""
    return SchemaValidator(validation_level=validation_level)















