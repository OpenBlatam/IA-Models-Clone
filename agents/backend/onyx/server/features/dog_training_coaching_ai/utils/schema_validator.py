"""
Schema Validation Utilities
============================
Utilidades para validación de esquemas.
"""

from typing import Any, Dict, List, Optional, Union, Callable
from datetime import datetime
import re

from .logger import get_logger
from ..core.exceptions import ValidationException

logger = get_logger(__name__)


class SchemaValidator:
    """Validador de esquemas."""
    
    def __init__(self, schema: Dict[str, Any]):
        """
        Inicializar validador.
        
        Args:
            schema: Esquema de validación
        """
        self.schema = schema
    
    def validate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validar datos contra esquema.
        
        Args:
            data: Datos a validar
            
        Returns:
            Resultado de validación
        """
        errors = []
        validated_data = {}
        
        for field, rules in self.schema.items():
            value = data.get(field)
            
            # Verificar requerido
            if rules.get("required", False) and value is None:
                errors.append(f"Field '{field}' is required")
                continue
            
            # Si no es requerido y no está presente, continuar
            if value is None:
                continue
            
            # Validar tipo
            expected_type = rules.get("type")
            if expected_type and not isinstance(value, expected_type):
                errors.append(f"Field '{field}' must be of type {expected_type.__name__}")
                continue
            
            # Validar con función personalizada
            validator_func = rules.get("validator")
            if validator_func:
                try:
                    if not validator_func(value):
                        errors.append(f"Field '{field}' failed custom validation")
                        continue
                except Exception as e:
                    errors.append(f"Field '{field}' validation error: {str(e)}")
                    continue
            
            # Validar rango (para números)
            if isinstance(value, (int, float)):
                min_val = rules.get("min")
                max_val = rules.get("max")
                if min_val is not None and value < min_val:
                    errors.append(f"Field '{field}' must be >= {min_val}")
                    continue
                if max_val is not None and value > max_val:
                    errors.append(f"Field '{field}' must be <= {max_val}")
                    continue
            
            # Validar longitud (para strings y listas)
            if isinstance(value, (str, list)):
                min_len = rules.get("min_length")
                max_len = rules.get("max_length")
                if min_len is not None and len(value) < min_len:
                    errors.append(f"Field '{field}' must have length >= {min_len}")
                    continue
                if max_len is not None and len(value) > max_len:
                    errors.append(f"Field '{field}' must have length <= {max_len}")
                    continue
            
            # Validar patrón (para strings)
            if isinstance(value, str):
                pattern = rules.get("pattern")
                if pattern and not re.match(pattern, value):
                    errors.append(f"Field '{field}' does not match required pattern")
                    continue
            
            # Validar enum
            enum_values = rules.get("enum")
            if enum_values and value not in enum_values:
                errors.append(f"Field '{field}' must be one of {enum_values}")
                continue
            
            validated_data[field] = value
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "data": validated_data if len(errors) == 0 else None
        }


def validate_schema(data: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validar datos contra esquema.
    
    Args:
        data: Datos a validar
        schema: Esquema de validación
        
    Returns:
        Resultado de validación
    """
    validator = SchemaValidator(schema)
    return validator.validate(data)


def create_schema(
    fields: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Crear esquema desde definición de campos.
    
    Args:
        fields: Lista de definiciones de campos
        
    Returns:
        Esquema
    """
    schema = {}
    
    for field in fields:
        name = field["name"]
        schema[name] = {
            "type": field.get("type", str),
            "required": field.get("required", False),
            "min": field.get("min"),
            "max": field.get("max"),
            "min_length": field.get("min_length"),
            "max_length": field.get("max_length"),
            "pattern": field.get("pattern"),
            "enum": field.get("enum"),
            "validator": field.get("validator")
        }
    
    return schema


# Esquemas predefinidos comunes
COMMON_SCHEMAS = {
    "email": {
        "type": str,
        "required": True,
        "pattern": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    },
    "url": {
        "type": str,
        "required": True,
        "pattern": r"^https?://.+"
    },
    "positive_integer": {
        "type": int,
        "required": True,
        "min": 1
    },
    "non_empty_string": {
        "type": str,
        "required": True,
        "min_length": 1
    }
}



