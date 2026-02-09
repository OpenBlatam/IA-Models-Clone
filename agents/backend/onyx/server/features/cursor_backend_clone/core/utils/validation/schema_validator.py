"""
Schema Validator - Validador de Esquemas
=========================================

Sistema para validar datos contra esquemas definidos.
"""

import logging
from typing import Any, Dict, List, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Error de validación"""
    def __init__(self, message: str, field: Optional[str] = None, errors: Optional[List[str]] = None):
        self.message = message
        self.field = field
        self.errors = errors or []
        super().__init__(message)


class FieldType(Enum):
    """Tipos de campo"""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    LIST = "list"
    DICT = "dict"
    ANY = "any"


@dataclass
class FieldSchema:
    """Esquema de campo"""
    name: str
    field_type: FieldType
    required: bool = True
    default: Any = None
    validator: Optional[Callable[[Any], bool]] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    pattern: Optional[str] = None
    enum_values: Optional[List[Any]] = None
    nested_schema: Optional['Schema'] = None
    description: Optional[str] = None


class Schema:
    """
    Esquema de validación.
    
    Define estructura y reglas de validación para datos.
    """
    
    def __init__(self, name: str):
        self.name = name
        self.fields: Dict[str, FieldSchema] = {}
    
    def add_field(
        self,
        name: str,
        field_type: FieldType,
        required: bool = True,
        default: Any = None,
        validator: Optional[Callable[[Any], bool]] = None,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        min_value: Optional[Union[int, float]] = None,
        max_value: Optional[Union[int, float]] = None,
        pattern: Optional[str] = None,
        enum_values: Optional[List[Any]] = None,
        nested_schema: Optional['Schema'] = None,
        description: Optional[str] = None
    ) -> FieldSchema:
        """
        Agregar campo al esquema.
        
        Args:
            name: Nombre del campo
            field_type: Tipo del campo
            required: Si es requerido
            default: Valor por defecto
            validator: Validador personalizado
            min_length: Longitud mínima (para strings/listas)
            max_length: Longitud máxima (para strings/listas)
            min_value: Valor mínimo (para números)
            max_value: Valor máximo (para números)
            pattern: Patrón regex (para strings)
            enum_values: Valores permitidos
            nested_schema: Esquema anidado (para dict/list)
            description: Descripción del campo
            
        Returns:
            FieldSchema creado
        """
        field = FieldSchema(
            name=name,
            field_type=field_type,
            required=required,
            default=default,
            validator=validator,
            min_length=min_length,
            max_length=max_length,
            min_value=min_value,
            max_value=max_value,
            pattern=pattern,
            enum_values=enum_values,
            nested_schema=nested_schema,
            description=description
        )
        
        self.fields[name] = field
        return field
    
    def validate(self, data: Dict[str, Any], strict: bool = True) -> Dict[str, Any]:
        """
        Validar datos contra el esquema.
        
        Args:
            data: Datos a validar
            strict: Si rechazar campos no definidos
            
        Returns:
            Datos validados y normalizados
            
        Raises:
            ValidationError: Si la validación falla
        """
        errors = []
        validated = {}
        
        # Validar campos definidos
        for field_name, field_schema in self.fields.items():
            value = data.get(field_name, field_schema.default)
            
            # Verificar requerido
            if field_schema.required and value is None:
                errors.append(f"Field '{field_name}' is required")
                continue
            
            # Si no está presente y tiene default, usar default
            if field_name not in data and field_schema.default is not None:
                value = field_schema.default
            
            # Validar tipo
            try:
                validated_value = self._validate_field(field_schema, value)
                validated[field_name] = validated_value
            except ValidationError as e:
                errors.append(f"Field '{field_name}': {e.message}")
        
        # Verificar campos no definidos en modo strict
        if strict:
            for key in data.keys():
                if key not in self.fields:
                    errors.append(f"Unexpected field: '{key}'")
        
        if errors:
            raise ValidationError(
                f"Validation failed for schema '{self.name}'",
                errors=errors
            )
        
        return validated
    
    def _validate_field(self, field_schema: FieldSchema, value: Any) -> Any:
        """Validar campo individual"""
        # Validar tipo
        if value is None and not field_schema.required:
            return field_schema.default
        
        if value is None:
            raise ValidationError("Value cannot be None for required field")
        
        # Validar tipo básico
        type_valid = self._check_type(field_schema.field_type, value)
        if not type_valid:
            raise ValidationError(
                f"Expected {field_schema.field_type.value}, got {type(value).__name__}"
            )
        
        # Validar longitud (strings/listas)
        if field_schema.min_length is not None:
            length = len(value) if isinstance(value, (str, list)) else 0
            if length < field_schema.min_length:
                raise ValidationError(f"Length must be at least {field_schema.min_length}")
        
        if field_schema.max_length is not None:
            length = len(value) if isinstance(value, (str, list)) else 0
            if length > field_schema.max_length:
                raise ValidationError(f"Length must be at most {field_schema.max_length}")
        
        # Validar valor (números)
        if field_schema.min_value is not None and isinstance(value, (int, float)):
            if value < field_schema.min_value:
                raise ValidationError(f"Value must be at least {field_schema.min_value}")
        
        if field_schema.max_value is not None and isinstance(value, (int, float)):
            if value > field_schema.max_value:
                raise ValidationError(f"Value must be at most {field_schema.max_value}")
        
        # Validar patrón (strings)
        if field_schema.pattern and isinstance(value, str):
            import re
            if not re.match(field_schema.pattern, value):
                raise ValidationError(f"Value does not match pattern: {field_schema.pattern}")
        
        # Validar enum
        if field_schema.enum_values and value not in field_schema.enum_values:
            raise ValidationError(f"Value must be one of: {field_schema.enum_values}")
        
        # Validar esquema anidado
        if field_schema.nested_schema:
            if field_schema.field_type == FieldType.DICT and isinstance(value, dict):
                value = field_schema.nested_schema.validate(value)
            elif field_schema.field_type == FieldType.LIST and isinstance(value, list):
                value = [
                    field_schema.nested_schema.validate(item) if isinstance(item, dict) else item
                    for item in value
                ]
        
        # Validar con función personalizada
        if field_schema.validator:
            if not field_schema.validator(value):
                raise ValidationError("Custom validation failed")
        
        return value
    
    def _check_type(self, field_type: FieldType, value: Any) -> bool:
        """Verificar tipo de valor"""
        type_map = {
            FieldType.STRING: str,
            FieldType.INTEGER: int,
            FieldType.FLOAT: (int, float),
            FieldType.BOOLEAN: bool,
            FieldType.LIST: list,
            FieldType.DICT: dict,
            FieldType.ANY: object
        }
        
        expected_type = type_map.get(field_type)
        if expected_type is None:
            return True
        
        if isinstance(expected_type, tuple):
            return isinstance(value, expected_type)
        
        return isinstance(value, expected_type)




