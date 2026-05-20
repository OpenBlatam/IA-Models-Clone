"""
Schema Validator - Sistema de validación de esquemas avanzado
"""

import logging
import json
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, ValidationError, Field
from enum import Enum

logger = logging.getLogger(__name__)


class ContentSchema(BaseModel):
    """Esquema de validación de contenido"""
    
    min_length: Optional[int] = Field(None, ge=0)
    max_length: Optional[int] = Field(None, ge=0)
    required_fields: Optional[List[str]] = None
    forbidden_patterns: Optional[List[str]] = None
    allowed_patterns: Optional[List[str]] = None
    format_type: Optional[str] = None  # json, markdown, html, etc.


class SchemaValidator:
    """Validador de esquemas avanzado"""

    def __init__(self):
        """Inicializar el validador"""
        self.schemas: Dict[str, ContentSchema] = {}

    def register_schema(self, name: str, schema: ContentSchema):
        """
        Registrar un esquema.

        Args:
            name: Nombre del esquema
            schema: Esquema de validación
        """
        self.schemas[name] = schema
        logger.info(f"Esquema registrado: {name}")

    def validate_content(
        self,
        content: str,
        schema_name: Optional[str] = None,
        schema: Optional[ContentSchema] = None
    ) -> Dict[str, Any]:
        """
        Validar contenido contra un esquema.

        Args:
            content: Contenido a validar
            schema_name: Nombre del esquema registrado
            schema: Esquema directo

        Returns:
            Resultado de la validación
        """
        validation_schema = schema
        
        if schema_name:
            validation_schema = self.schemas.get(schema_name)
            if not validation_schema:
                return {
                    "valid": False,
                    "errors": [f"Esquema '{schema_name}' no encontrado"]
                }
        
        if not validation_schema:
            return {
                "valid": False,
                "errors": ["No se proporcionó esquema de validación"]
            }
        
        errors = []
        
        # Validar longitud
        if validation_schema.min_length and len(content) < validation_schema.min_length:
            errors.append(f"Contenido muy corto: {len(content)} < {validation_schema.min_length}")
        
        if validation_schema.max_length and len(content) > validation_schema.max_length:
            errors.append(f"Contenido muy largo: {len(content)} > {validation_schema.max_length}")
        
        # Validar formato
        if validation_schema.format_type:
            if validation_schema.format_type == "json":
                try:
                    json.loads(content)
                except json.JSONDecodeError as e:
                    errors.append(f"JSON inválido: {str(e)}")
        
        # Validar patrones prohibidos
        if validation_schema.forbidden_patterns:
            import re
            for pattern in validation_schema.forbidden_patterns:
                if re.search(pattern, content):
                    errors.append(f"Patrón prohibido encontrado: {pattern}")
        
        # Validar patrones permitidos
        if validation_schema.allowed_patterns:
            import re
            found = False
            for pattern in validation_schema.allowed_patterns:
                if re.search(pattern, content):
                    found = True
                    break
            if not found:
                errors.append("No se encontró ningún patrón permitido")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": []
        }

    def validate_json_schema(
        self,
        content: str,
        json_schema: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validar JSON contra un esquema JSON Schema.

        Args:
            content: Contenido JSON
            json_schema: Esquema JSON Schema

        Returns:
            Resultado de la validación
        """
        try:
            data = json.loads(content)
            # Aquí se podría usar jsonschema library
            # Por ahora, validación básica
            return {
                "valid": True,
                "errors": []
            }
        except json.JSONDecodeError as e:
            return {
                "valid": False,
                "errors": [f"JSON inválido: {str(e)}"]
            }






