"""
Advanced Validator - Sistema de validación avanzada de datos
=============================================================
"""

import logging
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
from enum import Enum
import re

logger = logging.getLogger(__name__)


class ValidationLevel(str, Enum):
    """Niveles de validación"""
    STRICT = "strict"
    MODERATE = "moderate"
    LENIENT = "lenient"


class AdvancedValidator:
    """Sistema de validación avanzada"""
    
    def __init__(self):
        self.validation_rules: Dict[str, List[Callable]] = {}
        self.custom_validators: Dict[str, Callable] = {}
    
    def register_rule(self, field_name: str, validator: Callable, priority: int = 0):
        """Registra una regla de validación"""
        if field_name not in self.validation_rules:
            self.validation_rules[field_name] = []
        
        self.validation_rules[field_name].append({
            "validator": validator,
            "priority": priority
        })
        
        # Ordenar por prioridad
        self.validation_rules[field_name].sort(key=lambda x: x["priority"], reverse=True)
    
    def validate_prototype_request(self, data: Dict[str, Any], 
                                   level: ValidationLevel = ValidationLevel.MODERATE) -> Dict[str, Any]:
        """Valida una solicitud de prototipo"""
        errors = []
        warnings = []
        
        # Validar descripción
        description = data.get("product_description", "")
        if not description or len(description.strip()) < 10:
            errors.append({
                "field": "product_description",
                "message": "La descripción debe tener al menos 10 caracteres",
                "level": "error"
            })
        elif len(description) > 1000:
            warnings.append({
                "field": "product_description",
                "message": "La descripción es muy larga, puede afectar el procesamiento",
                "level": "warning"
            })
        
        # Validar presupuesto
        budget = data.get("budget")
        if budget is not None:
            if budget < 0:
                errors.append({
                    "field": "budget",
                    "message": "El presupuesto no puede ser negativo",
                    "level": "error"
                })
            elif budget > 100000:
                warnings.append({
                    "field": "budget",
                    "message": "Presupuesto muy alto, verifica que sea correcto",
                    "level": "warning"
                })
        
        # Validar tipo de producto
        product_type = data.get("product_type")
        valid_types = ["licuadora", "estufa", "maquina", "electrodomestico", "herramienta", "otro"]
        if product_type and product_type.lower() not in valid_types:
            if level == ValidationLevel.STRICT:
                errors.append({
                    "field": "product_type",
                    "message": f"Tipo de producto inválido. Debe ser uno de: {', '.join(valid_types)}",
                    "level": "error"
                })
            else:
                warnings.append({
                    "field": "product_type",
                    "message": f"Tipo de producto no reconocido: {product_type}",
                    "level": "warning"
                })
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "level": level.value
        }
    
    def validate_material(self, material: Dict[str, Any]) -> Dict[str, Any]:
        """Valida un material"""
        errors = []
        
        # Validar nombre
        name = material.get("name", "")
        if not name or len(name.strip()) < 2:
            errors.append("El nombre del material es requerido y debe tener al menos 2 caracteres")
        
        # Validar cantidad
        quantity = material.get("quantity", 0)
        if quantity <= 0:
            errors.append("La cantidad debe ser mayor a 0")
        
        # Validar precio
        price = material.get("price_per_unit", 0)
        if price < 0:
            errors.append("El precio no puede ser negativo")
        
        # Validar fuentes
        sources = material.get("sources", [])
        if not sources:
            errors.append("El material debe tener al menos una fuente")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors
        }
    
    def validate_email(self, email: str) -> bool:
        """Valida formato de email"""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))
    
    def validate_url(self, url: str) -> bool:
        """Valida formato de URL"""
        pattern = r'^https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*)?(?:\?(?:[\w&=%.])*)?(?:#(?:\w*))?$'
        return bool(re.match(pattern, url))
    
    def validate_phone(self, phone: str) -> bool:
        """Valida formato de teléfono"""
        # Remover espacios y caracteres especiales
        cleaned = re.sub(r'[\s\-\(\)]', '', phone)
        # Verificar que sean solo dígitos y tenga longitud razonable
        return cleaned.isdigit() and 7 <= len(cleaned) <= 15
    
    def sanitize_input(self, text: str, max_length: Optional[int] = None) -> str:
        """Sanitiza entrada de texto"""
        # Remover caracteres peligrosos
        sanitized = re.sub(r'[<>"\']', '', text)
        
        # Limitar longitud
        if max_length and len(sanitized) > max_length:
            sanitized = sanitized[:max_length]
        
        return sanitized.strip()
    
    def validate_and_sanitize(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Valida y sanitiza datos"""
        sanitized = {}
        
        for key, value in data.items():
            if isinstance(value, str):
                sanitized[key] = self.sanitize_input(value)
            else:
                sanitized[key] = value
        
        return sanitized




