"""
Content Validator - Sistema de validación de contenido mejorado
"""

import logging
import re
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Niveles de validación"""
    STRICT = "strict"
    MODERATE = "moderate"
    LENIENT = "lenient"


@dataclass
class ValidationRule:
    """Regla de validación"""
    name: str
    description: str
    validator: Callable
    level: ValidationLevel
    enabled: bool = True


class ContentValidator:
    """Validador de contenido mejorado"""

    def __init__(self, validation_level: ValidationLevel = ValidationLevel.MODERATE):
        """
        Inicializar validador.

        Args:
            validation_level: Nivel de validación
        """
        self.validation_level = validation_level
        self.rules: List[ValidationRule] = []
        self._register_default_rules()

    def _register_default_rules(self):
        """Registrar reglas por defecto"""
        # Regla: contenido no vacío
        def non_empty_validator(content: str) -> tuple[bool, Optional[str]]:
            if not content or not content.strip():
                return False, "El contenido no puede estar vacío"
            return True, None
        
        self.rules.append(ValidationRule(
            name="non_empty",
            description="Verificar que el contenido no esté vacío",
            validator=non_empty_validator,
            level=ValidationLevel.STRICT
        ))
        
        # Regla: longitud mínima
        def min_length_validator(content: str) -> tuple[bool, Optional[str]]:
            if len(content.strip()) < 10:
                return False, "El contenido es demasiado corto (mínimo 10 caracteres)"
            return True, None
        
        self.rules.append(ValidationRule(
            name="min_length",
            description="Verificar longitud mínima",
            validator=min_length_validator,
            level=ValidationLevel.MODERATE
        ))
        
        # Regla: sin caracteres especiales peligrosos
        def safe_chars_validator(content: str) -> tuple[bool, Optional[str]]:
            dangerous = ['<script', 'javascript:', 'onerror=', 'onclick=']
            for pattern in dangerous:
                if pattern.lower() in content.lower():
                    return False, f"Contenido potencialmente peligroso detectado: {pattern}"
            return True, None
        
        self.rules.append(ValidationRule(
            name="safe_chars",
            description="Verificar caracteres seguros",
            validator=safe_chars_validator,
            level=ValidationLevel.STRICT
        ))
        
        # Regla: formato válido
        def format_validator(content: str) -> tuple[bool, Optional[str]]:
            # Verificar que no tenga demasiados espacios consecutivos
            if re.search(r' {3,}', content):
                return False, "Demasiados espacios consecutivos"
            return True, None
        
        self.rules.append(ValidationRule(
            name="format",
            description="Verificar formato básico",
            validator=format_validator,
            level=ValidationLevel.LENIENT
        ))

    def validate(
        self,
        content: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Validar contenido.

        Args:
            content: Contenido a validar
            context: Contexto adicional

        Returns:
            Resultado de validación
        """
        errors = []
        warnings = []
        passed_rules = []
        
        # Aplicar reglas según nivel
        for rule in self.rules:
            if not rule.enabled:
                continue
            
            # Filtrar por nivel de validación
            if self.validation_level == ValidationLevel.STRICT:
                # Aplicar todas las reglas
                pass
            elif self.validation_level == ValidationLevel.MODERATE:
                # Aplicar solo STRICT y MODERATE
                if rule.level == ValidationLevel.LENIENT:
                    continue
            elif self.validation_level == ValidationLevel.LENIENT:
                # Aplicar solo STRICT
                if rule.level != ValidationLevel.STRICT:
                    continue
            
            try:
                is_valid, message = rule.validator(content)
                
                if is_valid:
                    passed_rules.append(rule.name)
                else:
                    if rule.level == ValidationLevel.STRICT:
                        errors.append({
                            "rule": rule.name,
                            "message": message,
                            "level": rule.level.value
                        })
                    else:
                        warnings.append({
                            "rule": rule.name,
                            "message": message,
                            "level": rule.level.value
                        })
            except Exception as e:
                logger.error(f"Error en regla {rule.name}: {e}")
                warnings.append({
                    "rule": rule.name,
                    "message": f"Error al validar: {str(e)}",
                    "level": "error"
                })
        
        is_valid = len(errors) == 0
        
        return {
            "valid": is_valid,
            "errors": errors,
            "warnings": warnings,
            "passed_rules": passed_rules,
            "total_rules": len(self.rules),
            "validation_level": self.validation_level.value
        }

    def register_rule(
        self,
        name: str,
        description: str,
        validator: Callable,
        level: ValidationLevel = ValidationLevel.MODERATE
    ):
        """
        Registrar nueva regla.

        Args:
            name: Nombre de la regla
            description: Descripción
            validator: Función validadora
            level: Nivel de validación
        """
        rule = ValidationRule(
            name=name,
            description=description,
            validator=validator,
            level=level
        )
        self.rules.append(rule)
        logger.info(f"Regla de validación registrada: {name}")

    def set_validation_level(self, level: ValidationLevel):
        """
        Establecer nivel de validación.

        Args:
            level: Nivel de validación
        """
        self.validation_level = level
        logger.info(f"Nivel de validación establecido: {level.value}")






