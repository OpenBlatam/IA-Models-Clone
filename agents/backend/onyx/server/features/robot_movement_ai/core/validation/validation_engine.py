"""
Validation Engine
=================

Motor de validación avanzado.
"""

import logging
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Nivel de validación."""
    STRICT = "strict"
    MODERATE = "moderate"
    LENIENT = "lenient"


@dataclass
class ValidationRule:
    """Regla de validación."""
    rule_id: str
    name: str
    validator: Callable[[Any], bool]
    error_message: str
    level: ValidationLevel = ValidationLevel.MODERATE
    enabled: bool = True


@dataclass
class ValidationResult:
    """Resultado de validación."""
    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ValidationEngine:
    """
    Motor de validación.
    
    Valida datos con reglas configurables.
    """
    
    def __init__(self, default_level: ValidationLevel = ValidationLevel.MODERATE):
        """
        Inicializar motor de validación.
        
        Args:
            default_level: Nivel de validación por defecto
        """
        self.rules: Dict[str, ValidationRule] = {}
        self.default_level = default_level
        self.validation_history: List[Dict[str, Any]] = []
    
    def add_rule(
        self,
        rule_id: str,
        name: str,
        validator: Callable[[Any], bool],
        error_message: str,
        level: Optional[ValidationLevel] = None,
        enabled: bool = True
    ) -> ValidationRule:
        """
        Agregar regla de validación.
        
        Args:
            rule_id: ID único de la regla
            name: Nombre de la regla
            validator: Función validadora
            error_message: Mensaje de error
            level: Nivel de validación
            enabled: Si está habilitada
            
        Returns:
            Regla creada
        """
        rule = ValidationRule(
            rule_id=rule_id,
            name=name,
            validator=validator,
            error_message=error_message,
            level=level or self.default_level,
            enabled=enabled
        )
        
        self.rules[rule_id] = rule
        logger.info(f"Added validation rule: {name} ({rule_id})")
        
        return rule
    
    def validate(
        self,
        data: Any,
        rule_ids: Optional[List[str]] = None,
        level: Optional[ValidationLevel] = None
    ) -> ValidationResult:
        """
        Validar datos.
        
        Args:
            data: Datos a validar
            rule_ids: IDs de reglas a aplicar (None = todas)
            level: Nivel de validación (None = usar default)
            
        Returns:
            Resultado de validación
        """
        validation_level = level or self.default_level
        errors = []
        warnings = []
        
        rules_to_apply = []
        if rule_ids:
            rules_to_apply = [self.rules[rid] for rid in rule_ids if rid in self.rules]
        else:
            rules_to_apply = list(self.rules.values())
        
        for rule in rules_to_apply:
            if not rule.enabled:
                continue
            
            # Verificar nivel
            if validation_level == ValidationLevel.STRICT:
                # Aplicar todas las reglas
                pass
            elif validation_level == ValidationLevel.MODERATE:
                # Aplicar reglas moderate y strict
                if rule.level == ValidationLevel.LENIENT:
                    continue
            elif validation_level == ValidationLevel.LENIENT:
                # Solo aplicar reglas lenient
                if rule.level != ValidationLevel.LENIENT:
                    continue
            
            try:
                if not rule.validator(data):
                    if rule.level == ValidationLevel.STRICT:
                        errors.append(rule.error_message)
                    else:
                        warnings.append(rule.error_message)
            except Exception as e:
                logger.error(f"Error in validation rule {rule.rule_id}: {e}")
                errors.append(f"Validation error: {str(e)}")
        
        result = ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
        
        # Registrar en historial
        self.validation_history.append({
            "data_type": type(data).__name__,
            "valid": result.valid,
            "errors_count": len(errors),
            "warnings_count": len(warnings),
            "level": validation_level.value
        })
        
        return result
    
    def get_validation_statistics(self) -> Dict[str, Any]:
        """Obtener estadísticas de validación."""
        if not self.validation_history:
            return {
                "total_validations": 0,
                "success_rate": 0.0,
                "average_errors": 0.0
            }
        
        total = len(self.validation_history)
        successful = sum(1 for v in self.validation_history if v["valid"])
        total_errors = sum(v["errors_count"] for v in self.validation_history)
        
        return {
            "total_validations": total,
            "successful_validations": successful,
            "failed_validations": total - successful,
            "success_rate": successful / total if total > 0 else 0.0,
            "average_errors": total_errors / total if total > 0 else 0.0
        }


# Instancia global
_validation_engine: Optional[ValidationEngine] = None


def get_validation_engine() -> ValidationEngine:
    """Obtener instancia global del motor de validación."""
    global _validation_engine
    if _validation_engine is None:
        _validation_engine = ValidationEngine()
    return _validation_engine






