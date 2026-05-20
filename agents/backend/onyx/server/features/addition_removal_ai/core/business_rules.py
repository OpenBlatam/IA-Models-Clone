"""
Business Rules - Sistema de validación de reglas de negocio
"""

import logging
import re
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class RuleSeverity(Enum):
    """Severidad de reglas"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    BLOCKING = "blocking"


@dataclass
class BusinessRule:
    """Regla de negocio"""
    name: str
    description: str
    condition: Callable
    severity: RuleSeverity
    message: str
    enabled: bool = True


class BusinessRulesEngine:
    """Motor de reglas de negocio"""

    def __init__(self):
        """Inicializar motor de reglas"""
        self.rules: List[BusinessRule] = []
        self.violations: List[Dict[str, Any]] = []

    def register_rule(
        self,
        name: str,
        description: str,
        condition: Callable,
        severity: RuleSeverity = RuleSeverity.WARNING,
        message: Optional[str] = None
    ):
        """
        Registrar una regla de negocio.

        Args:
            name: Nombre de la regla
            description: Descripción
            condition: Función condición
            severity: Severidad
            message: Mensaje personalizado
        """
        rule = BusinessRule(
            name=name,
            description=description,
            condition=condition,
            severity=severity,
            message=message or f"Violación de regla: {name}"
        )
        
        self.rules.append(rule)
        logger.info(f"Regla de negocio registrada: {name}")

    def validate(
        self,
        content: str,
        operation: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Validar contra reglas de negocio.

        Args:
            content: Contenido
            operation: Tipo de operación
            context: Contexto adicional

        Returns:
            Resultado de la validación
        """
        violations = []
        blocking = False
        
        for rule in self.rules:
            if not rule.enabled:
                continue
            
            try:
                # Evaluar condición
                if rule.condition(content, operation, context or {}):
                    violation = {
                        "rule": rule.name,
                        "description": rule.description,
                        "severity": rule.severity.value,
                        "message": rule.message,
                        "timestamp": __import__("datetime").datetime.utcnow().isoformat()
                    }
                    
                    violations.append(violation)
                    self.violations.append(violation)
                    
                    if rule.severity == RuleSeverity.BLOCKING:
                        blocking = True
            except Exception as e:
                logger.error(f"Error evaluando regla {rule.name}: {e}")
        
        return {
            "valid": len(violations) == 0,
            "blocking": blocking,
            "violations": violations,
            "violation_count": len(violations)
        }

    def get_violations(
        self,
        severity: Optional[RuleSeverity] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Obtener violaciones.

        Args:
            severity: Filtrar por severidad
            limit: Límite de resultados

        Returns:
            Lista de violaciones
        """
        violations = self.violations
        
        if severity:
            violations = [v for v in violations if v["severity"] == severity.value]
        
        return violations[-limit:][::-1]

    def clear_violations(self):
        """Limpiar violaciones"""
        self.violations.clear()


# Reglas de negocio predefinidas
def create_default_rules() -> List[BusinessRule]:
    """Crear reglas de negocio por defecto"""
    rules = []
    
    # Regla: contenido no vacío
    def non_empty_rule(content: str, operation: str, context: Dict) -> bool:
        return len(content.strip()) == 0
    
    rules.append(BusinessRule(
        name="non_empty_content",
        description="El contenido no debe estar vacío",
        condition=non_empty_rule,
        severity=RuleSeverity.ERROR,
        message="El contenido está vacío"
    ))
    
    # Regla: longitud máxima
    def max_length_rule(content: str, operation: str, context: Dict) -> bool:
        max_length = context.get("max_length", 1000000)
        return len(content) > max_length
    
    rules.append(BusinessRule(
        name="max_length",
        description="El contenido excede la longitud máxima",
        condition=max_length_rule,
        severity=RuleSeverity.WARNING,
        message="El contenido es muy largo"
    ))
    
    # Regla: sin caracteres prohibidos
    def forbidden_chars_rule(content: str, operation: str, context: Dict) -> bool:
        forbidden = context.get("forbidden_chars", [])
        return any(char in content for char in forbidden)
    
    rules.append(BusinessRule(
        name="forbidden_chars",
        description="El contenido contiene caracteres prohibidos",
        condition=forbidden_chars_rule,
        severity=RuleSeverity.ERROR,
        message="Caracteres prohibidos detectados"
    ))
    
    return rules






