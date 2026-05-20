"""
Document Validation - Sistema de Validación Avanzado
=====================================================

Sistema de validación de documentos con reglas personalizables.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Severidad de validación."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationRule:
    """Regla de validación."""
    rule_id: str
    name: str
    description: str
    validator: Callable[[Any], bool]
    severity: ValidationSeverity = ValidationSeverity.WARNING
    enabled: bool = True
    error_message: Optional[str] = None


@dataclass
class ValidationResult:
    """Resultado de validación."""
    valid: bool
    passed_rules: List[str] = field(default_factory=list)
    failed_rules: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    score: float = 100.0


class DocumentValidator:
    """Validador de documentos."""
    
    def __init__(self, analyzer):
        """Inicializar validador."""
        self.analyzer = analyzer
        self.rules: Dict[str, ValidationRule] = {}
        self.validation_history: List[ValidationResult] = []
    
    def register_rule(self, rule: ValidationRule):
        """Registrar regla de validación."""
        self.rules[rule.rule_id] = rule
        logger.info(f"Regla de validación registrada: {rule.name}")
    
    async def validate_document(
        self,
        document_content: str,
        analysis_result: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ValidationResult:
        """
        Validar documento.
        
        Args:
            document_content: Contenido del documento
            analysis_result: Resultado de análisis (opcional)
            metadata: Metadatos (opcional)
        
        Returns:
            ValidationResult con resultado
        """
        passed_rules = []
        failed_rules = []
        warnings = []
        errors = []
        
        context = {
            "content": document_content,
            "analysis": analysis_result,
            "metadata": metadata or {}
        }
        
        # Ejecutar todas las reglas
        for rule_id, rule in self.rules.items():
            if not rule.enabled:
                continue
            
            try:
                is_valid = rule.validator(context)
                
                if is_valid:
                    passed_rules.append(rule_id)
                else:
                    failed_rules.append({
                        "rule_id": rule_id,
                        "name": rule.name,
                        "severity": rule.severity.value,
                        "message": rule.error_message or rule.description
                    })
                    
                    if rule.severity == ValidationSeverity.CRITICAL:
                        errors.append(f"{rule.name}: {rule.error_message or rule.description}")
                    elif rule.severity == ValidationSeverity.ERROR:
                        errors.append(f"{rule.name}: {rule.error_message or rule.description}")
                    elif rule.severity == ValidationSeverity.WARNING:
                        warnings.append(f"{rule.name}: {rule.error_message or rule.description}")
            except Exception as e:
                logger.error(f"Error ejecutando regla {rule_id}: {e}")
                failed_rules.append({
                    "rule_id": rule_id,
                    "name": rule.name,
                    "severity": "error",
                    "message": f"Error ejecutando regla: {str(e)}"
                })
                errors.append(f"Error en regla {rule.name}: {str(e)}")
        
        # Calcular score
        total_rules = len([r for r in self.rules.values() if r.enabled])
        if total_rules > 0:
            score = (len(passed_rules) / total_rules) * 100.0
        else:
            score = 100.0
        
        # Penalizar por errores críticos
        critical_errors = len([r for r in failed_rules if r["severity"] == "critical"])
        score -= critical_errors * 20
        
        score = max(0, min(100, score))
        
        result = ValidationResult(
            valid=len(errors) == 0,
            passed_rules=passed_rules,
            failed_rules=failed_rules,
            warnings=warnings,
            errors=errors,
            score=score
        )
        
        self.validation_history.append(result)
        
        return result
    
    def get_validation_history(self, limit: int = 100) -> List[ValidationResult]:
        """Obtener historial de validaciones."""
        return self.validation_history[-limit:]


# Reglas predefinidas
def create_min_length_rule(min_length: int = 100) -> ValidationRule:
    """Crear regla de longitud mínima."""
    return ValidationRule(
        rule_id="min_length",
        name="Longitud Mínima",
        description=f"El documento debe tener al menos {min_length} caracteres",
        validator=lambda ctx: len(ctx.get("content", "")) >= min_length,
        severity=ValidationSeverity.WARNING,
        error_message=f"Documento muy corto (mínimo {min_length} caracteres requeridos)"
    )


def create_has_summary_rule() -> ValidationRule:
    """Crear regla de resumen."""
    return ValidationRule(
        rule_id="has_summary",
        name="Tiene Resumen",
        description="El documento debe tener un resumen",
        validator=lambda ctx: ctx.get("analysis") and hasattr(ctx["analysis"], "summary") and ctx["analysis"].summary,
        severity=ValidationSeverity.INFO,
        error_message="Documento sin resumen"
    )


def create_quality_threshold_rule(threshold: float = 70.0) -> ValidationRule:
    """Crear regla de umbral de calidad."""
    return ValidationRule(
        rule_id="quality_threshold",
        name="Umbral de Calidad",
        description=f"El documento debe tener calidad >= {threshold}",
        validator=lambda ctx: (
            ctx.get("metadata", {}).get("quality_score", 100) >= threshold
        ),
        severity=ValidationSeverity.WARNING,
        error_message=f"Calidad del documento por debajo del umbral ({threshold})"
    )


__all__ = [
    "DocumentValidator",
    "ValidationRule",
    "ValidationResult",
    "ValidationSeverity",
    "create_min_length_rule",
    "create_has_summary_rule",
    "create_quality_threshold_rule"
]


