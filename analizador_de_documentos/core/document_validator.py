"""
Validador de Documentos
========================

Sistema para validar documentos según reglas personalizadas.
"""

import logging
import re
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Severidad de validación"""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationRule:
    """Regla de validación"""
    name: str
    description: str
    severity: ValidationSeverity
    validator: Callable
    error_message: str
    enabled: bool = True


@dataclass
class ValidationResult:
    """Resultado de validación"""
    is_valid: bool
    errors: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[Dict[str, Any]] = field(default_factory=list)
    info: List[Dict[str, Any]] = field(default_factory=list)
    score: float = 0.0
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
        
        # Calcular score (0-100)
        total_issues = len(self.errors) + len(self.warnings) + len(self.info)
        if total_issues == 0:
            self.score = 100.0
        else:
            # Penalizar errores más que warnings
            error_penalty = len(self.errors) * 10
            warning_penalty = len(self.warnings) * 5
            info_penalty = len(self.info) * 1
            penalty = min(100, error_penalty + warning_penalty + info_penalty)
            self.score = max(0, 100 - penalty)


class DocumentValidator:
    """
    Validador de documentos con reglas personalizadas
    
    Permite definir y aplicar reglas de validación personalizadas
    para verificar la calidad, estructura y contenido de documentos.
    """
    
    def __init__(self):
        """Inicializar validador"""
        self.rules: Dict[str, ValidationRule] = {}
        self._register_default_rules()
        logger.info("DocumentValidator inicializado")
    
    def _register_default_rules(self):
        """Registrar reglas de validación por defecto"""
        # Regla: Longitud mínima
        self.add_rule(
            name="min_length",
            description="Documento debe tener longitud mínima",
            severity=ValidationSeverity.WARNING,
            validator=lambda content, min_len=100: len(content) >= min_len,
            error_message="Documento muy corto (mínimo {min_len} caracteres)",
            enabled=True
        )
        
        # Regla: Longitud máxima
        self.add_rule(
            name="max_length",
            description="Documento no debe exceder longitud máxima",
            severity=ValidationSeverity.WARNING,
            validator=lambda content, max_len=100000: len(content) <= max_len,
            error_message="Documento muy largo (máximo {max_len} caracteres)",
            enabled=True
        )
        
        # Regla: Contiene caracteres válidos
        self.add_rule(
            name="valid_characters",
            description="Documento debe contener caracteres válidos",
            severity=ValidationSeverity.ERROR,
            validator=lambda content: bool(re.search(r'[a-zA-Z0-9]', content)),
            error_message="Documento no contiene caracteres válidos",
            enabled=True
        )
        
        # Regla: Tiene párrafos
        self.add_rule(
            name="has_paragraphs",
            description="Documento debe tener párrafos",
            severity=ValidationSeverity.INFO,
            validator=lambda content: len(content.split('\n\n')) > 1,
            error_message="Documento debería tener múltiples párrafos",
            enabled=True
        )
        
        # Regla: No tiene demasiados espacios
        self.add_rule(
            name="no_excessive_spaces",
            description="Documento no debe tener espacios excesivos",
            severity=ValidationSeverity.WARNING,
            validator=lambda content: not bool(re.search(r'\s{4,}', content)),
            error_message="Documento contiene espacios excesivos",
            enabled=True
        )
    
    def add_rule(
        self,
        name: str,
        description: str,
        severity: ValidationSeverity,
        validator: Callable,
        error_message: str,
        enabled: bool = True
    ):
        """
        Agregar regla de validación
        
        Args:
            name: Nombre único de la regla
            description: Descripción de la regla
            severity: Severidad de la validación
            validator: Función que valida (debe retornar bool)
            error_message: Mensaje de error si falla
            enabled: Si la regla está habilitada
        """
        rule = ValidationRule(
            name=name,
            description=description,
            severity=severity,
            validator=validator,
            error_message=error_message,
            enabled=enabled
        )
        self.rules[name] = rule
        logger.debug(f"Regla agregada: {name}")
    
    def remove_rule(self, name: str):
        """Eliminar regla de validación"""
        if name in self.rules:
            del self.rules[name]
            logger.debug(f"Regla eliminada: {name}")
    
    def enable_rule(self, name: str):
        """Habilitar regla"""
        if name in self.rules:
            self.rules[name].enabled = True
    
    def disable_rule(self, name: str):
        """Deshabilitar regla"""
        if name in self.rules:
            self.rules[name].enabled = False
    
    async def validate(
        self,
        content: str,
        rules: Optional[List[str]] = None,
        custom_params: Optional[Dict[str, Any]] = None
    ) -> ValidationResult:
        """
        Validar documento
        
        Args:
            content: Contenido del documento
            rules: Lista de nombres de reglas a aplicar (None = todas)
            custom_params: Parámetros personalizados para reglas
        
        Returns:
            ValidationResult con resultados de validación
        """
        custom_params = custom_params or {}
        rules_to_apply = rules or list(self.rules.keys())
        
        errors = []
        warnings = []
        info = []
        
        for rule_name in rules_to_apply:
            if rule_name not in self.rules:
                continue
            
            rule = self.rules[rule_name]
            if not rule.enabled:
                continue
            
            try:
                # Ejecutar validador
                # Si el validador acepta parámetros, pasarlos
                if callable(rule.validator):
                    # Intentar pasar content y custom_params
                    try:
                        is_valid = rule.validator(content, **custom_params)
                    except TypeError:
                        # Si no acepta parámetros adicionales, solo pasar content
                        is_valid = rule.validator(content)
                else:
                    is_valid = False
                
                if not is_valid:
                    issue = {
                        "rule": rule_name,
                        "message": rule.error_message.format(**custom_params),
                        "description": rule.description
                    }
                    
                    if rule.severity == ValidationSeverity.ERROR:
                        errors.append(issue)
                    elif rule.severity == ValidationSeverity.WARNING:
                        warnings.append(issue)
                    else:
                        info.append(issue)
            except Exception as e:
                logger.error(f"Error ejecutando regla {rule_name}: {e}")
                errors.append({
                    "rule": rule_name,
                    "message": f"Error ejecutando regla: {str(e)}",
                    "description": rule.description
                })
        
        result = ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            info=info
        )
        
        return result
    
    def get_rules(self) -> List[Dict[str, Any]]:
        """Obtener lista de reglas"""
        return [
            {
                "name": rule.name,
                "description": rule.description,
                "severity": rule.severity.value,
                "enabled": rule.enabled
            }
            for rule in self.rules.values()
        ]
















