"""
Validators for Contabilidad Mexicana AI
========================================

Input validation with:
- ValidationRule protocol for extensibility
- ValidationRegistry for rule management
- Dataclass for validation results
- Decorator for automatic validation
"""

from typing import Dict, Any, List, Optional, Callable, Protocol, TypeVar, Set
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
from functools import wraps
import logging

logger = logging.getLogger(__name__)

T = TypeVar('T')


class ValidationError(Exception):
    """Custom exception for validation errors."""
    
    def __init__(self, message: str, field: Optional[str] = None, code: Optional[str] = None):
        super().__init__(message)
        self.message = message
        self.field = field
        self.code = code


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationIssue:
    """Represents a single validation issue."""
    message: str
    field: Optional[str] = None
    code: Optional[str] = None
    severity: ValidationSeverity = ValidationSeverity.ERROR
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "message": self.message,
            "field": self.field,
            "code": self.code,
            "severity": self.severity.value,
        }


@dataclass
class ValidationResult:
    """Result of validation."""
    is_valid: bool = True
    issues: List[ValidationIssue] = field(default_factory=list)
    
    def add_error(self, message: str, field: Optional[str] = None, code: Optional[str] = None):
        """Add an error."""
        self.issues.append(ValidationIssue(message, field, code, ValidationSeverity.ERROR))
        self.is_valid = False
    
    def add_warning(self, message: str, field: Optional[str] = None, code: Optional[str] = None):
        """Add a warning (doesn't affect validity)."""
        self.issues.append(ValidationIssue(message, field, code, ValidationSeverity.WARNING))
    
    @property
    def errors(self) -> List[ValidationIssue]:
        """Get only errors."""
        return [i for i in self.issues if i.severity == ValidationSeverity.ERROR]
    
    @property
    def warnings(self) -> List[ValidationIssue]:
        """Get only warnings."""
        return [i for i in self.issues if i.severity == ValidationSeverity.WARNING]
    
    def raise_if_invalid(self):
        """Raise ValidationError if invalid."""
        if not self.is_valid and self.errors:
            raise ValidationError(
                self.errors[0].message,
                self.errors[0].field,
                self.errors[0].code
            )


class ValidationRule(Protocol):
    """Protocol for validation rules."""
    
    def validate(self, value: Any, context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """Validate a value."""
        ...


class BaseValidationRule(ABC):
    """Abstract base class for validation rules."""
    
    @property
    @abstractmethod
    def rule_name(self) -> str:
        """Name of this rule."""
        pass
    
    @abstractmethod
    def validate(self, value: Any, context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """Validate a value."""
        pass


class RequiredRule(BaseValidationRule):
    """Validates that a value is not empty."""
    
    rule_name = "required"
    
    def __init__(self, field_name: str, message: Optional[str] = None):
        self.field_name = field_name
        self.message = message or f"{field_name} es requerido"
    
    def validate(self, value: Any, context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        result = ValidationResult()
        if not value or (isinstance(value, str) and not value.strip()):
            result.add_error(self.message, self.field_name, "required")
        return result


class ChoiceRule(BaseValidationRule):
    """Validates that a value is in a set of choices."""
    
    rule_name = "choice"
    
    def __init__(self, field_name: str, choices: Set[str], message: Optional[str] = None):
        self.field_name = field_name
        self.choices = choices
        self.message = message
    
    def validate(self, value: Any, context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        result = ValidationResult()
        if value and value not in self.choices:
            msg = self.message or f"{self.field_name} inválido: {value}. Válidos: {', '.join(sorted(self.choices))}"
            result.add_error(msg, self.field_name, "invalid_choice")
        return result


class LengthRule(BaseValidationRule):
    """Validates string length."""
    
    rule_name = "length"
    
    def __init__(
        self, 
        field_name: str, 
        min_length: Optional[int] = None, 
        max_length: Optional[int] = None
    ):
        self.field_name = field_name
        self.min_length = min_length
        self.max_length = max_length
    
    def validate(self, value: Any, context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        result = ValidationResult()
        if value and isinstance(value, str):
            if self.min_length and len(value) < self.min_length:
                result.add_error(
                    f"{self.field_name} debe tener al menos {self.min_length} caracteres",
                    self.field_name,
                    "min_length"
                )
            if self.max_length and len(value) > self.max_length:
                result.add_error(
                    f"{self.field_name} no puede exceder {self.max_length} caracteres",
                    self.field_name,
                    "max_length"
                )
        return result


class RangeRule(BaseValidationRule):
    """Validates numeric range."""
    
    rule_name = "range"
    
    def __init__(
        self, 
        field_name: str, 
        min_value: Optional[float] = None, 
        max_value: Optional[float] = None
    ):
        self.field_name = field_name
        self.min_value = min_value
        self.max_value = max_value
    
    def validate(self, value: Any, context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        result = ValidationResult()
        if value is not None:
            try:
                num_value = float(value)
                if self.min_value is not None and num_value < self.min_value:
                    result.add_error(
                        f"{self.field_name} no puede ser menor a {self.min_value}",
                        self.field_name,
                        "min_value"
                    )
                if self.max_value is not None and num_value > self.max_value:
                    result.add_error(
                        f"{self.field_name} no puede exceder {self.max_value}",
                        self.field_name,
                        "max_value"
                    )
            except (ValueError, TypeError):
                pass
        return result


class ValidationRegistry:
    """Registry for validation rules."""
    
    def __init__(self):
        self._validators: Dict[str, List[BaseValidationRule]] = {}
    
    def register(self, validator_name: str, rule: BaseValidationRule):
        """Register a rule for a validator."""
        if validator_name not in self._validators:
            self._validators[validator_name] = []
        self._validators[validator_name].append(rule)
    
    def validate(
        self, 
        validator_name: str, 
        values: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> ValidationResult:
        """Run all rules for a validator."""
        combined_result = ValidationResult()
        
        rules = self._validators.get(validator_name, [])
        for rule in rules:
            # Get the value for this rule's field
            value = values.get(rule.field_name) if hasattr(rule, 'field_name') else values
            result = rule.validate(value, context)
            
            for issue in result.issues:
                if issue.severity == ValidationSeverity.ERROR:
                    combined_result.add_error(issue.message, issue.field, issue.code)
                else:
                    combined_result.add_warning(issue.message, issue.field, issue.code)
        
        return combined_result


# === Constants ===

REGIMENES_VALIDOS = {
    "RESICO", "PFAE", "Plataformas", "Sueldos y Salarios",
    "Personas Físicas", "Personas Morales"
}

TIPOS_IMPUESTOS_VALIDOS = {"ISR", "IVA", "IEPS"}

NIVELES_DETALLE_VALIDOS = {"básico", "intermedio", "completo"}

TIPOS_DECLARACION_VALIDOS = {"mensual", "anual", "bimestral", "trimestral"}


# === ContadorValidator (refactored with registry) ===

class ContadorValidator:
    """Validates inputs for Contador AI services."""
    
    REGIMENES_VALIDOS = list(REGIMENES_VALIDOS)
    TIPOS_IMPUESTOS_VALIDOS = list(TIPOS_IMPUESTOS_VALIDOS)
    NIVELES_DETALLE_VALIDOS = list(NIVELES_DETALLE_VALIDOS)
    TIPOS_DECLARACION_VALIDOS = list(TIPOS_DECLARACION_VALIDOS)
    
    _registry = ValidationRegistry()
    
    @classmethod
    def _init_registry(cls):
        """Initialize validation rules."""
        # Regimen rules
        cls._registry.register("regimen", RequiredRule("regimen", "El régimen fiscal es requerido"))
        cls._registry.register("regimen", ChoiceRule("regimen", REGIMENES_VALIDOS))
        
        # Impuesto rules
        cls._registry.register("tipo_impuesto", RequiredRule("tipo_impuesto", "El tipo de impuesto es requerido"))
        cls._registry.register("tipo_impuesto", ChoiceRule("tipo_impuesto", TIPOS_IMPUESTOS_VALIDOS))
        
        # Pregunta rules
        cls._registry.register("pregunta", RequiredRule("pregunta", "La pregunta es requerida"))
        cls._registry.register("pregunta", LengthRule("pregunta", min_length=10, max_length=2000))
        
        # Tema rules
        cls._registry.register("tema", RequiredRule("tema", "El tema es requerido"))
        cls._registry.register("tema", LengthRule("tema", min_length=5, max_length=500))
        
        # Nivel detalle rules
        cls._registry.register("nivel_detalle", ChoiceRule("nivel_detalle", NIVELES_DETALLE_VALIDOS))
        
        # Tipo declaracion rules
        cls._registry.register("tipo_declaracion", RequiredRule("tipo_declaracion", "El tipo de declaración es requerido"))
        cls._registry.register("tipo_declaracion", ChoiceRule("tipo_declaracion", TIPOS_DECLARACION_VALIDOS))
    
    @staticmethod
    def validate_regimen(regimen: str) -> None:
        """Validate fiscal regime."""
        if not regimen:
            raise ValidationError("El régimen fiscal es requerido", "regimen")
        
        if regimen not in REGIMENES_VALIDOS:
            raise ValidationError(
                f"Régimen fiscal inválido: {regimen}. "
                f"Válidos: {', '.join(sorted(REGIMENES_VALIDOS))}",
                "regimen"
            )
    
    @staticmethod
    def validate_tipo_impuesto(tipo_impuesto: str) -> None:
        """Validate tax type."""
        if not tipo_impuesto:
            raise ValidationError("El tipo de impuesto es requerido", "tipo_impuesto")
        
        if tipo_impuesto not in TIPOS_IMPUESTOS_VALIDOS:
            raise ValidationError(
                f"Tipo de impuesto inválido: {tipo_impuesto}. "
                f"Válidos: {', '.join(sorted(TIPOS_IMPUESTOS_VALIDOS))}",
                "tipo_impuesto"
            )
    
    @staticmethod
    def validate_datos_calculo(datos: Dict[str, Any], regimen: str, tipo_impuesto: str) -> None:
        """Validate calculation data."""
        if not datos:
            raise ValidationError("Los datos para el cálculo son requeridos", "datos")
        
        # Validaciones específicas por régimen e impuesto
        if regimen == "RESICO" and tipo_impuesto == "ISR":
            if "ingresos_mensuales" not in datos and "ingresos_anuales" not in datos:
                raise ValidationError(
                    "Para RESICO con ISR se requiere 'ingresos_mensuales' o 'ingresos_anuales'",
                    "datos"
                )
            
            ingresos = datos.get("ingresos_mensuales") or datos.get("ingresos_anuales", 0)
            if ingresos < 0:
                raise ValidationError("Los ingresos no pueden ser negativos", "ingresos")
        
        elif tipo_impuesto == "IVA":
            if "base_imponible" not in datos:
                raise ValidationError("Para IVA se requiere 'base_imponible'", "datos")
            
            base = datos.get("base_imponible", 0)
            if base < 0:
                raise ValidationError("La base imponible no puede ser negativa", "base_imponible")
    
    @staticmethod
    def validate_pregunta(pregunta: str) -> None:
        """Validate question."""
        if not pregunta or not pregunta.strip():
            raise ValidationError("La pregunta es requerida", "pregunta")
        
        if len(pregunta) < 10:
            raise ValidationError("La pregunta debe tener al menos 10 caracteres", "pregunta")
        
        if len(pregunta) > 2000:
            raise ValidationError("La pregunta no puede exceder 2000 caracteres", "pregunta")
    
    @staticmethod
    def validate_tema(tema: str) -> None:
        """Validate topic."""
        if not tema or not tema.strip():
            raise ValidationError("El tema es requerido", "tema")
        
        if len(tema) < 5:
            raise ValidationError("El tema debe tener al menos 5 caracteres", "tema")
        
        if len(tema) > 500:
            raise ValidationError("El tema no puede exceder 500 caracteres", "tema")
    
    @staticmethod
    def validate_nivel_detalle(nivel_detalle: str) -> None:
        """Validate detail level."""
        if nivel_detalle not in NIVELES_DETALLE_VALIDOS:
            raise ValidationError(
                f"Nivel de detalle inválido: {nivel_detalle}. "
                f"Válidos: {', '.join(sorted(NIVELES_DETALLE_VALIDOS))}",
                "nivel_detalle"
            )
    
    @staticmethod
    def validate_tipo_tramite(tipo_tramite: str) -> None:
        """Validate procedure type."""
        if not tipo_tramite or not tipo_tramite.strip():
            raise ValidationError("El tipo de trámite es requerido", "tipo_tramite")
        
        if len(tipo_tramite) < 5:
            raise ValidationError("El tipo de trámite debe tener al menos 5 caracteres", "tipo_tramite")
    
    @staticmethod
    def validate_tipo_declaracion(tipo_declaracion: str) -> None:
        """Validate declaration type."""
        if not tipo_declaracion:
            raise ValidationError("El tipo de declaración es requerido", "tipo_declaracion")
        
        if tipo_declaracion not in TIPOS_DECLARACION_VALIDOS:
            raise ValidationError(
                f"Tipo de declaración inválido: {tipo_declaracion}. "
                f"Válidos: {', '.join(sorted(TIPOS_DECLARACION_VALIDOS))}",
                "tipo_declaracion"
            )
    
    @staticmethod
    def validate_periodo(periodo: str) -> None:
        """Validate fiscal period."""
        if not periodo or not periodo.strip():
            raise ValidationError("El período fiscal es requerido", "periodo")
        
        # Validar formato básico (YYYY-MM o YYYY)
        if len(periodo) not in [4, 7]:
            raise ValidationError("El período debe tener formato YYYY o YYYY-MM", "periodo")
        
        if len(periodo) == 7 and periodo[4] != "-":
            raise ValidationError("El período mensual debe tener formato YYYY-MM", "periodo")
    
    # === Composite Validators ===
    
    @staticmethod
    def validate_calculo_impuestos(
        regimen: str,
        tipo_impuesto: str,
        datos: Dict[str, Any]
    ) -> None:
        """Validate all inputs for tax calculation."""
        ContadorValidator.validate_regimen(regimen)
        ContadorValidator.validate_tipo_impuesto(tipo_impuesto)
        ContadorValidator.validate_datos_calculo(datos, regimen, tipo_impuesto)
    
    @staticmethod
    def validate_asesoria_fiscal(
        pregunta: str,
        contexto: Optional[Dict[str, Any]] = None
    ) -> None:
        """Validate all inputs for fiscal advice."""
        ContadorValidator.validate_pregunta(pregunta)
        
        if contexto and not isinstance(contexto, dict):
            raise ValidationError("El contexto debe ser un diccionario", "contexto")
    
    @staticmethod
    def validate_guia_fiscal(
        tema: str,
        nivel_detalle: str = "completo"
    ) -> None:
        """Validate all inputs for fiscal guide."""
        ContadorValidator.validate_tema(tema)
        ContadorValidator.validate_nivel_detalle(nivel_detalle)
    
    @staticmethod
    def validate_tramite_sat(
        tipo_tramite: str,
        detalles: Optional[Dict[str, Any]] = None
    ) -> None:
        """Validate all inputs for SAT procedure."""
        ContadorValidator.validate_tipo_tramite(tipo_tramite)
        
        if detalles and not isinstance(detalles, dict):
            raise ValidationError("Los detalles deben ser un diccionario", "detalles")
    
    @staticmethod
    def validate_ayuda_declaracion(
        tipo_declaracion: str,
        periodo: str,
        datos: Optional[Dict[str, Any]] = None
    ) -> None:
        """Validate all inputs for declaration assistance."""
        ContadorValidator.validate_tipo_declaracion(tipo_declaracion)
        ContadorValidator.validate_periodo(periodo)
        
        if datos and not isinstance(datos, dict):
            raise ValidationError("Los datos deben ser un diccionario", "datos")


# Initialize registry
ContadorValidator._init_registry()
