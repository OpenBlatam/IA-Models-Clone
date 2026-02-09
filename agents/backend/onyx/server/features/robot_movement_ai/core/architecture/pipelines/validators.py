"""
Pipeline Validators
==================

Validadores para datos y etapas de pipeline.
Optimizado siguiendo mejores prácticas de FastAPI y Python.
"""

from typing import Dict, Any, Optional, Callable, TypeVar, List, Tuple
from abc import ABC, abstractmethod
from pydantic import BaseModel, ValidationError, field_validator

T = TypeVar('T')


class ValidationResult(BaseModel):
    """Resultado de validación con mensaje de error estructurado."""
    is_valid: bool
    error_message: Optional[str] = None


def validate_stage(
    stage_name: str,
    data: T,
    context: Optional[Dict[str, Any]] = None,
    validator: Optional[Callable[[T, Optional[Dict[str, Any]]], bool]] = None
) -> ValidationResult:
    """
    Validar etapa de pipeline de forma funcional.
    
    Args:
        stage_name: Nombre de la etapa
        data: Datos a validar
        context: Contexto opcional
        validator: Función validadora opcional
        
    Returns:
        ValidationResult con el resultado de la validación
    """
    if not stage_name:
        return ValidationResult(
            is_valid=False,
            error_message="Stage name cannot be empty"
        )
    
    if validator is None:
        return ValidationResult(is_valid=True)
    
    try:
        is_valid = validator(data, context)
        if not is_valid:
            return ValidationResult(
                is_valid=False,
                error_message=f"Validation failed for stage: {stage_name}"
            )
        return ValidationResult(is_valid=True)
    except Exception as e:
        return ValidationResult(
            is_valid=False,
            error_message=f"Validation error in stage {stage_name}: {str(e)}"
        )


def validate_data(
    data: T,
    validator: Optional[Callable[[T], bool]] = None,
    error_message: Optional[str] = None
) -> ValidationResult:
    """
    Validar datos de forma funcional.
    
    Args:
        data: Datos a validar
        validator: Función validadora opcional
        error_message: Mensaje de error personalizado
        
    Returns:
        ValidationResult con el resultado de la validación
    """
    if validator is None:
        return ValidationResult(is_valid=True)
    
    try:
        is_valid = validator(data)
        if not is_valid:
            return ValidationResult(
                is_valid=False,
                error_message=error_message or "Data validation failed"
            )
        return ValidationResult(is_valid=True)
    except Exception as e:
        return ValidationResult(
            is_valid=False,
            error_message=f"Validation error: {str(e)}"
        )


class ContextValidatorConfig(BaseModel):
    """Configuración para validador de contexto usando Pydantic."""
    required_keys: List[str] = []
    key_validators: Dict[str, str] = {}


def validate_context(
    context: Optional[Dict[str, Any]],
    required_keys: Optional[List[str]] = None,
    key_validators: Optional[Dict[str, Callable[[Any], bool]]] = None
) -> Tuple[bool, Optional[str]]:
    """
    Validar contexto de pipeline de forma funcional.
    
    Usa guard clauses para manejo temprano de errores.
    
    Args:
        context: Contexto a validar
        required_keys: Claves requeridas
        key_validators: Validadores por clave
        
    Returns:
        Tupla (es_válido, mensaje_error)
    """
    if context is None:
        context = {}
    
    if not isinstance(context, dict):
        return False, "Context must be a dictionary"
    
    required_keys = required_keys or []
    key_validators = key_validators or {}
    
    for key in required_keys:
        if key not in context:
            return False, f"Required key missing: {key}"
    
    for key, validator in key_validators.items():
        if key not in context:
            continue
        
        try:
            if not validator(context[key]):
                return False, f"Validation failed for key: {key}"
        except Exception as e:
            return False, f"Validation error for key {key}: {str(e)}"
    
    return True, None


class StageValidator(ABC):
    """Interfaz base para validadores de etapas."""

    @abstractmethod
    def validate(
        self,
        stage_name: str,
        data: T,
        context: Optional[Dict[str, Any]] = None
    ) -> ValidationResult:
        """
        Validar etapa.

        Args:
            stage_name: Nombre de la etapa
            data: Datos
            context: Contexto

        Returns:
            ValidationResult con el resultado
        """
        pass


class DataValidator(ABC):
    """Interfaz base para validadores de datos."""

    @abstractmethod
    def validate(self, data: T) -> ValidationResult:
        """
        Validar datos.

        Args:
            data: Datos a validar

        Returns:
            ValidationResult con el resultado
        """
        pass


class ContextValidator:
    """
    Validador para contexto de pipeline.
    Optimizado con guard clauses y mejor manejo de errores.
    """

    def __init__(
        self,
        required_keys: Optional[List[str]] = None,
        key_validators: Optional[Dict[str, Callable[[Any], bool]]] = None
    ):
        """
        Inicializar validador de contexto.

        Args:
            required_keys: Claves requeridas
            key_validators: Validadores por clave
        """
        self.required_keys = required_keys or []
        self.key_validators = key_validators or {}

    def validate(
        self,
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Validar contexto usando función funcional.

        Args:
            context: Contexto a validar

        Returns:
            Tupla (es_válido, mensaje_error)
        """
        return validate_context(
            context=context,
            required_keys=self.required_keys,
            key_validators=self.key_validators
        )
