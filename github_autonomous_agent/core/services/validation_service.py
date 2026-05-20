"""
Servicio de Validación y Sanitización.
"""

from typing import Any, Dict, List, Optional, Callable
from datetime import datetime
import re
from enum import Enum

from config.logging_config import get_logger

logger = get_logger(__name__)


class ValidationError(Exception):
    """Error de validación."""
    def __init__(self, message: str, field: Optional[str] = None, code: Optional[str] = None):
        self.message = message
        self.field = field
        self.code = code
        super().__init__(self.message)


class ValidationRule:
    """Regla de validación."""
    
    def __init__(
        self,
        field: str,
        validator: Callable[[Any], bool],
        message: str,
        code: Optional[str] = None
    ):
        """
        Inicializar regla de validación.
        
        Args:
            field: Campo a validar
            validator: Función de validación
            message: Mensaje de error
            code: Código de error (opcional)
        """
        self.field = field
        self.validator = validator
        self.message = message
        self.code = code
    
    def validate(self, value: Any) -> bool:
        """
        Validar valor.
        
        Args:
            value: Valor a validar
            
        Returns:
            True si es válido
            
        Raises:
            ValidationError: Si la validación falla
        """
        if not self.validator(value):
            raise ValidationError(self.message, field=self.field, code=self.code)
        return True


class ValidationService:
    """Servicio de validación y sanitización."""
    
    def __init__(self):
        """Inicializar servicio de validación."""
        self.rules: Dict[str, List[ValidationRule]] = {}
        self.stats = {
            "total_validations": 0,
            "successful_validations": 0,
            "failed_validations": 0,
            "validations_by_field": {}
        }
    
    def add_rule(
        self,
        field: str,
        validator: Callable[[Any], bool],
        message: str,
        code: Optional[str] = None
    ) -> None:
        """
        Agregar regla de validación.
        
        Args:
            field: Campo a validar
            validator: Función de validación
            message: Mensaje de error
            code: Código de error (opcional)
        """
        if field not in self.rules:
            self.rules[field] = []
        
        rule = ValidationRule(field, validator, message, code)
        self.rules[field].append(rule)
    
    def validate(
        self,
        data: Dict[str, Any],
        fields: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Validar datos.
        
        Args:
            data: Datos a validar
            fields: Campos específicos a validar (opcional, valida todos si es None)
            
        Returns:
            Datos validados y sanitizados
            
        Raises:
            ValidationError: Si la validación falla
        """
        self.stats["total_validations"] += 1
        
        errors: List[ValidationError] = []
        validated_data = {}
        
        fields_to_validate = fields or list(self.rules.keys())
        
        for field in fields_to_validate:
            value = data.get(field)
            
            if field in self.rules:
                for rule in self.rules[field]:
                    try:
                        rule.validate(value)
                    except ValidationError as e:
                        errors.append(e)
                        self.stats["failed_validations"] += 1
                        if field not in self.stats["validations_by_field"]:
                            self.stats["validations_by_field"][field] = {"success": 0, "failed": 0}
                        self.stats["validations_by_field"][field]["failed"] += 1
            
            # Sanitizar y agregar a datos validados
            validated_data[field] = self.sanitize_value(value)
        
        if errors:
            error_messages = [f"{e.field}: {e.message}" for e in errors]
            raise ValidationError(
                f"Validation failed: {', '.join(error_messages)}",
                code="VALIDATION_FAILED"
            )
        
        self.stats["successful_validations"] += 1
        return validated_data
    
    def sanitize_value(self, value: Any) -> Any:
        """
        Sanitizar valor.
        
        Args:
            value: Valor a sanitizar
            
        Returns:
            Valor sanitizado
        """
        if value is None:
            return None
        
        if isinstance(value, str):
            # Remover caracteres peligrosos
            value = re.sub(r'[<>"\']', '', value)
            # Trim whitespace
            value = value.strip()
            # Limitar longitud
            if len(value) > 10000:
                value = value[:10000]
        
        elif isinstance(value, dict):
            return {k: self.sanitize_value(v) for k, v in value.items()}
        
        elif isinstance(value, list):
            return [self.sanitize_value(item) for item in value]
        
        return value
    
    def validate_email(self, email: str) -> bool:
        """
        Validar email.
        
        Args:
            email: Email a validar
            
        Returns:
            True si es válido
        """
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))
    
    def validate_url(self, url: str) -> bool:
        """
        Validar URL.
        
        Args:
            url: URL a validar
            
        Returns:
            True si es válido
        """
        pattern = r'^https?://[^\s/$.?#].[^\s]*$'
        return bool(re.match(pattern, url))
    
    def validate_github_repo(self, repo: str) -> bool:
        """
        Validar formato de repositorio de GitHub.
        
        Args:
            repo: Repositorio a validar (formato: owner/repo)
            
        Returns:
            True si es válido
        """
        pattern = r'^[a-zA-Z0-9._-]+/[a-zA-Z0-9._-]+$'
        return bool(re.match(pattern, repo))
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas."""
        return {
            **self.stats,
            "success_rate": (
                self.stats["successful_validations"] / self.stats["total_validations"]
                if self.stats["total_validations"] > 0
                else 0
            )
        }


# Validadores comunes
def is_required(value: Any) -> bool:
    """Validar que el valor no sea None o vacío."""
    if value is None:
        return False
    if isinstance(value, str) and not value.strip():
        return False
    if isinstance(value, (list, dict)) and len(value) == 0:
        return False
    return True


def is_min_length(min_len: int) -> Callable[[str], bool]:
    """Validar longitud mínima."""
    def validator(value: str) -> bool:
        return isinstance(value, str) and len(value) >= min_len
    return validator


def is_max_length(max_len: int) -> Callable[[str], bool]:
    """Validar longitud máxima."""
    def validator(value: str) -> bool:
        return isinstance(value, str) and len(value) <= max_len
    return validator


def is_in_range(min_val: float, max_val: float) -> Callable[[float], bool]:
    """Validar rango numérico."""
    def validator(value: float) -> bool:
        return isinstance(value, (int, float)) and min_val <= value <= max_val
    return validator



