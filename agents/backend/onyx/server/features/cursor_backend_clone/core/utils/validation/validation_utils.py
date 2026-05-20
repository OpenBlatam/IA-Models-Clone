"""
Validation Utils - Utilidades avanzadas de validación
======================================================

Utilidades adicionales para validación y sanitización de datos.
"""

import re
import logging
from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


@dataclass
class ValidationRule:
    """Regla de validación"""
    name: str
    pattern: Optional[str] = None
    validator: Optional[Callable[[str], bool]] = None
    error_message: str = "Validation failed"
    warning_message: Optional[str] = None


class AdvancedValidator:
    """
    Validador avanzado con reglas configurables.
    
    Permite definir reglas personalizadas de validación
    y aplicar múltiples validaciones en secuencia.
    """
    
    def __init__(self):
        self.rules: List[ValidationRule] = []
        self.custom_validators: Dict[str, Callable[[str], Tuple[bool, Optional[str]]]] = {}
    
    def add_rule(
        self,
        name: str,
        pattern: Optional[str] = None,
        validator: Optional[Callable[[str], bool]] = None,
        error_message: str = "Validation failed",
        warning_message: Optional[str] = None
    ) -> None:
        """
        Agregar regla de validación.
        
        Args:
            name: Nombre de la regla
            pattern: Patrón regex (opcional)
            validator: Función validadora (opcional)
            error_message: Mensaje de error
            warning_message: Mensaje de advertencia (opcional)
        """
        if pattern and validator:
            raise ValueError("Cannot specify both pattern and validator")
        
        rule = ValidationRule(
            name=name,
            pattern=pattern,
            validator=validator,
            error_message=error_message,
            warning_message=warning_message
        )
        self.rules.append(rule)
    
    def validate(self, value: str) -> Tuple[bool, List[str], List[str]]:
        """
        Validar valor contra todas las reglas.
        
        Args:
            value: Valor a validar
            
        Returns:
            Tupla de (is_valid, errors, warnings)
        """
        errors = []
        warnings = []
        
        for rule in self.rules:
            if rule.pattern:
                if not re.search(rule.pattern, value, re.IGNORECASE):
                    errors.append(f"{rule.name}: {rule.error_message}")
            elif rule.validator:
                try:
                    if not rule.validator(value):
                        errors.append(f"{rule.name}: {rule.error_message}")
                except Exception as e:
                    errors.append(f"{rule.name}: Validation error: {str(e)}")
            
            if rule.warning_message:
                warnings.append(f"{rule.name}: {rule.warning_message}")
        
        return len(errors) == 0, errors, warnings
    
    def register_validator(
        self,
        name: str,
        validator: Callable[[str], Tuple[bool, Optional[str]]]
    ) -> None:
        """
        Registrar validador personalizado.
        
        Args:
            name: Nombre del validador
            validator: Función que retorna (is_valid, error_message)
        """
        self.custom_validators[name] = validator
    
    def validate_with_custom(self, value: str, validator_name: str) -> Tuple[bool, Optional[str]]:
        """
        Validar usando validador personalizado.
        
        Args:
            value: Valor a validar
            validator_name: Nombre del validador
            
        Returns:
            Tupla de (is_valid, error_message)
        """
        if validator_name not in self.custom_validators:
            raise ValueError(f"Validator '{validator_name}' not found")
        
        return self.custom_validators[validator_name](value)


def validate_url(url: str, allowed_schemes: Optional[List[str]] = None) -> Tuple[bool, Optional[str]]:
    """
    Validar URL con esquemas permitidos.
    
    Args:
        url: URL a validar
        allowed_schemes: Lista de esquemas permitidos (default: http, https)
        
    Returns:
        Tupla de (is_valid, error_message)
    """
    if allowed_schemes is None:
        allowed_schemes = ["http", "https"]
    
    try:
        parsed = urlparse(url)
        
        if not parsed.scheme:
            return False, "URL must include scheme (http:// or https://)"
        
        if parsed.scheme not in allowed_schemes:
            return False, f"Scheme '{parsed.scheme}' not allowed. Allowed: {', '.join(allowed_schemes)}"
        
        if not parsed.netloc:
            return False, "URL must include hostname"
        
        # Validar longitud
        if len(url) > 2048:
            return False, "URL exceeds maximum length of 2048 characters"
        
        return True, None
        
    except Exception as e:
        return False, f"Invalid URL format: {str(e)}"


def validate_email(email: str) -> Tuple[bool, Optional[str]]:
    """
    Validar formato de email.
    
    Args:
        email: Email a validar
        
    Returns:
        Tupla de (is_valid, error_message)
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    if not re.match(pattern, email):
        return False, "Invalid email format"
    
    if len(email) > 254:  # RFC 5321
        return False, "Email exceeds maximum length of 254 characters"
    
    return True, None


def validate_json_string(json_str: str) -> Tuple[bool, Optional[str]]:
    """
    Validar que un string es JSON válido.
    
    Args:
        json_str: String a validar
        
    Returns:
        Tupla de (is_valid, error_message)
    """
    try:
        import json
        json.loads(json_str)
        return True, None
    except json.JSONDecodeError as e:
        return False, f"Invalid JSON: {str(e)}"
    except Exception as e:
        return False, f"JSON validation error: {str(e)}"


def sanitize_filename(filename: str, max_length: int = 255) -> str:
    """
    Sanitizar nombre de archivo.
    
    Args:
        filename: Nombre de archivo a sanitizar
        max_length: Longitud máxima
        
    Returns:
        Nombre de archivo sanitizado
    """
    # Remover caracteres peligrosos
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # Remover espacios al inicio y final
    sanitized = sanitized.strip()
    
    # Remover puntos al final (Windows)
    sanitized = sanitized.rstrip('.')
    
    # Limitar longitud
    if len(sanitized) > max_length:
        name, ext = sanitized.rsplit('.', 1) if '.' in sanitized else (sanitized, '')
        max_name_length = max_length - len(ext) - 1 if ext else max_length
        sanitized = name[:max_name_length] + (f'.{ext}' if ext else '')
    
    return sanitized or "unnamed"


def validate_and_sanitize_command(
    command: str,
    max_length: int = 10000,
    allow_empty: bool = False
) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Validar y sanitizar comando.
    
    Args:
        command: Comando a validar
        max_length: Longitud máxima
        allow_empty: Si permitir comandos vacíos
        
    Returns:
        Tupla de (is_valid, sanitized_command, error_message)
    """
    if not command:
        if allow_empty:
            return True, "", None
        return False, None, "Command cannot be empty"
    
    # Sanitizar
    sanitized = command.strip()
    
    # Remover caracteres de control excepto newline, tab, carriage return
    sanitized = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]', '', sanitized)
    
    if not sanitized and not allow_empty:
        return False, None, "Command cannot be empty after sanitization"
    
    if len(sanitized) > max_length:
        return False, None, f"Command exceeds maximum length of {max_length} characters"
    
    return True, sanitized, None




