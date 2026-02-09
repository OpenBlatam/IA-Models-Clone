"""
Validators - Sistema de validación
===================================

Validación de comandos y datos de entrada con soporte para
validación extensible y sanitización segura.
"""

import re
import logging
from typing import List, Dict, Optional, Callable, Any, Pattern
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """
    Resultado de validación.
    
    Attributes:
        valid: True si la validación pasó.
        errors: Lista de errores encontrados.
        warnings: Lista de advertencias encontradas.
    """
    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def add_error(self, error: str) -> None:
        """
        Agregar un error y marcar como inválido.
        
        Args:
            error: Mensaje de error.
        """
        if error and error not in self.errors:
            self.errors.append(error)
            self.valid = False
    
    def add_warning(self, warning: str) -> None:
        """
        Agregar una advertencia.
        
        Args:
            warning: Mensaje de advertencia.
        """
        if warning and warning not in self.warnings:
            self.warnings.append(warning)


class CommandValidator:
    """
    Validador de comandos con detección de patrones peligrosos.
    
    Valida comandos contra patrones bloqueados y genera advertencias
    para patrones potencialmente peligrosos.
    """
    
    # Patrones bloqueados (comandos peligrosos)
    BLOCKED_PATTERNS: List[Pattern[str]] = [
        re.compile(r"__import__\s*\(", re.IGNORECASE),
        re.compile(r"eval\s*\(", re.IGNORECASE),
        re.compile(r"exec\s*\(", re.IGNORECASE),
        re.compile(r"open\s*\([^)]*['\"]w[^)]*\)", re.IGNORECASE),
        re.compile(r"subprocess\.", re.IGNORECASE),
        re.compile(r"os\.system", re.IGNORECASE),
        re.compile(r"os\.popen", re.IGNORECASE),
        re.compile(r"shutil\.rmtree", re.IGNORECASE),
        re.compile(r"rm\s+-rf", re.IGNORECASE),
    ]
    
    # Patrones de advertencia (potencialmente peligrosos)
    WARNING_PATTERNS: List[Pattern[str]] = [
        re.compile(r"import\s+subprocess", re.IGNORECASE),
        re.compile(r"import\s+os", re.IGNORECASE),
        re.compile(r"import\s+shutil", re.IGNORECASE),
    ]
    
    # Caracteres peligrosos
    DANGEROUS_CHARS: List[str] = ['\x00', '\r']
    
    def __init__(
        self,
        min_length: int = 1,
        max_length: int = 10000,
        blocked_patterns: Optional[List[str]] = None,
        warning_patterns: Optional[List[str]] = None
    ) -> None:
        """
        Inicializar validador de comandos.
        
        Args:
            min_length: Longitud mínima del comando (default: 1).
            max_length: Longitud máxima del comando (default: 10000).
            blocked_patterns: Patrones adicionales a bloquear (opcional).
            warning_patterns: Patrones adicionales para advertencias (opcional).
        
        Raises:
            ValueError: Si los parámetros son inválidos.
        """
        if min_length < 0:
            raise ValueError("min_length must be non-negative")
        if max_length <= 0 or max_length < min_length:
            raise ValueError("max_length must be positive and >= min_length")
        
        self.min_length: int = min_length
        self.max_length: int = max_length
        
        # Compilar patrones personalizados si se proporcionan
        self.blocked_patterns: List[Pattern[str]] = list(self.BLOCKED_PATTERNS)
        if blocked_patterns:
            self.blocked_patterns.extend([
                re.compile(pattern, re.IGNORECASE)
                for pattern in blocked_patterns
            ])
        
        self.warning_patterns: List[Pattern[str]] = list(self.WARNING_PATTERNS)
        if warning_patterns:
            self.warning_patterns.extend([
                re.compile(pattern, re.IGNORECASE)
                for pattern in warning_patterns
            ])
    
    def validate(self, command: str, strict: bool = False) -> ValidationResult:
        """
        Validar comando.
        
        Args:
            command: Comando a validar.
            strict: Si es True, las advertencias también invalidan (default: False).
        
        Returns:
            ValidationResult con el resultado de la validación.
        
        Raises:
            TypeError: Si el comando no es un string.
        """
        if not isinstance(command, str):
            raise TypeError(f"Command must be a string, got {type(command)}")
        
        result = ValidationResult(valid=True)
        
        # Validar longitud
        self._validate_length(command, result)
        
        # Validar patrones bloqueados
        self._validate_blocked_patterns(command, result)
        
        # Validar patrones de advertencia
        self._validate_warning_patterns(command, result, strict)
        
        # Validar caracteres peligrosos
        self._validate_dangerous_chars(command, result)
        
        return result
    
    def _validate_length(self, command: str, result: ValidationResult) -> None:
        """Validar longitud del comando"""
        if len(command) < self.min_length:
            result.add_error(
                f"Command too short (min: {self.min_length}, got: {len(command)})"
            )
        
        if len(command) > self.max_length:
            result.add_error(
                f"Command too long (max: {self.max_length}, got: {len(command)})"
            )
    
    def _validate_blocked_patterns(
        self,
        command: str,
        result: ValidationResult
    ) -> None:
        """Validar patrones bloqueados"""
        for pattern in self.blocked_patterns:
            if pattern.search(command):
                result.add_error(
                    f"Blocked pattern detected: {pattern.pattern}"
                )
    
    def _validate_warning_patterns(
        self,
        command: str,
        result: ValidationResult,
        strict: bool
    ) -> None:
        """Validar patrones de advertencia"""
        for pattern in self.warning_patterns:
            if pattern.search(command):
                warning_msg = f"Potentially dangerous pattern: {pattern.pattern}"
                if strict:
                    result.add_error(warning_msg)
                else:
                    result.add_warning(warning_msg)
    
    def _validate_dangerous_chars(
        self,
        command: str,
        result: ValidationResult
    ) -> None:
        """Validar caracteres peligrosos"""
        for char in self.DANGEROUS_CHARS:
            if char in command:
                result.add_error(
                    f"Dangerous character detected: {repr(char)}"
                )
    
    def sanitize(self, command: str) -> str:
        """
        Sanitizar comando removiendo caracteres peligrosos.
        
        Args:
            command: Comando a sanitizar.
        
        Returns:
            Comando sanitizado.
        
        Raises:
            TypeError: Si el comando no es un string.
        """
        if not isinstance(command, str):
            raise TypeError(f"Command must be a string, got {type(command)}")
        
        sanitized = command
        
        # Eliminar caracteres peligrosos
        for char in self.DANGEROUS_CHARS:
            sanitized = sanitized.replace(char, '')
        
        # Limitar longitud
        if len(sanitized) > self.max_length:
            sanitized = sanitized[:self.max_length]
        
        return sanitized.strip()


class InputValidator:
    """
    Validador de entrada general para diferentes tipos de datos.
    
    Proporciona métodos estáticos para validar emails, URLs, JSON, etc.
    """
    
    # Patrones de validación compilados
    EMAIL_PATTERN: Pattern[str] = re.compile(
        r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    )
    
    URL_PATTERN: Pattern[str] = re.compile(
        r'^https?://[^\s/$.?#].[^\s]*$'
    )
    
    @staticmethod
    def validate_email(email: str) -> bool:
        """
        Validar formato de email.
        
        Args:
            email: Email a validar.
        
        Returns:
            True si el email es válido, False en caso contrario.
        """
        if not isinstance(email, str) or not email:
            return False
        return bool(InputValidator.EMAIL_PATTERN.match(email))
    
    @staticmethod
    def validate_url(url: str) -> bool:
        """
        Validar formato de URL.
        
        Args:
            url: URL a validar.
        
        Returns:
            True si la URL es válida, False en caso contrario.
        """
        if not isinstance(url, str) or not url:
            return False
        return bool(InputValidator.URL_PATTERN.match(url))
    
    @staticmethod
    def validate_json(json_str: str) -> bool:
        """
        Validar formato JSON.
        
        Args:
            json_str: String JSON a validar.
        
        Returns:
            True si el JSON es válido, False en caso contrario.
        """
        if not isinstance(json_str, str) or not json_str.strip():
            return False
        
        try:
            import json
            json.loads(json_str)
            return True
        except (ValueError, TypeError):
            return False
    
    @staticmethod
    def validate_length(
        value: str,
        min_len: int = 0,
        max_len: int = 1000
    ) -> bool:
        """
        Validar longitud de un string.
        
        Args:
            value: String a validar.
            min_len: Longitud mínima (default: 0).
            max_len: Longitud máxima (default: 1000).
        
        Returns:
            True si la longitud es válida, False en caso contrario.
        
        Raises:
            ValueError: Si min_len o max_len son inválidos.
        """
        if min_len < 0:
            raise ValueError("min_len must be non-negative")
        if max_len <= 0 or max_len < min_len:
            raise ValueError("max_len must be positive and >= min_len")
        
        if not isinstance(value, str):
            return False
        
        return min_len <= len(value) <= max_len
