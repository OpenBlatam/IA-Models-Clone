"""
Validators - Sistema de validación
===================================

Validación de comandos y datos de entrada.
"""

import re
import logging
from typing import List, Dict, Optional, Callable, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Resultado de validación"""
    valid: bool
    errors: List[str] = None
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []


class CommandValidator:
    """Validador de comandos"""
    
    def __init__(self):
        self.blocked_patterns: List[str] = [
            r"__import__\s*\(",
            r"eval\s*\(",
            r"exec\s*\(",
            r"open\s*\([^)]*['\"]w[^)]*\)",  # open(..., 'w')
            r"subprocess\.",
            r"os\.system",
            r"os\.popen",
            r"shutil\.rmtree",
            r"rm\s+-rf",
        ]
        
        self.warning_patterns: List[str] = [
            r"import\s+subprocess",
            r"import\s+os",
            r"import\s+shutil",
        ]
        
        self.max_length = 10000
        self.min_length = 1
    
    def validate(self, command: str, strict: bool = False) -> ValidationResult:
        """Validar comando"""
        result = ValidationResult(valid=True)
        
        # Validar longitud
        if len(command) < self.min_length:
            result.valid = False
            result.errors.append(f"Command too short (min: {self.min_length})")
        
        if len(command) > self.max_length:
            result.valid = False
            result.errors.append(f"Command too long (max: {self.max_length})")
        
        # Validar patrones bloqueados
        for pattern in self.blocked_patterns:
            if re.search(pattern, command, re.IGNORECASE):
                result.valid = False
                result.errors.append(f"Blocked pattern detected: {pattern}")
        
        # Validar patrones de advertencia
        for pattern in self.warning_patterns:
            if re.search(pattern, command, re.IGNORECASE):
                result.warnings.append(f"Potentially dangerous pattern: {pattern}")
        
        # Validar caracteres peligrosos
        dangerous_chars = ['\x00', '\r']
        for char in dangerous_chars:
            if char in command:
                result.valid = False
                result.errors.append(f"Dangerous character detected: {repr(char)}")
        
        return result
    
    def sanitize(self, command: str) -> str:
        """Sanitizar comando"""
        # Eliminar caracteres peligrosos
        dangerous_chars = ['\x00', '\r']
        sanitized = command
        for char in dangerous_chars:
            sanitized = sanitized.replace(char, '')
        
        # Limitar longitud
        if len(sanitized) > self.max_length:
            sanitized = sanitized[:self.max_length]
        
        return sanitized.strip()


class InputValidator:
    """Validador de entrada general"""
    
    @staticmethod
    def validate_email(email: str) -> bool:
        """Validar email"""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))
    
    @staticmethod
    def validate_url(url: str) -> bool:
        """Validar URL"""
        pattern = r'^https?://[^\s/$.?#].[^\s]*$'
        return bool(re.match(pattern, url))
    
    @staticmethod
    def validate_json(json_str: str) -> bool:
        """Validar JSON"""
        try:
            import json
            json.loads(json_str)
            return True
        except:
            return False
    
    @staticmethod
    def validate_length(value: str, min_len: int = 0, max_len: int = 1000) -> bool:
        """Validar longitud"""
        return min_len <= len(value) <= max_len


