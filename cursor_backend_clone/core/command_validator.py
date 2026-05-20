"""
Command Validator - Validador de comandos
==========================================

Valida y sanitiza comandos antes de ejecutarlos.
"""

import re
import logging
from typing import List, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Resultado de validación"""
    is_valid: bool
    sanitized_command: Optional[str] = None
    warnings: List[str] = None
    errors: List[str] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []
        if self.errors is None:
            self.errors = []


class CommandValidator:
    """Validador de comandos"""
    
    def __init__(
        self,
        max_length: int = 10000,
        allowed_patterns: Optional[List[str]] = None,
        blocked_patterns: Optional[List[str]] = None,
        require_safe_chars: bool = True
    ):
        self.max_length = max_length
        self.allowed_patterns = allowed_patterns or []
        self.blocked_patterns = blocked_patterns or [
            r'rm\s+-rf\s+/',
            r'dd\s+if=.*of=/dev/',
            r':\(\)\{.*\|\s*&\s*\};',
            r'mkfs\.',
            r'fdisk\s+/dev/',
            r'format\s+[c-z]:',
            r'del\s+/f\s+/s\s+/q\s+[c-z]:',
            r'__import__\s*\(',
            r'eval\s*\(',
            r'exec\s*\(',
            r'compile\s*\(',
        ]
        self.require_safe_chars = require_safe_chars
    
    def validate(self, command: str) -> ValidationResult:
        """Validar comando"""
        result = ValidationResult(is_valid=True)
        result.sanitized_command = command.strip()
        
        if not result.sanitized_command:
            result.is_valid = False
            result.errors.append("Command cannot be empty")
            return result
        
        if len(result.sanitized_command) > self.max_length:
            result.is_valid = False
            result.errors.append(f"Command exceeds maximum length of {self.max_length} characters")
            return result
        
        for pattern in self.blocked_patterns:
            if re.search(pattern, result.sanitized_command, re.IGNORECASE):
                result.is_valid = False
                result.errors.append(f"Command contains blocked pattern: {pattern}")
                return result
        
        if self.require_safe_chars:
            if not re.match(r'^[\x20-\x7E\n\r\t]+$', result.sanitized_command):
                result.warnings.append("Command contains non-printable characters")
        
        if len(result.sanitized_command) > 1000:
            result.warnings.append("Command is very long, may take significant time to execute")
        
        return result
    
    def sanitize(self, command: str) -> str:
        """Sanitizar comando"""
        sanitized = command.strip()
        sanitized = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]', '', sanitized)
        return sanitized

