"""
Security Validator System
=========================

Advanced security validation system for input sanitization and security checks.
"""

import logging
import re
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class SecurityLevel(Enum):
    """Security levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SecurityIssue:
    """Security issue."""
    level: SecurityLevel
    type: str
    message: str
    field: Optional[str] = None
    recommendation: Optional[str] = None


@dataclass
class ValidationResult:
    """Security validation result."""
    valid: bool
    issues: List[SecurityIssue] = field(default_factory=list)
    sanitized_data: Dict[str, Any] = field(default_factory=dict)


class SecurityValidator:
    """Advanced security validator."""
    
    def __init__(self, security_level: SecurityLevel = SecurityLevel.MEDIUM):
        """
        Initialize security validator.
        
        Args:
            security_level: Security level
        """
        self.security_level = security_level
        self.validators: Dict[str, List[Callable]] = {}
        self._register_default_validators()
    
    def _register_default_validators(self):
        """Register default validators."""
        self.validators['path_traversal'] = [self._validate_path_traversal]
        self.validators['sql_injection'] = [self._validate_sql_injection]
        self.validators['xss'] = [self._validate_xss]
        self.validators['command_injection'] = [self._validate_command_injection]
        self.validators['file_extension'] = [self._validate_file_extension]
    
    def validate(self, data: Dict[str, Any]) -> ValidationResult:
        """
        Validate data for security issues.
        
        Args:
            data: Data to validate
            
        Returns:
            Validation result
        """
        issues = []
        sanitized_data = {}
        
        for key, value in data.items():
            if isinstance(value, str):
                # Run all validators
                for validator_type, validators in self.validators.items():
                    for validator in validators:
                        issue = validator(key, value)
                        if issue:
                            issues.append(issue)
                
                # Sanitize value
                sanitized_value = self._sanitize_string(value)
                sanitized_data[key] = sanitized_value
            elif isinstance(value, dict):
                # Recursive validation
                nested_result = self.validate(value)
                issues.extend(nested_result.issues)
                sanitized_data[key] = nested_result.sanitized_data
            else:
                sanitized_data[key] = value
        
        valid = len(issues) == 0 or all(
            issue.level in [SecurityLevel.LOW, SecurityLevel.MEDIUM]
            for issue in issues
        )
        
        return ValidationResult(
            valid=valid,
            issues=issues,
            sanitized_data=sanitized_data
        )
    
    def _validate_path_traversal(self, field: str, value: str) -> Optional[SecurityIssue]:
        """Validate for path traversal attacks."""
        dangerous_patterns = ['../', '..\\', '/etc/', 'C:\\', '..']
        
        for pattern in dangerous_patterns:
            if pattern in value:
                return SecurityIssue(
                    level=SecurityLevel.HIGH,
                    type="path_traversal",
                    message=f"Potential path traversal detected in field '{field}'",
                    field=field,
                    recommendation="Remove path traversal sequences"
                )
        return None
    
    def _validate_sql_injection(self, field: str, value: str) -> Optional[SecurityIssue]:
        """Validate for SQL injection."""
        sql_patterns = [
            r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|EXECUTE)\b)",
            r"(--|;|/\*|\*/)",
            r"(\b(OR|AND)\s+\d+\s*=\s*\d+)"
        ]
        
        for pattern in sql_patterns:
            if re.search(pattern, value, re.IGNORECASE):
                return SecurityIssue(
                    level=SecurityLevel.HIGH,
                    type="sql_injection",
                    message=f"Potential SQL injection detected in field '{field}'",
                    field=field,
                    recommendation="Use parameterized queries"
                )
        return None
    
    def _validate_xss(self, field: str, value: str) -> Optional[SecurityIssue]:
        """Validate for XSS attacks."""
        xss_patterns = [
            r"<script[^>]*>",
            r"javascript:",
            r"on\w+\s*=",
            r"<iframe[^>]*>"
        ]
        
        for pattern in xss_patterns:
            if re.search(pattern, value, re.IGNORECASE):
                return SecurityIssue(
                    level=SecurityLevel.MEDIUM,
                    type="xss",
                    message=f"Potential XSS detected in field '{field}'",
                    field=field,
                    recommendation="Sanitize HTML content"
                )
        return None
    
    def _validate_command_injection(self, field: str, value: str) -> Optional[SecurityIssue]:
        """Validate for command injection."""
        command_patterns = [
            r"[;&|`$()]",
            r"\b(cat|ls|rm|mv|cp|chmod|chown)\b"
        ]
        
        for pattern in command_patterns:
            if re.search(pattern, value):
                return SecurityIssue(
                    level=SecurityLevel.HIGH,
                    type="command_injection",
                    message=f"Potential command injection detected in field '{field}'",
                    field=field,
                    recommendation="Avoid shell command execution"
                )
        return None
    
    def _validate_file_extension(self, field: str, value: str) -> Optional[SecurityIssue]:
        """Validate file extension."""
        dangerous_extensions = ['.exe', '.sh', '.bat', '.cmd', '.ps1', '.py', '.php', '.jsp']
        
        if any(value.lower().endswith(ext) for ext in dangerous_extensions):
            return SecurityIssue(
                level=SecurityLevel.MEDIUM,
                type="file_extension",
                message=f"Dangerous file extension in field '{field}'",
                field=field,
                recommendation="Restrict file types"
            )
        return None
    
    def _sanitize_string(self, value: str) -> str:
        """Sanitize string value."""
        # Remove null bytes
        value = value.replace('\x00', '')
        
        # Remove control characters (except newline and tab)
        value = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]', '', value)
        
        # Remove path traversal sequences
        value = value.replace('../', '').replace('..\\', '')
        
        return value.strip()
    
    def validate_file_path(self, file_path: str, allowed_dirs: Optional[List[str]] = None) -> ValidationResult:
        """
        Validate file path.
        
        Args:
            file_path: File path to validate
            allowed_dirs: Optional list of allowed directories
            
        Returns:
            Validation result
        """
        issues = []
        
        # Check for path traversal
        if '../' in file_path or '..\\' in file_path:
            issues.append(SecurityIssue(
                level=SecurityLevel.HIGH,
                type="path_traversal",
                message="Path traversal detected in file path",
                recommendation="Use absolute paths or validate relative paths"
            ))
        
        # Check if path is in allowed directories
        if allowed_dirs:
            path_obj = Path(file_path).resolve()
            is_allowed = any(
                str(path_obj).startswith(str(Path(allowed_dir).resolve()))
                for allowed_dir in allowed_dirs
            )
            
            if not is_allowed:
                issues.append(SecurityIssue(
                    level=SecurityLevel.MEDIUM,
                    type="unauthorized_path",
                    message="File path not in allowed directories",
                    recommendation="Restrict file access to allowed directories"
                ))
        
        valid = len(issues) == 0
        
        return ValidationResult(
            valid=valid,
            issues=issues,
            sanitized_data={"file_path": self._sanitize_string(file_path)}
        )



