from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
BUFFER_SIZE = 1024

import asyncio
import html
import logging
import os
import re
import shlex
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable, Set, Tuple
from enum import Enum
import urllib.parse
import ipaddress
import unicodedata
import base64
import hashlib
import threading
from contextlib import contextmanager
    import bleach
    import validators
            import resource
from typing import Any, List, Dict, Optional
"""
Input Sanitization and Secure Command Execution Examples
=======================================================

This module provides comprehensive input sanitization and secure command execution
capabilities to prevent command injection and other input-based attacks.

Features:
- Multi-layer input sanitization and validation
- Secure command execution with parameterized commands
- Shell command sanitization and escaping
- SQL injection prevention patterns
- XSS prevention and HTML sanitization
- File path validation and sanitization
- URL and email validation
- Input whitelisting and blacklisting
- Secure string handling and encoding
- Command execution with timeout and resource limits

Author: AI Assistant
License: MIT
"""


try:
    BLEACH_AVAILABLE = True
except ImportError:
    BLEACH_AVAILABLE = False

try:
    VALIDATORS_AVAILABLE = True
except ImportError:
    VALIDATORS_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SanitizationLevel(Enum):
    """Sanitization levels for different security requirements."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class InputType(Enum):
    """Types of input that require different sanitization approaches."""
    COMMAND = "command"
    SQL = "sql"
    HTML = "html"
    URL = "url"
    EMAIL = "email"
    FILE_PATH = "file_path"
    IP_ADDRESS = "ip_address"
    JSON = "json"
    XML = "xml"
    SHELL = "shell"
    USERNAME = "username"
    PASSWORD = "password"
    GENERAL = "general"


@dataclass
class SanitizationResult:
    """Result of input sanitization."""
    sanitized: str
    original: str
    input_type: InputType
    sanitization_level: SanitizationLevel
    threats_detected: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    sanitization_applied: List[str] = field(default_factory=list)
    risk_score: float = 0.0
    is_safe: bool = True


@dataclass
class CommandExecutionResult:
    """Result of secure command execution."""
    success: bool
    command: str
    arguments: List[str]
    stdout: str = ""
    stderr: str = ""
    exit_code: int = -1
    execution_time: float = 0.0
    error_message: str = ""
    security_events: List[str] = field(default_factory=list)
    resource_usage: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InputValidationResult:
    """Result of input validation."""
    valid: bool
    input_type: InputType
    validation_rules: List[str] = field(default_factory=list)
    violations: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    risk_level: str = "low"


class InputSanitizationError(Exception):
    """Custom exception for input sanitization errors."""
    pass


class CommandExecutionError(Exception):
    """Custom exception for command execution errors."""
    pass


class InputValidationError(Exception):
    """Custom exception for input validation errors."""
    pass


class SecurityViolationError(Exception):
    """Custom exception for security violations."""
    pass


class InputSanitizer:
    """Comprehensive input sanitization and validation."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize input sanitizer with configuration."""
        self.config = config or {}
        
        # Security patterns
        self.dangerous_patterns = {
            'command_injection': [
                r'(\||&|;|`|\\$\\()',
                r'(\b(cmd|command|exec|system|eval|subprocess)\b)',
                r'(/bin/bash|/bin/sh|cmd\.exe)',
                r'(\$\{.*\}|\$\(.*\))',
                r'(\b(rm|del|format|fdisk|mkfs)\b)'
            ],
            'sql_injection': [
                r"(\b(union|select|insert|update|delete|drop|create|alter)\b)",
                r"(--|#|/\*|\*/)",
                r"(\b(exec|execute|xp_|sp_)\b)",
                r"('|"|;|--|#|/\*|\*/)",
                r"(\b(script|javascript|vbscript|onload|onerror)\b)"
            ],
            'xss': [
                r"(<script[^>]*>.*?</script>)",
                r"(javascript:)",
                r"(on\w+\s*=)",
                r"(<iframe[^>]*>)",
                r"(<object[^>]*>)",
                r"(<embed[^>]*>)",
                r"(<form[^>]*>)",
                r"(<input[^>]*>)",
                r"(<textarea[^>]*>)",
                r"(<select[^>]*>)"
            ],
            'path_traversal': [
                r"(\.\./|\.\.\\)",
                r"(/etc/passwd|/etc/shadow)",
                r"(c:\\windows\\system32)",
                r"(%2e%2e%2f|%2e%2e%5c)",
                r"(~|%7e)",
                r"(/proc/|/sys/)"
            ],
            'shell_injection': [
                r"(\$[a-zA-Z_][a-zA-Z0-9_]*)",
                r"(\$\{[^}]*\})",
                r"(\$\([^)]*\))",
                r"(\`[^`]*\`)",
                r"(\b(export|set|unset|env)\b)"
            ]
        }
        
        # Whitelist patterns
        self.whitelist_patterns = {
            'username': r'^[a-zA-Z0-9_-]{3,20}$',
            'email': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
            'ip_address': r'^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$',
            'url': r'^https?://[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(?:/[a-zA-Z0-9._/-]*)?$',
            'file_path': r'^[a-zA-Z0-9/._-]+$',
            'command': r'^[a-zA-Z0-9/._-]+$'
        }
        
        # Blacklist patterns
        self.blacklist_patterns = {
            'dangerous_commands': [
                'rm', 'del', 'format', 'fdisk', 'mkfs', 'dd', 'shred',
                'chmod', 'chown', 'sudo', 'su', 'passwd', 'useradd',
                'userdel', 'groupadd', 'groupdel', 'mount', 'umount'
            ],
            'dangerous_paths': [
                '/etc/passwd', '/etc/shadow', '/etc/sudoers',
                '/proc/', '/sys/', '/dev/', '/boot/', '/root/',
                'c:\\windows\\system32', 'c:\\windows\\system',
                'c:\\windows\\syswow64'
            ]
        }
    
    def _detect_threats(self, input_data: str, input_type: InputType) -> List[str]:
        """Detect security threats in input data."""
        threats = []
        
        # Check for dangerous patterns
        for threat_type, patterns in self.dangerous_patterns.items():
            for pattern in patterns:
                if re.search(pattern, input_data, re.IGNORECASE):
                    threats.append(threat_type)
                    break
        
        # Check for blacklisted content
        if input_type == InputType.COMMAND:
            for dangerous_cmd in self.blacklist_patterns['dangerous_commands']:
                if dangerous_cmd in input_data.lower():
                    threats.append('dangerous_command')
                    break
        
        if input_type == InputType.FILE_PATH:
            for dangerous_path in self.blacklist_patterns['dangerous_paths']:
                if dangerous_path in input_data.lower():
                    threats.append('dangerous_path')
                    break
        
        return threats
    
    def _apply_whitelist_validation(self, input_data: str, input_type: InputType) -> bool:
        """Apply whitelist validation based on input type."""
        if input_type not in self.whitelist_patterns:
            return True
        
        pattern = self.whitelist_patterns[input_type]
        return bool(re.match(pattern, input_data))
    
    def sanitize_input(self, input_data: str, input_type: InputType, 
                      sanitization_level: SanitizationLevel = SanitizationLevel.HIGH) -> SanitizationResult:
        """Sanitize input data based on type and security level."""
        if input_data is None:
            return SanitizationResult(
                sanitized="",
                original="",
                input_type=input_type,
                sanitization_level=sanitization_level,
                is_safe=False,
                threats_detected=["null_input"]
            )
        
        original_input = str(input_data)
        sanitized_input = original_input
        threats_detected = self._detect_threats(original_input, input_type)
        warnings = []
        sanitization_applied = []
        risk_score = 0.0
        
        # Apply type-specific sanitization
        if input_type == InputType.HTML:
            sanitized_input = self._sanitize_html(original_input, sanitization_level)
            sanitization_applied.append("html_sanitization")
        
        elif input_type == InputType.SQL:
            sanitized_input = self._sanitize_sql(original_input, sanitization_level)
            sanitization_applied.append("sql_sanitization")
        
        elif input_type == InputType.COMMAND:
            sanitized_input = self._sanitize_command(original_input, sanitization_level)
            sanitization_applied.append("command_sanitization")
        
        elif input_type == InputType.FILE_PATH:
            sanitized_input = self._sanitize_file_path(original_input, sanitization_level)
            sanitization_applied.append("file_path_sanitization")
        
        elif input_type == InputType.URL:
            sanitized_input = self._sanitize_url(original_input, sanitization_level)
            sanitization_applied.append("url_sanitization")
        
        elif input_type == InputType.EMAIL:
            sanitized_input = self._sanitize_email(original_input, sanitization_level)
            sanitization_applied.append("email_sanitization")
        
        # Apply general sanitization
        sanitized_input = self._apply_general_sanitization(sanitized_input, sanitization_level)
        sanitization_applied.append("general_sanitization")
        
        # Check whitelist validation
        if not self._apply_whitelist_validation(sanitized_input, input_type):
            threats_detected.append("whitelist_violation")
            warnings.append(f"Input does not match {input_type.value} pattern")
        
        # Calculate risk score
        risk_score = len(threats_detected) * 0.2
        if risk_score > 1.0:
            risk_score = 1.0
        
        # Determine if input is safe
        is_safe = len(threats_detected) == 0 and risk_score < 0.5
        
        return SanitizationResult(
            sanitized=sanitized_input,
            original=original_input,
            input_type=input_type,
            sanitization_level=sanitization_level,
            threats_detected=threats_detected,
            warnings=warnings,
            sanitization_applied=sanitization_applied,
            risk_score=risk_score,
            is_safe=is_safe
        )
    
    def _sanitize_html(self, input_data: str, level: SanitizationLevel) -> str:
        """Sanitize HTML input."""
        if BLEACH_AVAILABLE:
            # Use bleach for HTML sanitization
            allowed_tags = ['p', 'br', 'strong', 'em', 'u', 'ol', 'ul', 'li']
            allowed_attributes = {}
            
            if level == SanitizationLevel.LOW:
                allowed_tags.extend(['a', 'img'])
                allowed_attributes = {'a': ['href'], 'img': ['src', 'alt']}
            
            return bleach.clean(input_data, tags=allowed_tags, attributes=allowed_attributes, strip=True)
        else:
            # Fallback to basic HTML escaping
            return html.escape(input_data, quote=True)
    
    def _sanitize_sql(self, input_data: str, level: SanitizationLevel) -> str:
        """Sanitize SQL input (use parameterized queries instead)."""
        # Remove SQL comments
        sanitized = re.sub(r'--.*$', '', input_data, flags=re.MULTILINE)
        sanitized = re.sub(r'/\*.*?\*/', '', sanitized, flags=re.DOTALL)
        
        # Escape single quotes
        sanitized = sanitized.replace("'", "''")
        
        # Remove dangerous SQL keywords for high security
        if level in [SanitizationLevel.HIGH, SanitizationLevel.CRITICAL]:
            dangerous_keywords = ['union', 'select', 'insert', 'update', 'delete', 'drop', 'create', 'alter']
            for keyword in dangerous_keywords:
                sanitized = re.sub(rf'\b{keyword}\b', '', sanitized, flags=re.IGNORECASE)
        
        return sanitized.strip()
    
    def _sanitize_command(self, input_data: str, level: SanitizationLevel) -> str:
        """Sanitize command input."""
        # Remove shell metacharacters
        sanitized = re.sub(r'[;&|`$(){}[\]<>]', '', input_data)
        
        # Remove environment variable references
        sanitized = re.sub(r'\$\{[^}]*\}', '', sanitized)
        sanitized = re.sub(r'\$[a-zA-Z_][a-zA-Z0-9_]*', '', sanitized)
        
        # Remove command substitution
        sanitized = re.sub(r'\$\([^)]*\)', '', sanitized)
        sanitized = re.sub(r'`[^`]*`', '', sanitized)
        
        # For critical level, only allow alphanumeric and basic punctuation
        if level == SanitizationLevel.CRITICAL:
            sanitized = re.sub(r'[^a-zA-Z0-9/._-]', '', sanitized)
        
        return sanitized.strip()
    
    def _sanitize_file_path(self, input_data: str, level: SanitizationLevel) -> str:
        """Sanitize file path input."""
        # Remove path traversal sequences
        sanitized = re.sub(r'\.\./', '', input_data)
        sanitized = re.sub(r'\.\.\\', '', sanitized)
        sanitized = re.sub(r'%2e%2e%2f', '', sanitized)
        sanitized = re.sub(r'%2e%2e%5c', '', sanitized)
        
        # Remove dangerous paths
        for dangerous_path in self.blacklist_patterns['dangerous_paths']:
            sanitized = sanitized.replace(dangerous_path, '')
        
        # Normalize path separators
        sanitized = sanitized.replace('\\', '/')
        
        # Remove multiple slashes
        sanitized = re.sub(r'/+', '/', sanitized)
        
        # For critical level, only allow basic path characters
        if level == SanitizationLevel.CRITICAL:
            sanitized = re.sub(r'[^a-zA-Z0-9/._-]', '', sanitized)
        
        return sanitized.strip('/')
    
    def _sanitize_url(self, input_data: str, level: SanitizationLevel) -> str:
        """Sanitize URL input."""
        try:
            parsed = urllib.parse.urlparse(input_data)
            
            # Only allow http and https protocols
            if parsed.scheme not in ['http', 'https']:
                return ""
            
            # Remove dangerous protocols from path
            sanitized_path = re.sub(r'(javascript|data|vbscript):', '', parsed.path)
            
            # Reconstruct URL
            sanitized = urllib.parse.urlunparse((
                parsed.scheme,
                parsed.netloc,
                sanitized_path,
                parsed.params,
                parsed.query,
                parsed.fragment
            ))
            
            return sanitized
        except Exception:
            return ""
    
    def _sanitize_email(self, input_data: str, level: SanitizationLevel) -> str:
        """Sanitize email input."""
        # Remove any HTML or script tags
        sanitized = re.sub(r'<[^>]*>', '', input_data)
        
        # Remove dangerous characters
        sanitized = re.sub(r'[<>"\']', '', sanitized)
        
        # Validate email format
        if not re.match(self.whitelist_patterns['email'], sanitized):
            return ""
        
        return sanitized.lower().strip()
    
    def _apply_general_sanitization(self, input_data: str, level: SanitizationLevel) -> str:
        """Apply general sanitization rules."""
        # Remove null bytes
        sanitized = input_data.replace('\x00', '')
        
        # Remove control characters (except newline and tab)
        sanitized = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', sanitized)
        
        # Normalize unicode
        sanitized = unicodedata.normalize('NFKC', sanitized)
        
        # Remove excessive whitespace
        sanitized = re.sub(r'\s+', ' ', sanitized).strip()
        
        # For high security levels, remove more characters
        if level in [SanitizationLevel.HIGH, SanitizationLevel.CRITICAL]:
            sanitized = re.sub(r'[^\w\s\-_.,!?@#$%&*()+=]', '', sanitized)
        
        return sanitized


class SecureCommandExecutor:
    """Secure command execution with input validation and sanitization."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize secure command executor."""
        self.config = config or {}
        self.sanitizer = InputSanitizer(config)
        
        # Security settings
        self.max_execution_time = self.config.get('max_execution_time', 30)
        self.max_output_size = self.config.get('max_output_size', 1024 * 1024)  # 1MB
        self.allowed_commands = self.config.get('allowed_commands', set())
        self.blocked_commands = self.config.get('blocked_commands', set())
        self.working_directory = self.config.get('working_directory', '/tmp')
        self.user_id = self.config.get('user_id', None)
        self.group_id = self.config.get('group_id', None)
        
        # Security events
        self.security_events: List[str] = []
    
    def _log_security_event(self, event: str):
        """Log security event."""
        self.security_events.append(event)
        logger.warning(f"Security Event: {event}")
    
    def _validate_command(self, command: str, arguments: List[str]) -> bool:
        """Validate command and arguments for security."""
        # Check if command is in allowed list
        if self.allowed_commands and command not in self.allowed_commands:
            self._log_security_event(f"Command not in allowed list: {command}")
            return False
        
        # Check if command is blocked
        if command in self.blocked_commands:
            self._log_security_event(f"Command is blocked: {command}")
            return False
        
        # Validate command path
        if not os.path.exists(command) and not self._is_system_command(command):
            self._log_security_event(f"Command not found: {command}")
            return False
        
        # Validate arguments
        for arg in arguments:
            if not self._validate_argument(arg):
                self._log_security_event(f"Invalid argument: {arg}")
                return False
        
        return True
    
    def _is_system_command(self, command: str) -> bool:
        """Check if command is a system command."""
        system_commands = ['ls', 'cat', 'grep', 'find', 'echo', 'date', 'whoami', 'pwd']
        return command in system_commands
    
    def _validate_argument(self, argument: str) -> bool:
        """Validate command argument."""
        # Sanitize argument
        sanitized = self.sanitizer.sanitize_input(argument, InputType.COMMAND, SanitizationLevel.HIGH)
        
        if not sanitized.is_safe:
            return False
        
        # Check for dangerous patterns
        dangerous_patterns = [
            r'\.\./',  # Path traversal
            r'[;&|`$(){}[\]<>]',  # Shell metacharacters
            r'\$\{[^}]*\}',  # Environment variables
            r'`[^`]*`',  # Command substitution
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, argument):
                return False
        
        return True
    
    def execute_command(self, command: str, arguments: List[str] = None, 
                       timeout: int = None, working_dir: str = None) -> CommandExecutionResult:
        """Execute command securely with input validation."""
        start_time = time.time()
        arguments = arguments or []
        
        # Validate inputs
        if not command:
            return CommandExecutionResult(
                success=False,
                command=command,
                arguments=arguments,
                error_message="Command cannot be empty",
                security_events=["empty_command"]
            )
        
        # Sanitize command
        command_sanitized = self.sanitizer.sanitize_input(command, InputType.COMMAND, SanitizationLevel.HIGH)
        if not command_sanitized.is_safe:
            return CommandExecutionResult(
                success=False,
                command=command,
                arguments=arguments,
                error_message="Command contains security threats",
                security_events=command_sanitized.threats_detected
            )
        
        # Sanitize arguments
        sanitized_arguments = []
        for arg in arguments:
            arg_sanitized = self.sanitizer.sanitize_input(arg, InputType.COMMAND, SanitizationLevel.HIGH)
            if not arg_sanitized.is_safe:
                return CommandExecutionResult(
                    success=False,
                    command=command,
                    arguments=arguments,
                    error_message=f"Argument contains security threats: {arg}",
                    security_events=arg_sanitized.threats_detected
                )
            sanitized_arguments.append(arg_sanitized.sanitized)
        
        # Validate command and arguments
        if not self._validate_command(command_sanitized.sanitized, sanitized_arguments):
            return CommandExecutionResult(
                success=False,
                command=command,
                arguments=arguments,
                error_message="Command validation failed",
                security_events=["command_validation_failed"]
            )
        
        # Set execution parameters
        timeout = timeout or self.max_execution_time
        working_dir = working_dir or self.working_directory
        
        try:
            # Prepare command execution
            cmd = [command_sanitized.sanitized] + sanitized_arguments
            
            # Set up process creation flags
            creation_flags = 0
            if os.name == 'nt':  # Windows
                creation_flags = subprocess.CREATE_NO_WINDOW
            
            # Execute command with security constraints
            process = subprocess.Popen(
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL,
                cwd=working_dir,
                creationflags=creation_flags,
                preexec_fn=self._set_process_limits if os.name != 'nt' else None
            )
            
            # Execute with timeout
            try:
                stdout, stderr = process.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate()
                return CommandExecutionResult(
                    success=False,
                    command=command,
                    arguments=arguments,
                    error_message=f"Command execution timed out after {timeout}s",
                    security_events=["execution_timeout"]
                )
            
            execution_time = time.time() - start_time
            
            # Check output size limits
            if len(stdout) > self.max_output_size:
                self._log_security_event("Output size exceeded limit")
                stdout = stdout[:self.max_output_size] + b"... (truncated)"
            
            if len(stderr) > self.max_output_size:
                stderr = stderr[:self.max_output_size] + b"... (truncated)"
            
            # Decode output
            stdout_str = stdout.decode('utf-8', errors='ignore')
            stderr_str = stderr.decode('utf-8', errors='ignore')
            
            success = process.returncode == 0
            
            return CommandExecutionResult(
                success=success,
                command=command_sanitized.sanitized,
                arguments=sanitized_arguments,
                stdout=stdout_str,
                stderr=stderr_str,
                exit_code=process.returncode,
                execution_time=execution_time,
                resource_usage={
                    'memory_usage': 'unknown',
                    'cpu_usage': 'unknown'
                }
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            self._log_security_event(f"Command execution error: {e}")
            
            return CommandExecutionResult(
                success=False,
                command=command,
                arguments=arguments,
                error_message=f"Command execution failed: {e}",
                execution_time=execution_time,
                security_events=["execution_error"]
            )
    
    def _set_process_limits(self) -> Any:
        """Set process resource limits (Unix only)."""
        try:
            
            # Set memory limit (100MB)
            resource.setrlimit(resource.RLIMIT_AS, (100 * 1024 * 1024, -1))
            
            # Set CPU time limit
            resource.setrlimit(resource.RLIMIT_CPU, (30, -1))
            
            # Set file size limit
            resource.setrlimit(resource.RLIMIT_FSIZE, (1024 * 1024, -1))
            
            # Set number of processes limit
            resource.setrlimit(resource.RLIMIT_NPROC, (10, -1))
            
        except ImportError:
            pass  # resource module not available
        except Exception as e:
            logger.warning(f"Failed to set process limits: {e}")


class InputValidator:
    """Comprehensive input validation with multiple validation layers."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize input validator."""
        self.config = config or {}
        self.sanitizer = InputSanitizer(config)
        
        # Validation rules
        self.validation_rules = {
            'username': {
                'min_length': 3,
                'max_length': 20,
                'pattern': r'^[a-zA-Z0-9_-]+$',
                'forbidden_words': ['admin', 'root', 'system', 'test']
            },
            'email': {
                'pattern': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
                'max_length': 254,
                'forbidden_domains': ['example.com', 'test.com']
            },
            'password': {
                'min_length': 12,
                'max_length': 128,
                'require_uppercase': True,
                'require_lowercase': True,
                'require_numbers': True,
                'require_special': True,
                'forbidden_patterns': [
                    r'password',
                    r'123456',
                    r'qwerty',
                    r'admin'
                ]
            },
            'url': {
                'allowed_protocols': ['http', 'https'],
                'max_length': 2048,
                'forbidden_domains': ['malicious.com', 'evil.com']
            },
            'file_path': {
                'max_length': 255,
                'allowed_extensions': ['.txt', '.pdf', '.jpg', '.png'],
                'forbidden_patterns': [
                    r'\.\./',
                    r'/etc/',
                    r'/proc/',
                    r'/sys/'
                ]
            }
        }
    
    def validate_input(self, input_data: str, input_type: InputType) -> InputValidationResult:
        """Validate input according to type-specific rules."""
        if not input_data:
            return InputValidationResult(
                valid=False,
                input_type=input_type,
                violations=["empty_input"]
            )
        
        rules = self.validation_rules.get(input_type.value, {})
        violations = []
        suggestions = []
        
        # Length validation
        if 'min_length' in rules and len(input_data) < rules['min_length']:
            violations.append(f"Too short (minimum {rules['min_length']} characters)")
            suggestions.append(f"Increase length to at least {rules['min_length']} characters")
        
        if 'max_length' in rules and len(input_data) > rules['max_length']:
            violations.append(f"Too long (maximum {rules['max_length']} characters)")
            suggestions.append(f"Reduce length to at most {rules['max_length']} characters")
        
        # Pattern validation
        if 'pattern' in rules:
            if not re.match(rules['pattern'], input_data):
                violations.append("Does not match required pattern")
                suggestions.append("Check input format requirements")
        
        # Forbidden words/patterns
        if 'forbidden_words' in rules:
            for word in rules['forbidden_words']:
                if word.lower() in input_data.lower():
                    violations.append(f"Contains forbidden word: {word}")
                    suggestions.append(f"Avoid using '{word}' in input")
        
        if 'forbidden_patterns' in rules:
            for pattern in rules['forbidden_patterns']:
                if re.search(pattern, input_data, re.IGNORECASE):
                    violations.append(f"Contains forbidden pattern: {pattern}")
                    suggestions.append("Remove forbidden pattern from input")
        
        # Type-specific validation
        if input_type == InputType.EMAIL:
            violations.extend(self._validate_email(input_data, rules))
        elif input_type == InputType.URL:
            violations.extend(self._validate_url(input_data, rules))
        elif input_type == InputType.FILE_PATH:
            violations.extend(self._validate_file_path(input_data, rules))
        elif input_type == InputType.PASSWORD:
            violations.extend(self._validate_password(input_data, rules))
        
        # Security validation
        security_result = self.sanitizer.sanitize_input(input_data, input_type, SanitizationLevel.HIGH)
        if not security_result.is_safe:
            violations.extend(security_result.threats_detected)
            suggestions.extend(security_result.recommendations)
        
        valid = len(violations) == 0
        
        # Determine risk level
        risk_level = "low"
        if len(violations) > 3:
            risk_level = "high"
        elif len(violations) > 1:
            risk_level = "medium"
        
        return InputValidationResult(
            valid=valid,
            input_type=input_type,
            validation_rules=list(rules.keys()),
            violations=violations,
            suggestions=suggestions,
            risk_level=risk_level
        )
    
    def _validate_email(self, email: str, rules: Dict[str, Any]) -> List[str]:
        """Validate email address."""
        violations = []
        
        # Check domain
        if 'forbidden_domains' in rules:
            domain = email.split('@')[-1] if '@' in email else ''
            if domain in rules['forbidden_domains']:
                violations.append(f"Forbidden domain: {domain}")
        
        return violations
    
    def _validate_url(self, url: str, rules: Dict[str, Any]) -> List[str]:
        """Validate URL."""
        violations = []
        
        try:
            parsed = urllib.parse.urlparse(url)
            
            # Check protocol
            if 'allowed_protocols' in rules:
                if parsed.scheme not in rules['allowed_protocols']:
                    violations.append(f"Protocol not allowed: {parsed.scheme}")
            
            # Check domain
            if 'forbidden_domains' in rules:
                domain = parsed.netloc
                if domain in rules['forbidden_domains']:
                    violations.append(f"Forbidden domain: {domain}")
        
        except Exception:
            violations.append("Invalid URL format")
        
        return violations
    
    def _validate_file_path(self, file_path: str, rules: Dict[str, Any]) -> List[str]:
        """Validate file path."""
        violations = []
        
        # Check extension
        if 'allowed_extensions' in rules:
            ext = Path(file_path).suffix.lower()
            if ext not in rules['allowed_extensions']:
                violations.append(f"File extension not allowed: {ext}")
        
        return violations
    
    def _validate_password(self, password: str, rules: Dict[str, Any]) -> List[str]:
        """Validate password strength."""
        violations = []
        
        # Check character requirements
        if rules.get('require_uppercase', False) and not re.search(r'[A-Z]', password):
            violations.append("Missing uppercase letter")
        
        if rules.get('require_lowercase', False) and not re.search(r'[a-z]', password):
            violations.append("Missing lowercase letter")
        
        if rules.get('require_numbers', False) and not re.search(r'\d', password):
            violations.append("Missing number")
        
        if rules.get('require_special', False) and not re.search(r'[!@#$%^&*()_+\-=\[\]{};\':"\\|,.<>\/?]', password):
            violations.append("Missing special character")
        
        return violations


# Example usage functions
def demonstrate_input_sanitization():
    """Demonstrate input sanitization."""
    sanitizer = InputSanitizer()
    
    test_inputs = [
        ("<script>alert('xss')</script>", InputType.HTML),
        ("'; DROP TABLE users; --", InputType.SQL),
        ("ls; rm -rf /", InputType.COMMAND),
        ("../../../etc/passwd", InputType.FILE_PATH),
        ("javascript:alert('xss')", InputType.URL),
        ("admin@example.com", InputType.EMAIL),
        ("normal_input", InputType.GENERAL)
    ]
    
    for input_data, input_type in test_inputs:
        result = sanitizer.sanitize_input(input_data, input_type, SanitizationLevel.HIGH)
        print(f"\nInput Type: {input_type.value}")
        print(f"Original: {input_data}")
        print(f"Sanitized: {result.sanitized}")
        print(f"Safe: {result.is_safe}")
        print(f"Threats: {result.threats_detected}")
        print(f"Risk Score: {result.risk_score:.2f}")


def demonstrate_secure_command_execution():
    """Demonstrate secure command execution."""
    config = {
        'allowed_commands': {'ls', 'cat', 'echo', 'date', 'whoami'},
        'blocked_commands': {'rm', 'del', 'format', 'sudo'},
        'max_execution_time': 10,
        'max_output_size': 1024
    }
    
    executor = SecureCommandExecutor(config)
    
    # Test safe commands
    safe_commands = [
        ("ls", ["-la"]),
        ("echo", ["Hello World"]),
        ("date", []),
        ("whoami", [])
    ]
    
    for command, args in safe_commands:
        result = executor.execute_command(command, args)
        print(f"\nCommand: {command} {' '.join(args)}")
        print(f"Success: {result.success}")
        print(f"Exit Code: {result.exit_code}")
        print(f"Output: {result.stdout[:100]}...")
        print(f"Execution Time: {result.execution_time:.3f}s")
    
    # Test dangerous commands (should be blocked)
    dangerous_commands = [
        ("rm", ["-rf", "/"]),
        ("ls", ["; rm -rf /"]),
        ("echo", ["$(rm -rf /)"]),
        ("cat", ["../../../etc/passwd"])
    ]
    
    for command, args in dangerous_commands:
        result = executor.execute_command(command, args)
        print(f"\nDangerous Command: {command} {' '.join(args)}")
        print(f"Success: {result.success}")
        print(f"Error: {result.error_message}")
        print(f"Security Events: {result.security_events}")


def demonstrate_input_validation():
    """Demonstrate input validation."""
    validator = InputValidator()
    
    test_cases = [
        ("admin", InputType.USERNAME),
        ("user@example.com", InputType.EMAIL),
        ("weak", InputType.PASSWORD),
        ("https://malicious.com/script.js", InputType.URL),
        ("../../../etc/passwd", InputType.FILE_PATH)
    ]
    
    for input_data, input_type in test_cases:
        result = validator.validate_input(input_data, input_type)
        print(f"\nInput Type: {input_type.value}")
        print(f"Input: {input_data}")
        print(f"Valid: {result.valid}")
        print(f"Risk Level: {result.risk_level}")
        print(f"Violations: {result.violations}")
        print(f"Suggestions: {result.suggestions}")


def main():
    """Main function demonstrating input sanitization and secure command execution."""
    logger.info("Starting input sanitization and secure command execution examples")
    
    # Demonstrate input sanitization
    try:
        demonstrate_input_sanitization()
    except Exception as e:
        logger.error(f"Input sanitization demonstration failed: {e}")
    
    # Demonstrate secure command execution
    try:
        demonstrate_secure_command_execution()
    except Exception as e:
        logger.error(f"Secure command execution demonstration failed: {e}")
    
    # Demonstrate input validation
    try:
        demonstrate_input_validation()
    except Exception as e:
        logger.error(f"Input validation demonstration failed: {e}")
    
    logger.info("Input sanitization and secure command execution examples completed")


match __name__:
    case "__main__":
    main() 