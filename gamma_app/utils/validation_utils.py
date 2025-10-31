"""
Gamma App - Validation Utilities
Advanced validation and sanitization utilities
"""

import re
import html
import bleach
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass
from enum import Enum
import email_validator
import phonenumbers
from urllib.parse import urlparse
import logging

logger = logging.getLogger(__name__)

class ValidationError(Exception):
    """Custom validation error"""
    pass

class ValidationSeverity(Enum):
    """Validation severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class ValidationResult:
    """Validation result"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    sanitized_value: Any = None
    severity: ValidationSeverity = ValidationSeverity.ERROR

class Validator:
    """Advanced validator class"""
    
    def __init__(self):
        self.custom_validators: Dict[str, Callable] = {}
        self.sanitizers: Dict[str, Callable] = {}
    
    def register_validator(self, name: str, validator_func: Callable):
        """Register custom validator"""
        self.custom_validators[name] = validator_func
    
    def register_sanitizer(self, name: str, sanitizer_func: Callable):
        """Register custom sanitizer"""
        self.sanitizers[name] = sanitizer_func
    
    def validate_email(self, email: str, strict: bool = True) -> ValidationResult:
        """Validate email address"""
        errors = []
        warnings = []
        
        try:
            # Use email-validator for validation
            validated_email = email_validator.validate_email(email, check_deliverability=strict)
            sanitized_email = validated_email.email.lower().strip()
            
            # Additional checks
            if len(sanitized_email) > 254:
                errors.append("Email address too long")
            
            if sanitized_email.count('@') != 1:
                errors.append("Invalid email format")
            
            # Check for suspicious patterns
            suspicious_patterns = [
                r'\+.*\+',  # Multiple plus signs
                r'\.{2,}',  # Multiple consecutive dots
                r'^\.|\.$',  # Leading or trailing dots
            ]
            
            for pattern in suspicious_patterns:
                if re.search(pattern, sanitized_email):
                    warnings.append("Email contains suspicious patterns")
                    break
            
            return ValidationResult(
                is_valid=len(errors) == 0,
                errors=errors,
                warnings=warnings,
                sanitized_value=sanitized_email
            )
            
        except email_validator.EmailNotValidError as e:
            return ValidationResult(
                is_valid=False,
                errors=[f"Invalid email: {str(e)}"],
                warnings=warnings
            )
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                errors=[f"Email validation error: {str(e)}"],
                warnings=warnings
            )
    
    def validate_phone(self, phone: str, country_code: str = "US") -> ValidationResult:
        """Validate phone number"""
        errors = []
        warnings = []
        
        try:
            # Parse phone number
            parsed_phone = phonenumbers.parse(phone, country_code)
            
            # Check if valid
            if not phonenumbers.is_valid_number(parsed_phone):
                errors.append("Invalid phone number")
            
            # Check if possible
            if not phonenumbers.is_possible_number(parsed_phone):
                warnings.append("Phone number may not be possible")
            
            # Format phone number
            formatted_phone = phonenumbers.format_number(
                parsed_phone, phonenumbers.PhoneNumberFormat.E164
            )
            
            return ValidationResult(
                is_valid=len(errors) == 0,
                errors=errors,
                warnings=warnings,
                sanitized_value=formatted_phone
            )
            
        except phonenumbers.NumberParseException as e:
            return ValidationResult(
                is_valid=False,
                errors=[f"Phone number parsing error: {str(e)}"]
            )
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                errors=[f"Phone validation error: {str(e)}"]
            )
    
    def validate_url(self, url: str, allowed_schemes: List[str] = None) -> ValidationResult:
        """Validate URL"""
        errors = []
        warnings = []
        
        if allowed_schemes is None:
            allowed_schemes = ['http', 'https']
        
        try:
            # Parse URL
            parsed_url = urlparse(url)
            
            # Check scheme
            if parsed_url.scheme not in allowed_schemes:
                errors.append(f"URL scheme not allowed. Allowed: {', '.join(allowed_schemes)}")
            
            # Check if URL is valid
            if not parsed_url.netloc:
                errors.append("Invalid URL: missing domain")
            
            # Check for suspicious patterns
            suspicious_patterns = [
                r'\.(exe|bat|cmd|scr|pif|com|vbs|js)$',  # Executable files
                r'javascript:',  # JavaScript protocol
                r'data:',  # Data protocol
                r'vbscript:',  # VBScript protocol
            ]
            
            for pattern in suspicious_patterns:
                if re.search(pattern, url, re.IGNORECASE):
                    warnings.append("URL contains suspicious patterns")
                    break
            
            # Sanitize URL
            sanitized_url = f"{parsed_url.scheme}://{parsed_url.netloc}{parsed_url.path}"
            if parsed_url.query:
                sanitized_url += f"?{parsed_url.query}"
            if parsed_url.fragment:
                sanitized_url += f"#{parsed_url.fragment}"
            
            return ValidationResult(
                is_valid=len(errors) == 0,
                errors=errors,
                warnings=warnings,
                sanitized_value=sanitized_url
            )
            
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                errors=[f"URL validation error: {str(e)}"]
            )
    
    def validate_password(self, password: str, min_length: int = 8) -> ValidationResult:
        """Validate password strength"""
        errors = []
        warnings = []
        
        # Length check
        if len(password) < min_length:
            errors.append(f"Password must be at least {min_length} characters long")
        
        # Character type checks
        if not re.search(r'[a-z]', password):
            errors.append("Password must contain at least one lowercase letter")
        
        if not re.search(r'[A-Z]', password):
            errors.append("Password must contain at least one uppercase letter")
        
        if not re.search(r'\d', password):
            errors.append("Password must contain at least one digit")
        
        if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            warnings.append("Password should contain at least one special character")
        
        # Common password patterns
        common_patterns = [
            r'password',
            r'123456',
            r'qwerty',
            r'admin',
            r'letmein',
            r'welcome',
            r'monkey',
            r'dragon',
            r'master',
            r'hello',
        ]
        
        for pattern in common_patterns:
            if re.search(pattern, password, re.IGNORECASE):
                warnings.append("Password contains common patterns")
                break
        
        # Sequential characters
        if re.search(r'(.)\1{2,}', password):
            warnings.append("Password contains repeated characters")
        
        # Sequential numbers/letters
        if re.search(r'(012|123|234|345|456|567|678|789|890)', password):
            warnings.append("Password contains sequential numbers")
        
        if re.search(r'(abc|bcd|cde|def|efg|fgh|ghi|hij|ijk|jkl|klm|lmn|mno|nop|opq|pqr|qrs|rst|stu|tuv|uvw|vwx|wxy|xyz)', password, re.IGNORECASE):
            warnings.append("Password contains sequential letters")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            sanitized_value=password
        )
    
    def validate_username(self, username: str, min_length: int = 3, max_length: int = 30) -> ValidationResult:
        """Validate username"""
        errors = []
        warnings = []
        
        # Length check
        if len(username) < min_length:
            errors.append(f"Username must be at least {min_length} characters long")
        
        if len(username) > max_length:
            errors.append(f"Username must be no more than {max_length} characters long")
        
        # Character validation
        if not re.match(r'^[a-zA-Z0-9_-]+$', username):
            errors.append("Username can only contain letters, numbers, underscores, and hyphens")
        
        # Start/end validation
        if username.startswith('-') or username.endswith('-'):
            errors.append("Username cannot start or end with a hyphen")
        
        if username.startswith('_') or username.endswith('_'):
            errors.append("Username cannot start or end with an underscore")
        
        # Reserved usernames
        reserved_usernames = [
            'admin', 'administrator', 'root', 'user', 'guest', 'test',
            'api', 'www', 'mail', 'ftp', 'support', 'help', 'info',
            'contact', 'about', 'privacy', 'terms', 'legal', 'security'
        ]
        
        if username.lower() in reserved_usernames:
            errors.append("Username is reserved")
        
        # Sanitize username
        sanitized_username = username.lower().strip()
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            sanitized_value=sanitized_username
        )
    
    def validate_json(self, json_string: str) -> ValidationResult:
        """Validate JSON string"""
        errors = []
        warnings = []
        
        try:
            import json
            parsed_json = json.loads(json_string)
            
            # Check for common issues
            if isinstance(parsed_json, dict):
                # Check for circular references (basic check)
                if len(str(parsed_json)) > 1000000:  # 1MB limit
                    warnings.append("JSON object is very large")
                
                # Check for suspicious keys
                suspicious_keys = ['__proto__', 'constructor', 'prototype']
                for key in parsed_json.keys():
                    if key in suspicious_keys:
                        warnings.append(f"Suspicious key found: {key}")
            
            return ValidationResult(
                is_valid=True,
                errors=errors,
                warnings=warnings,
                sanitized_value=parsed_json
            )
            
        except json.JSONDecodeError as e:
            return ValidationResult(
                is_valid=False,
                errors=[f"Invalid JSON: {str(e)}"]
            )
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                errors=[f"JSON validation error: {str(e)}"]
            )
    
    def sanitize_html(self, html_content: str, allowed_tags: List[str] = None) -> ValidationResult:
        """Sanitize HTML content"""
        errors = []
        warnings = []
        
        if allowed_tags is None:
            allowed_tags = [
                'p', 'br', 'strong', 'em', 'u', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
                'ul', 'ol', 'li', 'a', 'img', 'blockquote', 'code', 'pre'
            ]
        
        try:
            # Use bleach for HTML sanitization
            sanitized_html = bleach.clean(
                html_content,
                tags=allowed_tags,
                attributes={
                    'a': ['href', 'title'],
                    'img': ['src', 'alt', 'title', 'width', 'height'],
                },
                protocols=['http', 'https', 'mailto']
            )
            
            # Check for potential XSS
            if '<script' in html_content.lower() or 'javascript:' in html_content.lower():
                warnings.append("HTML content contained potentially dangerous elements")
            
            return ValidationResult(
                is_valid=True,
                errors=errors,
                warnings=warnings,
                sanitized_value=sanitized_html
            )
            
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                errors=[f"HTML sanitization error: {str(e)}"]
            )
    
    def sanitize_sql(self, sql_string: str) -> ValidationResult:
        """Sanitize SQL string (basic protection)"""
        errors = []
        warnings = []
        
        # SQL injection patterns
        sql_patterns = [
            r'union\s+select',
            r'drop\s+table',
            r'delete\s+from',
            r'insert\s+into',
            r'update\s+set',
            r'exec\s*\(',
            r'execute\s*\(',
            r'sp_',
            r'xp_',
            r'--',
            r'/\*',
            r'\*/',
            r';',
            r'waitfor\s+delay',
            r'benchmark\s*\(',
        ]
        
        for pattern in sql_patterns:
            if re.search(pattern, sql_string, re.IGNORECASE):
                warnings.append("SQL string contains potentially dangerous patterns")
                break
        
        # Basic sanitization
        sanitized_sql = html.escape(sql_string)
        
        return ValidationResult(
            is_valid=True,
            errors=errors,
            warnings=warnings,
            sanitized_value=sanitized_sql
        )
    
    def validate_file_extension(self, filename: str, allowed_extensions: List[str]) -> ValidationResult:
        """Validate file extension"""
        errors = []
        warnings = []
        
        if not filename:
            errors.append("Filename is required")
            return ValidationResult(is_valid=False, errors=errors)
        
        # Get file extension
        file_ext = filename.lower().split('.')[-1] if '.' in filename else ''
        
        if not file_ext:
            errors.append("File must have an extension")
        elif file_ext not in allowed_extensions:
            errors.append(f"File extension '{file_ext}' not allowed. Allowed: {', '.join(allowed_extensions)}")
        
        # Check for suspicious extensions
        suspicious_extensions = [
            'exe', 'bat', 'cmd', 'scr', 'pif', 'com', 'vbs', 'js', 'jar',
            'php', 'asp', 'aspx', 'jsp', 'py', 'pl', 'sh', 'ps1'
        ]
        
        if file_ext in suspicious_extensions:
            warnings.append(f"File extension '{file_ext}' may be potentially dangerous")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            sanitized_value=filename
        )
    
    def validate_file_size(self, file_size: int, max_size: int) -> ValidationResult:
        """Validate file size"""
        errors = []
        warnings = []
        
        if file_size <= 0:
            errors.append("File size must be greater than 0")
        elif file_size > max_size:
            errors.append(f"File size exceeds maximum allowed size of {max_size} bytes")
        
        # Warning for large files
        if file_size > max_size * 0.8:
            warnings.append("File size is close to the maximum limit")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            sanitized_value=file_size
        )
    
    def validate_ip_address(self, ip_address: str) -> ValidationResult:
        """Validate IP address"""
        errors = []
        warnings = []
        
        try:
            import ipaddress
            
            # Parse IP address
            ip = ipaddress.ip_address(ip_address)
            
            # Check for private IPs
            if ip.is_private:
                warnings.append("IP address is private")
            
            # Check for loopback
            if ip.is_loopback:
                warnings.append("IP address is loopback")
            
            # Check for link-local
            if ip.is_link_local:
                warnings.append("IP address is link-local")
            
            return ValidationResult(
                is_valid=True,
                errors=errors,
                warnings=warnings,
                sanitized_value=str(ip)
            )
            
        except ValueError as e:
            return ValidationResult(
                is_valid=False,
                errors=[f"Invalid IP address: {str(e)}"]
            )
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                errors=[f"IP validation error: {str(e)}"]
            )
    
    def validate_custom(self, value: Any, validator_name: str) -> ValidationResult:
        """Validate using custom validator"""
        if validator_name not in self.custom_validators:
            return ValidationResult(
                is_valid=False,
                errors=[f"Custom validator '{validator_name}' not found"]
            )
        
        try:
            return self.custom_validators[validator_name](value)
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                errors=[f"Custom validation error: {str(e)}"]
            )
    
    def sanitize_custom(self, value: Any, sanitizer_name: str) -> Any:
        """Sanitize using custom sanitizer"""
        if sanitizer_name not in self.sanitizers:
            return value
        
        try:
            return self.sanitizers[sanitizer_name](value)
        except Exception as e:
            logger.error(f"Custom sanitization error: {e}")
            return value

# Global validator instance
validator = Validator()

def validate_email(email: str, strict: bool = True) -> ValidationResult:
    """Validate email using global validator"""
    return validator.validate_email(email, strict)

def validate_phone(phone: str, country_code: str = "US") -> ValidationResult:
    """Validate phone using global validator"""
    return validator.validate_phone(phone, country_code)

def validate_url(url: str, allowed_schemes: List[str] = None) -> ValidationResult:
    """Validate URL using global validator"""
    return validator.validate_url(url, allowed_schemes)

def validate_password(password: str, min_length: int = 8) -> ValidationResult:
    """Validate password using global validator"""
    return validator.validate_password(password, min_length)

def validate_username(username: str, min_length: int = 3, max_length: int = 30) -> ValidationResult:
    """Validate username using global validator"""
    return validator.validate_username(username, min_length, max_length)

def sanitize_html(html_content: str, allowed_tags: List[str] = None) -> ValidationResult:
    """Sanitize HTML using global validator"""
    return validator.sanitize_html(html_content, allowed_tags)

























