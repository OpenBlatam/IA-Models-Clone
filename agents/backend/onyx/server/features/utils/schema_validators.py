from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
BUFFER_SIZE = 1024

from __future__ import annotations
from typing import Any, Dict, List, Optional, Type, TypeVar, Union, Callable, Set
from datetime import datetime, date, timedelta
import re
import hashlib
import json
from functools import wraps, lru_cache
import time
from pydantic import ValidationError, field_validator, model_validator
import structlog
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Schema Validators - Custom Validation and Business Rules
Advanced validation system with custom validators, business rules, and performance monitoring.
"""



logger = structlog.get_logger(__name__)

T = TypeVar('T')

class ValidationResult:
    """Result of a validation operation."""
    
    def __init__(self, is_valid: bool, errors: List[str] = None, warnings: List[str] = None):
        
    """__init__ function."""
self.is_valid = is_valid
        self.errors = errors or []
        self.warnings = warnings or []
        self.timestamp = datetime.utcnow()
    
    def add_error(self, error: str) -> None:
        """Add an error message."""
        self.errors.append(error)
        self.is_valid = False
    
    def add_warning(self, warning: str) -> None:
        """Add a warning message."""
        self.warnings.append(warning)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_valid": self.is_valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "timestamp": self.timestamp.isoformat(),
            "error_count": len(self.errors),
            "warning_count": len(self.warnings)
        }

class BaseValidator:
    """Base class for all validators."""
    
    def __init__(self, name: str, description: str = ""):
        
    """__init__ function."""
self.name = name
        self.description = description
        self.validation_count = 0
        self.error_count = 0
        self.start_time = time.perf_counter()
    
    def validate(self, value: Any, **kwargs) -> ValidationResult:
        """Validate a value."""
        self.validation_count += 1
        start_time = time.perf_counter()
        
        try:
            result = self._validate_impl(value, **kwargs)
            if not result.is_valid:
                self.error_count += 1
            
            # Log validation performance
            duration = time.perf_counter() - start_time
            if duration > 0.1:  # Log slow validations
                logger.warning(
                    "Slow validation detected",
                    validator=self.name,
                    duration_ms=duration * 1000,
                    value_type=type(value).__name__
                )
            
            return result
            
        except Exception as e:
            self.error_count += 1
            logger.error(
                "Validation error",
                validator=self.name,
                error=str(e),
                value_type=type(value).__name__
            )
            return ValidationResult(False, [f"Validation error: {str(e)}"])
    
    def _validate_impl(self, value: Any, **kwargs) -> ValidationResult:
        """Implementation of validation logic."""
        raise NotImplementedError
    
    def get_stats(self) -> Dict[str, Any]:
        """Get validation statistics."""
        total_time = time.perf_counter() - self.start_time
        return {
            "name": self.name,
            "description": self.description,
            "validation_count": self.validation_count,
            "error_count": self.error_count,
            "success_rate": (self.validation_count - self.error_count) / max(self.validation_count, 1),
            "total_time_seconds": total_time,
            "average_time_ms": (total_time / max(self.validation_count, 1)) * 1000
        }

class StringValidator(BaseValidator):
    """Validator for string fields."""
    
    def __init__(self, min_length: int = 0, max_length: int = None, pattern: str = None, 
                 allowed_values: Set[str] = None, case_sensitive: bool = True):
        
    """__init__ function."""
super().__init__("StringValidator")
        self.min_length = min_length
        self.max_length = max_length
        self.pattern = pattern
        self.allowed_values = allowed_values
        self.case_sensitive = case_sensitive
    
    def _validate_impl(self, value: Any, **kwargs) -> ValidationResult:
        result = ValidationResult(True)
        
        if not isinstance(value, str):
            result.add_error("Value must be a string")
            return result
        
        # Length validation
        if len(value) < self.min_length:
            result.add_error(f"String must be at least {self.min_length} characters long")
        
        if self.max_length and len(value) > self.max_length:
            result.add_error(f"String must be at most {self.max_length} characters long")
        
        # Pattern validation
        if self.pattern and not re.match(self.pattern, value):
            result.add_error(f"String does not match pattern: {self.pattern}")
        
        # Allowed values validation
        if self.allowed_values:
            check_value = value if self.case_sensitive else value.lower()
            if check_value not in self.allowed_values:
                result.add_error(f"Value must be one of: {', '.join(self.allowed_values)}")
        
        return result

class EmailValidator(BaseValidator):
    """Validator for email addresses."""
    
    def __init__(self, allow_disposable: bool = False, check_mx: bool = False):
        
    """__init__ function."""
super().__init__("EmailValidator")
        self.allow_disposable = allow_disposable
        self.check_mx = check_mx
        self._disposable_domains = self._load_disposable_domains()
    
    def _validate_impl(self, value: Any, **kwargs) -> ValidationResult:
        result = ValidationResult(True)
        
        if not isinstance(value, str):
            result.add_error("Email must be a string")
            return result
        
        # Basic email pattern
        email_pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        if not re.match(email_pattern, value):
            result.add_error("Invalid email format")
            return result
        
        # Normalize email
        email = value.lower().strip()
        
        # Check for disposable domains
        if not self.allow_disposable:
            domain = email.split('@')[1]
            if domain in self._disposable_domains:
                result.add_error("Disposable email addresses are not allowed")
        
        # Check for common spam patterns
        spam_patterns = [
            r"^[0-9]+@",  # Starts with numbers
            r"^[a-z]{1,2}@",  # Very short local part
            r"@[0-9]+\.[a-z]+$",  # IP-based domain
        ]
        
        for pattern in spam_patterns:
            if re.match(pattern, email):
                result.add_warning("Email address may be suspicious")
                break
        
        return result
    
    @lru_cache(maxsize=1)
    def _load_disposable_domains(self) -> Set[str]:
        """Load list of disposable email domains."""
        # Common disposable domains
        return {
            '10minutemail.com', 'guerrillamail.com', 'mailinator.com',
            'tempmail.org', 'throwaway.email', 'yopmail.com'
        }

class PhoneValidator(BaseValidator):
    """Validator for phone numbers."""
    
    def __init__(self, country_code: str = "US", allow_international: bool = True):
        
    """__init__ function."""
super().__init__("PhoneValidator")
        self.country_code = country_code
        self.allow_international = allow_international
    
    def _validate_impl(self, value: Any, **kwargs) -> ValidationResult:
        result = ValidationResult(True)
        
        if not isinstance(value, str):
            result.add_error("Phone number must be a string")
            return result
        
        # Remove all non-digit characters
        digits_only = re.sub(r'\D', '', value)
        
        # Validate length
        if len(digits_only) < 10:
            result.add_error("Phone number must be at least 10 digits")
        elif len(digits_only) > 15:
            result.add_error("Phone number must be at most 15 digits")
        
        # Country-specific validation
        if self.country_code == "US" and len(digits_only) == 10:
            # US phone number validation
            area_code = digits_only[:3]
            if area_code.startswith('0') or area_code.startswith('1'):
                result.add_error("Invalid US area code")
        
        return result

class URLValidator(BaseValidator):
    """Validator for URLs."""
    
    def __init__(self, allowed_schemes: Set[str] = None, allowed_domains: Set[str] = None,
                 block_suspicious: bool = True):
        
    """__init__ function."""
super().__init__("URLValidator")
        self.allowed_schemes = allowed_schemes or {'http', 'https'}
        self.allowed_domains = allowed_domains
        self.block_suspicious = block_suspicious
    
    def _validate_impl(self, value: Any, **kwargs) -> ValidationResult:
        result = ValidationResult(True)
        
        if not isinstance(value, str):
            result.add_error("URL must be a string")
            return result
        
        # Basic URL pattern
        url_pattern = r"^https?://[^\s/$.?#].[^\s]*$"
        if not re.match(url_pattern, value):
            result.add_error("Invalid URL format")
            return result
        
        # Check scheme
        scheme = value.split('://')[0].lower()
        if scheme not in self.allowed_schemes:
            result.add_error(f"URL scheme '{scheme}' not allowed")
        
        # Check domain
        if self.allowed_domains:
            domain = value.split('://')[1].split('/')[0].lower()
            if domain not in self.allowed_domains:
                result.add_error(f"Domain '{domain}' not allowed")
        
        # Check for suspicious patterns
        if self.block_suspicious:
            suspicious_patterns = [
                r'\.(tk|ml|ga|cf|gq)$',  # Free domains
                r'[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}',  # IP addresses
                r'bit\.ly|goo\.gl|tinyurl\.com',  # URL shorteners
            ]
            
            for pattern in suspicious_patterns:
                if re.search(pattern, value, re.IGNORECASE):
                    result.add_warning("URL may be suspicious")
                    break
        
        return result

class DateValidator(BaseValidator):
    """Validator for dates."""
    
    def __init__(self, min_date: date = None, max_date: date = None, 
                 allow_future: bool = True, allow_past: bool = True):
        
    """__init__ function."""
super().__init__("DateValidator")
        self.min_date = min_date
        self.max_date = max_date
        self.allow_future = allow_future
        self.allow_past = allow_past
    
    def _validate_impl(self, value: Any, **kwargs) -> ValidationResult:
        result = ValidationResult(True)
        
        # Convert string to date if needed
        if isinstance(value, str):
            try:
                value = datetime.fromisoformat(value).date()
            except ValueError:
                result.add_error("Invalid date format")
                return result
        
        if not isinstance(value, (date, datetime)):
            result.add_error("Value must be a date")
            return result
        
        # Convert datetime to date
        if isinstance(value, datetime):
            value = value.date()
        
        today = date.today()
        
        # Check date range
        if self.min_date and value < self.min_date:
            result.add_error(f"Date must be after {self.min_date}")
        
        if self.max_date and value > self.max_date:
            result.add_error(f"Date must be before {self.max_date}")
        
        # Check future/past restrictions
        if not self.allow_future and value > today:
            result.add_error("Future dates are not allowed")
        
        if not self.allow_past and value < today:
            result.add_error("Past dates are not allowed")
        
        return result

class FileValidator(BaseValidator):
    """Validator for file uploads."""
    
    def __init__(self, allowed_types: Set[str] = None, max_size_mb: int = 50,
                 allowed_extensions: Set[str] = None):
        
    """__init__ function."""
super().__init__("FileValidator")
        self.allowed_types = allowed_types or {'image/jpeg', 'image/png', 'image/gif', 'application/pdf'}
        self.max_size_mb = max_size_mb
        self.allowed_extensions = allowed_extensions or {'jpg', 'jpeg', 'png', 'gif', 'pdf', 'doc', 'docx'}
    
    def _validate_impl(self, value: Any, **kwargs) -> ValidationResult:
        result = ValidationResult(True)
        
        # Check if value is a file-like object or dict with file info
        if hasattr(value, 'content_type') and hasattr(value, 'size'):
            # File-like object
            content_type = value.content_type
            size_mb = value.size / (1024 * 1024)
            filename = getattr(value, 'filename', '')
        elif isinstance(value, dict):
            # Dict with file info
            content_type = value.get('content_type')
            size_mb = value.get('size_mb', 0)
            filename = value.get('filename', '')
        else:
            result.add_error("Invalid file format")
            return result
        
        # Check content type
        if content_type not in self.allowed_types:
            result.add_error(f"File type '{content_type}' not allowed")
        
        # Check file size
        if size_mb > self.max_size_mb:
            result.add_error(f"File size ({size_mb:.1f}MB) exceeds maximum ({self.max_size_mb}MB)")
        
        # Check file extension
        if filename:
            extension = filename.split('.')[-1].lower()
            if extension not in self.allowed_extensions:
                result.add_error(f"File extension '.{extension}' not allowed")
        
        return result

class BusinessRuleValidator(BaseValidator):
    """Validator for business-specific rules."""
    
    def __init__(self, rules: List[Callable] = None):
        
    """__init__ function."""
super().__init__("BusinessRuleValidator")
        self.rules = rules or []
    
    def add_rule(self, rule: Callable) -> None:
        """Add a business rule."""
        self.rules.append(rule)
    
    def _validate_impl(self, value: Any, **kwargs) -> ValidationResult:
        result = ValidationResult(True)
        
        for rule in self.rules:
            try:
                rule_result = rule(value, **kwargs)
                if isinstance(rule_result, str):
                    result.add_error(rule_result)
                elif isinstance(rule_result, bool) and not rule_result:
                    result.add_error(f"Business rule validation failed")
            except Exception as e:
                result.add_error(f"Business rule error: {str(e)}")
        
        return result

class ValidationRegistry:
    """Registry for managing validators."""
    
    def __init__(self) -> Any:
        self._validators: Dict[str, BaseValidator] = {}
        self._validator_stats: Dict[str, Dict[str, Any]] = {}
    
    def register_validator(self, name: str, validator: BaseValidator) -> None:
        """Register a validator."""
        self._validators[name] = validator
        logger.info(f"Registered validator: {name}")
    
    def get_validator(self, name: str) -> Optional[BaseValidator]:
        """Get a validator by name."""
        return self._validators.get(name)
    
    def validate(self, name: str, value: Any, **kwargs) -> ValidationResult:
        """Validate a value using a registered validator."""
        validator = self.get_validator(name)
        if not validator:
            return ValidationResult(False, [f"Validator '{name}' not found"])
        
        result = validator.validate(value, **kwargs)
        
        # Update statistics
        if name not in self._validator_stats:
            self._validator_stats[name] = {"total": 0, "errors": 0, "warnings": 0}
        
        self._validator_stats[name]["total"] += 1
        if not result.is_valid:
            self._validator_stats[name]["errors"] += 1
        if result.warnings:
            self._validator_stats[name]["warnings"] += len(result.warnings)
        
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        """Get validation statistics."""
        stats = {}
        for name, validator in self._validators.items():
            stats[name] = {
                **validator.get_stats(),
                "registry_stats": self._validator_stats.get(name, {})
            }
        return stats

# Global validation registry
validation_registry = ValidationRegistry()

# Register common validators
validation_registry.register_validator("string", StringValidator())
validation_registry.register_validator("email", EmailValidator())
validation_registry.register_validator("phone", PhoneValidator())
validation_registry.register_validator("url", URLValidator())
validation_registry.register_validator("date", DateValidator())
validation_registry.register_validator("file", FileValidator())

# Business rule examples
def validate_user_age(value: Dict[str, Any], **kwargs) -> Union[bool, str]:
    """Validate user age is at least 13 years old."""
    if 'birth_date' in value:
        birth_date = value['birth_date']
        if isinstance(birth_date, str):
            birth_date = datetime.fromisoformat(birth_date).date()
        
        age = (date.today() - birth_date).days / 365.25
        if age < 13:
            return "User must be at least 13 years old"
    
    return True

def validate_password_strength(value: Dict[str, Any], **kwargs) -> Union[bool, str]:
    """Validate password strength."""
    password = value.get('password', '')
    
    if len(password) < 8:
        return "Password must be at least 8 characters long"
    
    if not re.search(r'[A-Z]', password):
        return "Password must contain at least one uppercase letter"
    
    if not re.search(r'[a-z]', password):
        return "Password must contain at least one lowercase letter"
    
    if not re.search(r'\d', password):
        return "Password must contain at least one digit"
    
    if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
        return "Password must contain at least one special character"
    
    return True

def validate_unique_email(value: Dict[str, Any], **kwargs) -> Union[bool, str]:
    """Validate email uniqueness (mock implementation)."""
    email = value.get('email', '')
    
    # Mock check - in real implementation, this would query the database
    existing_emails = {'test@example.com', 'admin@example.com'}
    if email in existing_emails:
        return "Email address already exists"
    
    return True

# Register business rule validators
user_business_validator = BusinessRuleValidator([
    validate_user_age,
    validate_password_strength,
    validate_unique_email
])
validation_registry.register_validator("user_business", user_business_validator)

# Pydantic field validators
def validate_email_field(value: str) -> str:
    """Pydantic field validator for email."""
    result = validation_registry.validate("email", value)
    if not result.is_valid:
        raise ValueError(result.errors[0])
    return value.lower().strip()

def validate_phone_field(value: str) -> str:
    """Pydantic field validator for phone."""
    result = validation_registry.validate("phone", value)
    if not result.is_valid:
        raise ValueError(result.errors[0])
    return re.sub(r'\D', '', value)

def validate_url_field(value: str) -> str:
    """Pydantic field validator for URL."""
    result = validation_registry.validate("url", value)
    if not result.is_valid:
        raise ValueError(result.errors[0])
    return value if value.startswith(('http://', 'https://')) else f'https://{value}'

def validate_date_field(value: Union[str, date, datetime]) -> date:
    """Pydantic field validator for date."""
    if isinstance(value, str):
        value = datetime.fromisoformat(value).date()
    elif isinstance(value, datetime):
        value = value.date()
    
    result = validation_registry.validate("date", value)
    if not result.is_valid:
        raise ValueError(result.errors[0])
    
    return value

# Decorator for business rule validation
def validate_business_rules(validator_name: str = None):
    """Decorator for business rule validation."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Extract data from function arguments
            data = {}
            if args and isinstance(args[0], dict):
                data = args[0]
            elif kwargs:
                data = kwargs
            
            # Validate business rules
            if validator_name:
                result = validation_registry.validate(validator_name, data)
                if not result.is_valid:
                    raise ValidationError(f"Business validation failed: {', '.join(result.errors)}")
            
            return func(*args, **kwargs)
        return wrapper
    return decorator 