"""Advanced validation utilities."""

from typing import Any, Callable, List, Optional, Dict
import re
from datetime import datetime
from uuid import UUID
from fastapi import UploadFile

from core.exceptions import ImageValidationError
from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)


class Validator:
    """Validation utility class with static methods."""
    
    @staticmethod
    def is_email(value: str) -> bool:
        """Check if value is valid email."""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, value))
    
    @staticmethod
    def validate_email(value: str) -> bool:
        """Validate email (alias for is_email)."""
        return Validator.is_email(value)
    
    @staticmethod
    def is_url(value: str) -> bool:
        """Check if value is valid URL."""
        pattern = r'^https?://[^\s/$.?#].[^\s]*$'
        return bool(re.match(pattern, value))
    
    @staticmethod
    def validate_url(value: str) -> bool:
        """Validate URL (alias for is_url)."""
        return Validator.is_url(value)
    
    @staticmethod
    def is_phone(value: str) -> bool:
        """Check if value is valid phone number."""
        pattern = r'^\+?[\d\s\-\(\)]{10,}$'
        return bool(re.match(pattern, value))
    
    @staticmethod
    def validate_phone(value: str) -> bool:
        """Validate phone (alias for is_phone)."""
        return Validator.is_phone(value)
    
    @staticmethod
    def is_credit_card(value: str) -> bool:
        """Check if value is valid credit card (Luhn algorithm)."""
        value = value.replace(' ', '').replace('-', '')
        if not value.isdigit():
            return False
        
        def luhn_check(card_number: str) -> bool:
            def digits_of(n):
                return [int(d) for d in str(n)]
            digits = digits_of(card_number)
            odd_digits = digits[-1::-2]
            even_digits = digits[-2::-2]
            checksum = sum(odd_digits)
            for d in even_digits:
                checksum += sum(digits_of(d * 2))
            return checksum % 10 == 0
        
        return luhn_check(value)
    
    @staticmethod
    def is_uuid(value: str) -> bool:
        """Check if value is valid UUID."""
        pattern = r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
        return bool(re.match(pattern, value, re.IGNORECASE))
    
    @staticmethod
    def is_strong_password(value: str, min_length: int = 8) -> bool:
        """Check if value is strong password."""
        if len(value) < min_length:
            return False
        has_upper = bool(re.search(r'[A-Z]', value))
        has_lower = bool(re.search(r'[a-z]', value))
        has_digit = bool(re.search(r'\d', value))
        has_special = bool(re.search(r'[!@#$%^&*(),.?":{}|<>]', value))
        return has_upper and has_lower and has_digit and has_special
    
    @staticmethod
    def is_date(value: str, format: str = '%Y-%m-%d') -> bool:
        """Check if value is valid date."""
        try:
            datetime.strptime(value, format)
            return True
        except (ValueError, TypeError):
            return False
    
    @staticmethod
    def is_in_range(value: Any, min_val: Any, max_val: Any) -> bool:
        """Check if value is in range."""
        return min_val <= value <= max_val
    
    @staticmethod
    def is_one_of(value: Any, options: List[Any]) -> bool:
        """Check if value is one of options."""
        return value in options
    
    @staticmethod
    def matches_pattern(value: str, pattern: str) -> bool:
        """Check if value matches pattern."""
        return bool(re.match(pattern, value))
    
    @staticmethod
    def has_length(value: Any, length: int) -> bool:
        """Check if value has specific length."""
        try:
            return len(value) == length
        except TypeError:
            return False
    
    @staticmethod
    def has_min_length(value: Any, min_length: int) -> bool:
        """Check if value has minimum length."""
        try:
            return len(value) >= min_length
        except TypeError:
            return False
    
    @staticmethod
    def has_max_length(value: Any, max_length: int) -> bool:
        """Check if value has maximum length."""
        try:
            return len(value) <= max_length
        except TypeError:
            return False


class ValidationRule:
    """Validation rule."""
    
    def __init__(self, validator: Callable, message: str = "Validation failed"):
        self.validator = validator
        self.message = message
    
    def validate(self, value: Any) -> tuple[bool, Optional[str]]:
        """
        Validate value.
        
        Returns:
            (is_valid, error_message)
        """
        try:
            if self.validator(value):
                return True, None
            return False, self.message
        except Exception as e:
            return False, f"Validation error: {str(e)}"


class ValidationSchema:
    """Validation schema."""
    
    def __init__(self):
        self._rules: Dict[str, List[ValidationRule]] = {}
    
    def add_rule(self, field: str, rule: ValidationRule) -> 'ValidationSchema':
        """Add validation rule for field."""
        if field not in self._rules:
            self._rules[field] = []
        self._rules[field].append(rule)
        return self
    
    def validate(self, data: Dict[str, Any]) -> tuple[bool, Dict[str, List[str]]]:
        """
        Validate data against schema.
        
        Returns:
            (is_valid, errors)
        """
        errors: Dict[str, List[str]] = {}
        
        for field, rules in self._rules.items():
            value = data.get(field)
            for rule in rules:
                is_valid, error = rule.validate(value)
                if not is_valid:
                    if field not in errors:
                        errors[field] = []
                    errors[field].append(error)
        
        return len(errors) == 0, errors


# FastAPI-specific validators
async def validate_uploaded_file(file: UploadFile) -> bytes:
    """
    Validate uploaded file.
    
    Args:
        file: Uploaded file
        
    Returns:
        File content as bytes
        
    Raises:
        ImageValidationError: If file is invalid
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise ImageValidationError(
            f"Invalid file type: {file.content_type}. Expected image file."
        )
    
    if file.filename:
        ext = file.filename.split(".")[-1].lower()
        if ext not in settings.supported_formats:
            raise ImageValidationError(
                f"Unsupported file format: {ext}. "
                f"Supported formats: {', '.join(settings.supported_formats)}"
            )
    
    content = await file.read()
    size_mb = len(content) / (1024 * 1024)
    if size_mb > settings.max_image_size_mb:
        raise ImageValidationError(
            f"File size ({size_mb:.2f}MB) exceeds maximum ({settings.max_image_size_mb}MB)"
        )
    
    await file.seek(0)
    return content


def validate_intensity(intensity: Optional[float], default: float = 0.5) -> float:
    """
    Validate and normalize intensity value.
    
    Args:
        intensity: Intensity value to validate
        default: Default value if None
        
    Returns:
        Validated intensity value
    """
    if intensity is None:
        return default
    
    if not isinstance(intensity, (int, float)):
        raise ValueError("Intensity must be a number")
    
    if intensity < 0.0 or intensity > 1.0:
        raise ValueError("Intensity must be between 0.0 and 1.0")
    
    return float(intensity)


def validate_visualization_id(visualization_id: str) -> str:
    """
    Validate visualization ID format.
    
    Args:
        visualization_id: ID to validate
        
    Returns:
        Validated ID
        
    Raises:
        ValueError: If ID is invalid
    """
    if not visualization_id:
        raise ValueError("Visualization ID cannot be empty")
    
    if len(visualization_id) > 100:
        raise ValueError("Visualization ID too long")
    
    if len(visualization_id) != 36 and "-" not in visualization_id:
        raise ValueError("Invalid visualization ID format")
    
    return visualization_id


# Simple validation functions
def validate_uuid(uuid_string: str) -> bool:
    """
    Validate UUID format.
    
    Args:
        uuid_string: UUID string to validate
        
    Returns:
        True if valid UUID
    """
    try:
        UUID(uuid_string)
        return True
    except (ValueError, TypeError):
        return False


def validate_filename(filename: str) -> bool:
    """
    Validate filename format.
    
    Args:
        filename: Filename to validate
        
    Returns:
        True if valid
    """
    if not filename or len(filename) > 255:
        return False
    
    dangerous_pattern = r'[<>:"|?*\x00-\x1f]'
    if re.search(dangerous_pattern, filename):
        return False
    
    if '..' in filename or '/' in filename or '\\' in filename:
        return False
    
    return True


def validate_intensity_range(intensity: float, min_val: float = 0.0, max_val: float = 1.0) -> bool:
    """
    Validate intensity is within range.
    
    Args:
        intensity: Intensity value
        min_val: Minimum value
        max_val: Maximum value
        
    Returns:
        True if valid
    """
    return min_val <= intensity <= max_val


def sanitize_string(value: str, max_length: int = 1000) -> str:
    """
    Sanitize string input.
    
    Args:
        value: String to sanitize
        max_length: Maximum length
        
    Returns:
        Sanitized string
    """
    if not value:
        return ""
    
    sanitized = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', value)
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length]
    
    return sanitized.strip()


# Class-based validators (from validation_advanced.py)
class BaseValidator:
    """Base validator class for class-based validation."""
    
    def __init__(self, error_message: str = "Validation failed"):
        self.error_message = error_message
    
    def validate(self, value: Any) -> bool:
        """Validate value."""
        raise NotImplementedError
    
    def __call__(self, value: Any) -> bool:
        """Make validator callable."""
        return self.validate(value)


class EmailValidator(BaseValidator):
    """Email validator class."""
    
    def __init__(self):
        super().__init__("Invalid email address")
        from email.utils import parseaddr
        self.parseaddr = parseaddr
        self.pattern = re.compile(
            r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        )
    
    def validate(self, value: Any) -> bool:
        """Validate email address."""
        if not isinstance(value, str):
            return False
        name, addr = self.parseaddr(value)
        if not addr:
            return False
        return bool(self.pattern.match(addr.lower()))


class URLValidator(BaseValidator):
    """URL validator class."""
    
    def __init__(self, allowed_schemes: Optional[List[str]] = None):
        super().__init__("Invalid URL")
        self.allowed_schemes = allowed_schemes or ["http", "https"]
    
    def validate(self, value: Any) -> bool:
        """Validate URL."""
        if not isinstance(value, str):
            return False
        try:
            from urllib.parse import urlparse
            parsed = urlparse(value)
            if not parsed.scheme or parsed.scheme not in self.allowed_schemes:
                return False
            if not parsed.netloc:
                return False
            return True
        except Exception:
            return False


class PhoneValidator(BaseValidator):
    """Phone number validator class."""
    
    def __init__(self):
        super().__init__("Invalid phone number")
        self.pattern = re.compile(r'^\+?[1-9]\d{1,14}$')
    
    def validate(self, value: Any) -> bool:
        """Validate phone number."""
        if not isinstance(value, str):
            return False
        cleaned = re.sub(r'[\s\-\(\)]', '', value)
        return bool(self.pattern.match(cleaned))


class RangeValidator(BaseValidator):
    """Range validator class."""
    
    def __init__(self, min_value: float, max_value: float):
        super().__init__(f"Value must be between {min_value} and {max_value}")
        self.min_value = min_value
        self.max_value = max_value
    
    def validate(self, value: Any) -> bool:
        """Validate value is in range."""
        try:
            num_value = float(value)
            return self.min_value <= num_value <= self.max_value
        except (ValueError, TypeError):
            return False


class LengthValidator(BaseValidator):
    """Length validator class."""
    
    def __init__(self, min_length: int, max_length: Optional[int] = None):
        if max_length:
            super().__init__(f"Length must be between {min_length} and {max_length}")
        else:
            super().__init__(f"Length must be at least {min_length}")
        self.min_length = min_length
        self.max_length = max_length
    
    def validate(self, value: Any) -> bool:
        """Validate value length."""
        if not hasattr(value, '__len__'):
            return False
        length = len(value)
        if length < self.min_length:
            return False
        if self.max_length and length > self.max_length:
            return False
        return True


class RegexValidator(BaseValidator):
    """Regex validator class."""
    
    def __init__(self, pattern: str, flags: int = 0):
        super().__init__(f"Value does not match pattern: {pattern}")
        self.pattern = re.compile(pattern, flags)
    
    def validate(self, value: Any) -> bool:
        """Validate value matches regex."""
        if not isinstance(value, str):
            return False
        return bool(self.pattern.match(value))


class CompositeValidator(BaseValidator):
    """Composite validator class (all validators must pass)."""
    
    def __init__(self, validators: List[BaseValidator]):
        super().__init__("Composite validation failed")
        self.validators = validators
    
    def validate(self, value: Any) -> bool:
        """Validate using all validators."""
        return all(validator.validate(value) for validator in self.validators)


