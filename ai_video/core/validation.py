from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

# Constants
BUFFER_SIZE = 1024

import re
import json
from typing import Any, Dict, List, Optional, Union, Callable, Type, Tuple
from dataclasses import dataclass, field
from datetime import datetime, date
from pathlib import Path
import logging
from enum import Enum
import functools
            from urllib.parse import urlparse
from typing import Any, List, Dict, Optional
import asyncio
"""
AI Video System - Validation Module

Production-ready validation utilities including schema validation,
data validation, type checking, and validation decorators.
"""


logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Custom validation error with detailed information."""
    
    def __init__(self, message: str, field: Optional[str] = None, value: Any = None):
        
    """__init__ function."""
self.message = message
        self.field = field
        self.value = value
        super().__init__(self.message)


@dataclass
class ValidationResult:
    """Result of a validation operation."""
    is_valid: bool
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    data: Optional[Any] = None


class DataType(Enum):
    """Supported data types for validation."""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"
    DATE = "date"
    DATETIME = "datetime"
    EMAIL = "email"
    URL = "url"
    FILE_PATH = "file_path"
    JSON = "json"


@dataclass
class FieldSchema:
    """Schema definition for a field."""
    name: str
    data_type: DataType
    required: bool = True
    default: Optional[Any] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    pattern: Optional[str] = None
    allowed_values: Optional[List[Any]] = None
    custom_validator: Optional[Callable[[Any], Tuple[bool, str]]] = None
    description: Optional[str] = None


class SchemaValidator:
    """
    Validates data against defined schemas.
    
    Features:
    - Schema definition
    - Type validation
    - Constraint validation
    - Custom validators
    - Nested validation
    """
    
    def __init__(self) -> Any:
        self.schemas: Dict[str, Dict[str, FieldSchema]] = {}
        self.validators: Dict[str, Callable] = {}
    
    def register_schema(
        self,
        name: str,
        fields: List[FieldSchema]
    ) -> None:
        """Register a schema for validation."""
        self.schemas[name] = {field.name: field for field in fields}
    
    def validate_data(
        self,
        schema_name: str,
        data: Dict[str, Any]
    ) -> ValidationResult:
        """Validate data against a registered schema."""
        if schema_name not in self.schemas:
            return ValidationResult(
                is_valid=False,
                errors=[ValidationError(f"Schema '{schema_name}' not found")]
            )
        
        schema = self.schemas[schema_name]
        result = ValidationResult(is_valid=True, data=data.copy())
        
        # Validate each field
        for field_name, field_schema in schema.items():
            field_value = data.get(field_name, field_schema.default)
            
            # Check if required field is missing
            if field_schema.required and field_value is None:
                result.errors.append(ValidationError(
                    f"Required field '{field_name}' is missing",
                    field=field_name
                ))
                result.is_valid = False
                continue
            
            # Skip validation if field is not provided and not required
            if field_value is None:
                continue
            
            # Validate field
            field_result = self._validate_field(field_schema, field_value, field_name)
            
            if not field_result.is_valid:
                result.errors.extend(field_result.errors)
                result.is_valid = False
            
            result.warnings.extend(field_result.warnings)
            
            # Update data with validated value
            result.data[field_name] = field_result.data
        
        return result
    
    def _validate_field(
        self,
        field_schema: FieldSchema,
        value: Any,
        field_name: str
    ) -> ValidationResult:
        """Validate a single field."""
        result = ValidationResult(is_valid=True, data=value)
        
        # Type validation
        type_result = self._validate_type(field_schema.data_type, value, field_name)
        if not type_result.is_valid:
            result.errors.extend(type_result.errors)
            result.is_valid = False
            return result
        
        validated_value = type_result.data
        
        # Length validation for strings and arrays
        if field_schema.data_type in [DataType.STRING, DataType.ARRAY]:
            length_result = self._validate_length(
                field_schema, validated_value, field_name
            )
            if not length_result.is_valid:
                result.errors.extend(length_result.errors)
                result.is_valid = False
        
        # Value range validation for numbers
        if field_schema.data_type in [DataType.INTEGER, DataType.FLOAT]:
            range_result = self._validate_range(
                field_schema, validated_value, field_name
            )
            if not range_result.is_valid:
                result.errors.extend(range_result.errors)
                result.is_valid = False
        
        # Pattern validation for strings
        if field_schema.data_type == DataType.STRING and field_schema.pattern:
            pattern_result = self._validate_pattern(
                field_schema.pattern, validated_value, field_name
            )
            if not pattern_result.is_valid:
                result.errors.extend(pattern_result.errors)
                result.is_valid = False
        
        # Allowed values validation
        if field_schema.allowed_values is not None:
            allowed_result = self._validate_allowed_values(
                field_schema.allowed_values, validated_value, field_name
            )
            if not allowed_result.is_valid:
                result.errors.extend(allowed_result.errors)
                result.is_valid = False
        
        # Custom validation
        if field_schema.custom_validator:
            custom_result = self._validate_custom(
                field_schema.custom_validator, validated_value, field_name
            )
            if not custom_result.is_valid:
                result.errors.extend(custom_result.errors)
                result.is_valid = False
        
        result.data = validated_value
        return result
    
    def _validate_type(
        self,
        data_type: DataType,
        value: Any,
        field_name: str
    ) -> ValidationResult:
        """Validate data type."""
        result = ValidationResult(is_valid=True, data=value)
        
        try:
            if data_type == DataType.STRING:
                if not isinstance(value, str):
                    result.errors.append(ValidationError(
                        f"Field '{field_name}' must be a string",
                        field=field_name,
                        value=value
                    ))
                    result.is_valid = False
            
            elif data_type == DataType.INTEGER:
                if isinstance(value, str):
                    try:
                        result.data = int(value)
                    except ValueError:
                        result.errors.append(ValidationError(
                            f"Field '{field_name}' must be a valid integer",
                            field=field_name,
                            value=value
                        ))
                        result.is_valid = False
                elif not isinstance(value, int):
                    result.errors.append(ValidationError(
                        f"Field '{field_name}' must be an integer",
                        field=field_name,
                        value=value
                    ))
                    result.is_valid = False
            
            elif data_type == DataType.FLOAT:
                if isinstance(value, str):
                    try:
                        result.data = float(value)
                    except ValueError:
                        result.errors.append(ValidationError(
                            f"Field '{field_name}' must be a valid number",
                            field=field_name,
                            value=value
                        ))
                        result.is_valid = False
                elif not isinstance(value, (int, float)):
                    result.errors.append(ValidationError(
                        f"Field '{field_name}' must be a number",
                        field=field_name,
                        value=value
                    ))
                    result.is_valid = False
            
            elif data_type == DataType.BOOLEAN:
                if isinstance(value, str):
                    if value.lower() in ['true', '1', 'yes', 'on']:
                        result.data = True
                    elif value.lower() in ['false', '0', 'no', 'off']:
                        result.data = False
                    else:
                        result.errors.append(ValidationError(
                            f"Field '{field_name}' must be a valid boolean",
                            field=field_name,
                            value=value
                        ))
                        result.is_valid = False
                elif not isinstance(value, bool):
                    result.errors.append(ValidationError(
                        f"Field '{field_name}' must be a boolean",
                        field=field_name,
                        value=value
                    ))
                    result.is_valid = False
            
            elif data_type == DataType.ARRAY:
                if not isinstance(value, list):
                    result.errors.append(ValidationError(
                        f"Field '{field_name}' must be an array",
                        field=field_name,
                        value=value
                    ))
                    result.is_valid = False
            
            elif data_type == DataType.OBJECT:
                if not isinstance(value, dict):
                    result.errors.append(ValidationError(
                        f"Field '{field_name}' must be an object",
                        field=field_name,
                        value=value
                    ))
                    result.is_valid = False
            
            elif data_type == DataType.DATE:
                if isinstance(value, str):
                    try:
                        result.data = datetime.strptime(value, '%Y-%m-%d').date()
                    except ValueError:
                        result.errors.append(ValidationError(
                            f"Field '{field_name}' must be a valid date (YYYY-MM-DD)",
                            field=field_name,
                            value=value
                        ))
                        result.is_valid = False
                elif not isinstance(value, date):
                    result.errors.append(ValidationError(
                        f"Field '{field_name}' must be a date",
                        field=field_name,
                        value=value
                    ))
                    result.is_valid = False
            
            elif data_type == DataType.DATETIME:
                if isinstance(value, str):
                    try:
                        result.data = datetime.fromisoformat(value.replace('Z', '+00:00'))
                    except ValueError:
                        result.errors.append(ValidationError(
                            f"Field '{field_name}' must be a valid datetime (ISO format)",
                            field=field_name,
                            value=value
                        ))
                        result.is_valid = False
                elif not isinstance(value, datetime):
                    result.errors.append(ValidationError(
                        f"Field '{field_name}' must be a datetime",
                        field=field_name,
                        value=value
                    ))
                    result.is_valid = False
            
            elif data_type == DataType.EMAIL:
                if not isinstance(value, str):
                    result.errors.append(ValidationError(
                        f"Field '{field_name}' must be a string",
                        field=field_name,
                        value=value
                    ))
                    result.is_valid = False
                elif not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', value):
                    result.errors.append(ValidationError(
                        f"Field '{field_name}' must be a valid email address",
                        field=field_name,
                        value=value
                    ))
                    result.is_valid = False
            
            elif data_type == DataType.URL:
                if not isinstance(value, str):
                    result.errors.append(ValidationError(
                        f"Field '{field_name}' must be a string",
                        field=field_name,
                        value=value
                    ))
                    result.is_valid = False
                elif not re.match(r'^https?://[^\s/$.?#].[^\s]*$', value):
                    result.errors.append(ValidationError(
                        f"Field '{field_name}' must be a valid URL",
                        field=field_name,
                        value=value
                    ))
                    result.is_valid = False
            
            elif data_type == DataType.FILE_PATH:
                if not isinstance(value, str):
                    result.errors.append(ValidationError(
                        f"Field '{field_name}' must be a string",
                        field=field_name,
                        value=value
                    ))
                    result.is_valid = False
                else:
                    path = Path(value)
                    if not path.exists():
                        result.warnings.append(f"File path '{value}' does not exist")
            
            elif data_type == DataType.JSON:
                if isinstance(value, str):
                    try:
                        result.data = json.loads(value)
                    except json.JSONDecodeError:
                        result.errors.append(ValidationError(
                            f"Field '{field_name}' must be valid JSON",
                            field=field_name,
                            value=value
                        ))
                        result.is_valid = False
                elif not isinstance(value, (dict, list)):
                    result.errors.append(ValidationError(
                        f"Field '{field_name}' must be a JSON object or array",
                        field=field_name,
                        value=value
                    ))
                    result.is_valid = False
        
        except Exception as e:
            result.errors.append(ValidationError(
                f"Type validation error for field '{field_name}': {str(e)}",
                field=field_name,
                value=value
            ))
            result.is_valid = False
        
        return result
    
    def _validate_length(
        self,
        field_schema: FieldSchema,
        value: Any,
        field_name: str
    ) -> ValidationResult:
        """Validate length constraints."""
        result = ValidationResult(is_valid=True, data=value)
        
        if field_schema.data_type == DataType.STRING:
            length = len(value)
        elif field_schema.data_type == DataType.ARRAY:
            length = len(value)
        else:
            return result
        
        if field_schema.min_length is not None and length < field_schema.min_length:
            result.errors.append(ValidationError(
                f"Field '{field_name}' must be at least {field_schema.min_length} characters long",
                field=field_name,
                value=value
            ))
            result.is_valid = False
        
        if field_schema.max_length is not None and length > field_schema.max_length:
            result.errors.append(ValidationError(
                f"Field '{field_name}' must be at most {field_schema.max_length} characters long",
                field=field_name,
                value=value
            ))
            result.is_valid = False
        
        return result
    
    def _validate_range(
        self,
        field_schema: FieldSchema,
        value: Union[int, float],
        field_name: str
    ) -> ValidationResult:
        """Validate numeric range constraints."""
        result = ValidationResult(is_valid=True, data=value)
        
        if field_schema.min_value is not None and value < field_schema.min_value:
            result.errors.append(ValidationError(
                f"Field '{field_name}' must be at least {field_schema.min_value}",
                field=field_name,
                value=value
            ))
            result.is_valid = False
        
        if field_schema.max_value is not None and value > field_schema.max_value:
            result.errors.append(ValidationError(
                f"Field '{field_name}' must be at most {field_schema.max_value}",
                field=field_name,
                value=value
            ))
            result.is_valid = False
        
        return result
    
    def _validate_pattern(
        self,
        pattern: str,
        value: str,
        field_name: str
    ) -> ValidationResult:
        """Validate string pattern."""
        result = ValidationResult(is_valid=True, data=value)
        
        if not re.match(pattern, value):
            result.errors.append(ValidationError(
                f"Field '{field_name}' does not match required pattern",
                field=field_name,
                value=value
            ))
            result.is_valid = False
        
        return result
    
    def _validate_allowed_values(
        self,
        allowed_values: List[Any],
        value: Any,
        field_name: str
    ) -> ValidationResult:
        """Validate against allowed values."""
        result = ValidationResult(is_valid=True, data=value)
        
        if value not in allowed_values:
            result.errors.append(ValidationError(
                f"Field '{field_name}' must be one of: {allowed_values}",
                field=field_name,
                value=value
            ))
            result.is_valid = False
        
        return result
    
    def _validate_custom(
        self,
        validator: Callable[[Any], Tuple[bool, str]],
        value: Any,
        field_name: str
    ) -> ValidationResult:
        """Validate using custom validator."""
        result = ValidationResult(is_valid=True, data=value)
        
        try:
            is_valid, message = validator(value)
            if not is_valid:
                result.errors.append(ValidationError(
                    f"Field '{field_name}': {message}",
                    field=field_name,
                    value=value
                ))
                result.is_valid = False
        except Exception as e:
            result.errors.append(ValidationError(
                f"Custom validation error for field '{field_name}': {str(e)}",
                field=field_name,
                value=value
            ))
            result.is_valid = False
        
        return result


class DataValidator:
    """
    Validates data structures and content.
    
    Features:
    - Data structure validation
    - Content validation
    - File validation
    - JSON validation
    """
    
    def __init__(self) -> Any:
        self.validators: Dict[str, Callable] = {}
    
    def validate_json_structure(
        self,
        data: Any,
        required_keys: Optional[List[str]] = None,
        optional_keys: Optional[List[str]] = None,
        allowed_types: Optional[Dict[str, Type]] = None
    ) -> ValidationResult:
        """Validate JSON structure."""
        result = ValidationResult(is_valid=True, data=data)
        
        if not isinstance(data, dict):
            result.errors.append(ValidationError(
                "Data must be a JSON object",
                value=data
            ))
            result.is_valid = False
            return result
        
        # Check required keys
        if required_keys:
            for key in required_keys:
                if key not in data:
                    result.errors.append(ValidationError(
                        f"Required key '{key}' is missing",
                        field=key
                    ))
                    result.is_valid = False
        
        # Check for unexpected keys
        if required_keys or optional_keys:
            allowed_keys = set(required_keys or []) | set(optional_keys or [])
            unexpected_keys = set(data.keys()) - allowed_keys
            if unexpected_keys:
                result.warnings.append(
                    f"Unexpected keys found: {list(unexpected_keys)}"
                )
        
        # Check types
        if allowed_types:
            for key, expected_type in allowed_types.items():
                if key in data:
                    if not isinstance(data[key], expected_type):
                        result.errors.append(ValidationError(
                            f"Key '{key}' must be of type {expected_type.__name__}",
                            field=key,
                            value=data[key]
                        ))
                        result.is_valid = False
        
        return result
    
    def validate_file_content(
        self,
        file_path: str,
        allowed_extensions: Optional[List[str]] = None,
        max_size_mb: Optional[int] = None,
        content_validator: Optional[Callable[[bytes], bool]] = None
    ) -> ValidationResult:
        """Validate file content."""
        result = ValidationResult(is_valid=True)
        
        try:
            path = Path(file_path)
            
            # Check if file exists
            if not path.exists():
                result.errors.append(ValidationError(
                    f"File '{file_path}' does not exist",
                    value=file_path
                ))
                result.is_valid = False
                return result
            
            # Check file extension
            if allowed_extensions:
                if path.suffix.lower() not in allowed_extensions:
                    result.errors.append(ValidationError(
                        f"File extension '{path.suffix}' not allowed. Allowed: {allowed_extensions}",
                        value=file_path
                    ))
                    result.is_valid = False
            
            # Check file size
            if max_size_mb:
                file_size_mb = path.stat().st_size / (1024 * 1024)
                if file_size_mb > max_size_mb:
                    result.errors.append(ValidationError(
                        f"File size {file_size_mb:.2f}MB exceeds maximum {max_size_mb}MB",
                        value=file_path
                    ))
                    result.is_valid = False
            
            # Validate content
            if content_validator:
                try:
                    with open(path, 'rb') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                        content = f.read()
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                    
                    if not content_validator(content):
                        result.errors.append(ValidationError(
                            "File content validation failed",
                            value=file_path
                        ))
                        result.is_valid = False
                except Exception as e:
                    result.errors.append(ValidationError(
                        f"Error reading file: {str(e)}",
                        value=file_path
                    ))
                    result.is_valid = False
        
        except Exception as e:
            result.errors.append(ValidationError(
                f"File validation error: {str(e)}",
                value=file_path
            ))
            result.is_valid = False
        
        return result
    
    def validate_url_content(
        self,
        url: str,
        allowed_domains: Optional[List[str]] = None,
        content_validator: Optional[Callable[[str], bool]] = None
    ) -> ValidationResult:
        """Validate URL content."""
        result = ValidationResult(is_valid=True)
        
        # Basic URL validation
        if not re.match(r'^https?://[^\s/$.?#].[^\s]*$', url):
            result.errors.append(ValidationError(
                "Invalid URL format",
                value=url
            ))
            result.is_valid = False
            return result
        
        # Check domain
        if allowed_domains:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            
            if not any(allowed in domain for allowed in allowed_domains):
                result.errors.append(ValidationError(
                    f"Domain not allowed. Allowed domains: {allowed_domains}",
                    value=url
                ))
                result.is_valid = False
        
        # Content validation (placeholder)
        if content_validator:
            # This would typically involve fetching the URL content
            # For now, we'll just note that content validation is requested
            result.warnings.append("Content validation requested but not implemented")
        
        return result


# Validation decorators
def validate_schema(schema_name: str):
    """Decorator to validate function arguments against a schema."""
    def decorator(func) -> Any:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            # Validate kwargs against schema
            validator = SchemaValidator()
            result = validator.validate_data(schema_name, kwargs)
            
            if not result.is_valid:
                error_messages = [f"{e.field}: {e.message}" for e in result.errors]
                raise ValidationError(f"Schema validation failed: {'; '.join(error_messages)}")
            
            return await func(*args, **result.data)
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            # Validate kwargs against schema
            validator = SchemaValidator()
            result = validator.validate_data(schema_name, kwargs)
            
            if not result.is_valid:
                error_messages = [f"{e.field}: {e.message}" for e in result.errors]
                raise ValidationError(f"Schema validation failed: {'; '.join(error_messages)}")
            
            return func(*args, **result.data)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def validate_input(validation_func: Callable[[Any], ValidationResult]):
    """Decorator to validate function input using a custom validation function."""
    def decorator(func) -> Any:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            # Validate input
            result = validation_func(kwargs)
            
            if not result.is_valid:
                error_messages = [e.message for e in result.errors]
                raise ValidationError(f"Input validation failed: {'; '.join(error_messages)}")
            
            return await func(*args, **kwargs)
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            # Validate input
            result = validation_func(kwargs)
            
            if not result.is_valid:
                error_messages = [e.message for e in result.errors]
                raise ValidationError(f"Input validation failed: {'; '.join(error_messages)}")
            
            return func(*args, **kwargs)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


# Global validator instances
schema_validator = SchemaValidator()
data_validator = DataValidator()

# Predefined schemas
VIDEO_GENERATION_SCHEMA = [
    FieldSchema("input_text", DataType.STRING, required=True, max_length=10000),
    FieldSchema("output_format", DataType.STRING, required=False, default="mp4",
                allowed_values=["mp4", "avi", "mov", "webm"]),
    FieldSchema("duration", DataType.INTEGER, required=False, default=30,
                min_value=1, max_value=600),
    FieldSchema("quality", DataType.STRING, required=False, default="medium",
                allowed_values=["low", "medium", "high", "ultra"]),
    FieldSchema("plugins", DataType.ARRAY, required=False, default=[]),
    FieldSchema("metadata", DataType.OBJECT, required=False, default={})
]

PLUGIN_CONFIG_SCHEMA = [
    FieldSchema("name", DataType.STRING, required=True, max_length=100),
    FieldSchema("version", DataType.STRING, required=True, pattern=r'^\d+\.\d+\.\d+$'),
    FieldSchema("enabled", DataType.BOOLEAN, required=False, default=True),
    FieldSchema("config", DataType.OBJECT, required=False, default={}),
    FieldSchema("priority", DataType.INTEGER, required=False, default=0,
                min_value=0, max_value=100)
]

# Register schemas
schema_validator.register_schema("video_generation", VIDEO_GENERATION_SCHEMA)
schema_validator.register_schema("plugin_config", PLUGIN_CONFIG_SCHEMA)


# Utility functions
def validate_email(email: str) -> bool:
    """Validate email address format."""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))


def validate_url(url: str) -> bool:
    """Validate URL format."""
    pattern = r'^https?://[^\s/$.?#].[^\s]*$'
    return bool(re.match(pattern, url))


def validate_phone(phone: str) -> bool:
    """Validate phone number format."""
    pattern = r'^\+?[\d\s\-\(\)]{10,20}$'
    return bool(re.match(pattern, phone))


def validate_filename(filename: str) -> bool:
    """Validate filename for safe storage."""
    # Check for dangerous characters
    dangerous_chars = r'[<>:"/\\|?*]'
    if re.search(dangerous_chars, filename):
        return False
    
    # Check length
    if len(filename) > 255:
        return False
    
    # Check for reserved names
    reserved_names = ['CON', 'PRN', 'AUX', 'NUL'] + [f'COM{i}' for i in range(1, 10)] + [f'LPT{i}' for i in range(1, 10)]
    if filename.upper() in reserved_names:
        return False
    
    return True


def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe storage."""
    # Remove dangerous characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # Replace spaces with underscores
    filename = re.sub(r'\s+', '_', filename)
    
    # Remove leading/trailing dots and underscores
    filename = filename.strip('._')
    
    # Limit length
    if len(filename) > 255:
        name, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')
        filename = name[:255-len(ext)-1] + ('.' + ext if ext else '')
    
    return filename 