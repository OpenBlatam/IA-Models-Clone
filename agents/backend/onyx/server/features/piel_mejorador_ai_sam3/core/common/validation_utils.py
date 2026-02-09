"""
Validation Utilities for Piel Mejorador AI SAM3
==============================================

Unified validation utilities.
"""

import logging
from typing import Any, Optional, Callable, List, Dict, Union
from pathlib import Path

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Custom validation error."""
    pass


class Validator:
    """Unified validator with common validation patterns."""
    
    @staticmethod
    def validate_not_none(value: Any, name: str = "value") -> None:
        """Validate value is not None."""
        if value is None:
            raise ValidationError(f"{name} cannot be None")
    
    @staticmethod
    def validate_not_empty(value: Any, name: str = "value") -> None:
        """Validate value is not empty."""
        if not value:
            raise ValidationError(f"{name} cannot be empty")
    
    @staticmethod
    def validate_type(value: Any, expected_type: type, name: str = "value") -> None:
        """Validate value type."""
        if not isinstance(value, expected_type):
            raise ValidationError(
                f"{name} must be of type {expected_type.__name__}, "
                f"got {type(value).__name__}"
            )
    
    @staticmethod
    def validate_range(
        value: float,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        name: str = "value"
    ) -> None:
        """Validate numeric range."""
        if min_value is not None and value < min_value:
            raise ValidationError(f"{name} must be >= {min_value}, got {value}")
        
        if max_value is not None and value > max_value:
            raise ValidationError(f"{name} must be <= {max_value}, got {value}")
    
    @staticmethod
    def validate_in(
        value: Any,
        allowed_values: List[Any],
        name: str = "value"
    ) -> None:
        """Validate value is in allowed list."""
        if value not in allowed_values:
            raise ValidationError(
                f"{name} must be one of {allowed_values}, got {value}"
            )
    
    @staticmethod
    def validate_file_exists(
        file_path: Union[str, Path],
        name: str = "file"
    ) -> None:
        """Validate file exists."""
        path = Path(file_path)
        if not path.exists():
            raise ValidationError(f"{name} does not exist: {file_path}")
        
        if not path.is_file():
            raise ValidationError(f"{name} is not a file: {file_path}")
    
    @staticmethod
    def validate_file_size(
        file_path: Union[str, Path],
        max_size_mb: float,
        name: str = "file"
    ) -> None:
        """Validate file size."""
        path = Path(file_path)
        if not path.exists():
            return  # Existence checked separately
        
        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb > max_size_mb:
            raise ValidationError(
                f"{name} size ({size_mb:.2f}MB) exceeds maximum ({max_size_mb}MB)"
            )
    
    @staticmethod
    def validate_path_safe(
        file_path: Union[str, Path],
        name: str = "path"
    ) -> None:
        """Validate path is safe (no path traversal)."""
        path = Path(file_path).resolve()
        
        # Check for path traversal attempts
        if '..' in str(path):
            raise ValidationError(f"{name} contains path traversal: {file_path}")
        
        # Check for absolute paths in sensitive contexts
        if path.is_absolute():
            # This is a basic check - adjust based on requirements
            pass


def validate_with(
    validators: List[Callable[[Any], None]],
    value: Any,
    name: str = "value"
) -> None:
    """
    Apply multiple validators to a value.
    
    Args:
        validators: List of validator functions
        value: Value to validate
        name: Name for error messages
        
    Raises:
        ValidationError: If validation fails
    """
    for validator in validators:
        validator(value, name)


def validate_dict(
    data: Dict[str, Any],
    schema: Dict[str, List[Callable[[Any], None]]]
) -> List[str]:
    """
    Validate dictionary against schema.
    
    Args:
        data: Dictionary to validate
        schema: Schema mapping keys to validators
        
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    
    for key, validators in schema.items():
        if key not in data:
            errors.append(f"Missing required key: {key}")
            continue
        
        try:
            validate_with(validators, data[key], key)
        except ValidationError as e:
            errors.append(str(e))
    
    return errors

