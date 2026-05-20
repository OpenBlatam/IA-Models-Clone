"""
Advanced Config Validator
=========================

Advanced configuration validation utilities.
"""

import re
from typing import Dict, Any, List, Callable, Optional
from pathlib import Path
from dataclasses import dataclass


@dataclass
class ValidationRule:
    """Validation rule."""
    field: str
    validator: Callable[[Any], bool]
    message: str
    required: bool = True


class AdvancedConfigValidator:
    """Advanced configuration validator."""
    
    @staticmethod
    def validate_string(value: Any, min_length: int = 0, max_length: Optional[int] = None, pattern: Optional[str] = None) -> bool:
        """
        Validate string value.
        
        Args:
            value: Value to validate
            min_length: Minimum length
            max_length: Maximum length
            pattern: Optional regex pattern
            
        Returns:
            True if valid
        """
        if not isinstance(value, str):
            return False
        
        if len(value) < min_length:
            return False
        
        if max_length and len(value) > max_length:
            return False
        
        if pattern and not re.match(pattern, value):
            return False
        
        return True
    
    @staticmethod
    def validate_number(value: Any, min_value: Optional[float] = None, max_value: Optional[float] = None) -> bool:
        """
        Validate number value.
        
        Args:
            value: Value to validate
            min_value: Minimum value
            max_value: Maximum value
            
        Returns:
            True if valid
        """
        try:
            num = float(value)
            if min_value is not None and num < min_value:
                return False
            if max_value is not None and num > max_value:
                return False
            return True
        except (ValueError, TypeError):
            return False
    
    @staticmethod
    def validate_path(value: Any, must_exist: bool = False, must_be_file: bool = False, must_be_dir: bool = False) -> bool:
        """
        Validate path value.
        
        Args:
            value: Value to validate
            must_exist: Path must exist
            must_be_file: Path must be a file
            must_be_dir: Path must be a directory
            
        Returns:
            True if valid
        """
        try:
            path = Path(value)
            
            if must_exist and not path.exists():
                return False
            
            if must_be_file and not path.is_file():
                return False
            
            if must_be_dir and not path.is_dir():
                return False
            
            return True
        except Exception:
            return False
    
    @staticmethod
    def validate_url(value: Any) -> bool:
        """
        Validate URL value.
        
        Args:
            value: Value to validate
            
        Returns:
            True if valid
        """
        if not isinstance(value, str):
            return False
        
        url_pattern = re.compile(
            r'^https?://'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
            r'localhost|'  # localhost...
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)
        
        return bool(url_pattern.match(value))
    
    @staticmethod
    def validate_email(value: Any) -> bool:
        """
        Validate email value.
        
        Args:
            value: Value to validate
            
        Returns:
            True if valid
        """
        if not isinstance(value, str):
            return False
        
        email_pattern = re.compile(
            r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        )
        
        return bool(email_pattern.match(value))
    
    @staticmethod
    def validate_config_with_rules(config: Dict[str, Any], rules: List[ValidationRule]) -> tuple[bool, List[str]]:
        """
        Validate configuration with rules.
        
        Args:
            config: Configuration dictionary
            rules: List of validation rules
            
        Returns:
            Tuple of (is_valid, errors)
        """
        errors = []
        
        for rule in rules:
            value = config.get(rule.field)
            
            # Check required
            if rule.required and value is None:
                errors.append(f"{rule.field}: Required field is missing")
                continue
            
            # Skip validation if not required and value is None
            if not rule.required and value is None:
                continue
            
            # Validate value
            if not rule.validator(value):
                errors.append(f"{rule.field}: {rule.message}")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_nested_config(config: Dict[str, Any], schema: Dict[str, Any]) -> tuple[bool, List[str]]:
        """
        Validate nested configuration against schema.
        
        Args:
            config: Configuration dictionary
            schema: Validation schema
            
        Returns:
            Tuple of (is_valid, errors)
        """
        errors = []
        
        def validate_field(path: str, value: Any, schema_item: Dict[str, Any]):
            """Recursively validate field."""
            # Check required
            if schema_item.get("required", False) and value is None:
                errors.append(f"{path}: Required field is missing")
                return
            
            if value is None:
                return
            
            # Check type
            expected_type = schema_item.get("type")
            if expected_type and not isinstance(value, expected_type):
                errors.append(f"{path}: Expected type {expected_type.__name__}, got {type(value).__name__}")
                return
            
            # Check validator
            validator = schema_item.get("validator")
            if validator and not validator(value):
                errors.append(f"{path}: Validation failed")
                return
            
            # Recursively validate nested dicts
            if isinstance(value, dict) and "fields" in schema_item:
                for field_name, field_schema in schema_item["fields"].items():
                    validate_field(f"{path}.{field_name}", value.get(field_name), field_schema)
        
        # Validate top-level fields
        for field_name, field_schema in schema.items():
            validate_field(field_name, config.get(field_name), field_schema)
        
        return len(errors) == 0, errors




