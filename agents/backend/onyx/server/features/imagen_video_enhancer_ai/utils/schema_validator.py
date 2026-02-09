"""
Schema Validator
================

Utilities for validating data against schemas.
"""

from typing import Any, Dict, List, Optional, Type
from typing import get_origin, get_args
import inspect


class SchemaValidator:
    """Validator for data schemas."""
    
    @staticmethod
    def validate_against_schema(
        data: Any,
        schema: Dict[str, Type],
        strict: bool = False
    ) -> tuple[bool, Optional[str]]:
        """
        Validate data against a schema.
        
        Args:
            data: Data to validate
            schema: Schema dictionary with field names and types
            strict: If True, reject extra fields
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not isinstance(data, dict):
            return False, "Data must be a dictionary"
        
        # Check for extra fields
        if strict:
            extra_fields = set(data.keys()) - set(schema.keys())
            if extra_fields:
                return False, f"Extra fields not allowed: {extra_fields}"
        
        # Validate each field
        for field_name, field_type in schema.items():
            if field_name not in data:
                return False, f"Missing required field: {field_name}"
            
            value = data[field_name]
            if not SchemaValidator._check_type(value, field_type):
                return False, f"Field '{field_name}' has invalid type: expected {field_type}, got {type(value)}"
        
        return True, None
    
    @staticmethod
    def _check_type(value: Any, expected_type: Type) -> bool:
        """Check if value matches expected type."""
        # Handle None
        if expected_type is None or (hasattr(expected_type, "__origin__") and None in get_args(expected_type)):
            return value is None
        
        # Handle Union types
        origin = get_origin(expected_type)
        if origin is type(None) or (hasattr(expected_type, "__origin__") and expected_type.__origin__ is type(None)):
            return value is None
        
        if origin:
            # Handle Optional
            args = get_args(expected_type)
            if len(args) == 2 and type(None) in args:
                non_none_type = args[0] if args[1] is type(None) else args[1]
                return value is None or SchemaValidator._check_type(value, non_none_type)
            
            # Handle List
            if origin is list:
                if not isinstance(value, list):
                    return False
                if args:
                    return all(SchemaValidator._check_type(item, args[0]) for item in value)
                return True
            
            # Handle Dict
            if origin is dict:
                if not isinstance(value, dict):
                    return False
                if args:
                    key_type, value_type = args[0], args[1]
                    return all(
                        SchemaValidator._check_type(k, key_type) and SchemaValidator._check_type(v, value_type)
                        for k, v in value.items()
                    )
                return True
        
        # Standard type check
        return isinstance(value, expected_type)
    
    @staticmethod
    def validate_request_schema(
        request_data: Dict[str, Any],
        required_fields: List[str],
        optional_fields: Optional[Dict[str, Type]] = None
    ) -> tuple[bool, Optional[str]]:
        """
        Validate request data schema.
        
        Args:
            request_data: Request data dictionary
            required_fields: List of required field names
            optional_fields: Optional dictionary of optional fields and types
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not isinstance(request_data, dict):
            return False, "Request data must be a dictionary"
        
        # Check required fields
        missing = [field for field in required_fields if field not in request_data]
        if missing:
            return False, f"Missing required fields: {missing}"
        
        # Validate optional fields if provided
        if optional_fields:
            for field_name, field_type in optional_fields.items():
                if field_name in request_data:
                    value = request_data[field_name]
                    if not SchemaValidator._check_type(value, field_type):
                        return False, f"Field '{field_name}' has invalid type"
        
        return True, None




