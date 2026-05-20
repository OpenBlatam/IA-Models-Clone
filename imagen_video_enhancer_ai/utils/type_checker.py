"""
Type Checker Utilities
======================

Utilities for runtime type checking and validation.
"""

from typing import Any, Type, Union, get_origin, get_args
import inspect


class TypeChecker:
    """Runtime type checking utilities."""
    
    @staticmethod
    def check_type(value: Any, expected_type: Type) -> bool:
        """
        Check if value matches expected type.
        
        Args:
            value: Value to check
            expected_type: Expected type
            
        Returns:
            True if value matches type
        """
        # Handle None
        if expected_type is None or (hasattr(expected_type, "__origin__") and None in get_args(expected_type)):
            return value is None
        
        # Handle Union types
        origin = get_origin(expected_type)
        if origin is Union:
            args = get_args(expected_type)
            return any(TypeChecker.check_type(value, arg) for arg in args)
        
        # Handle Optional
        if hasattr(expected_type, "__origin__") and expected_type.__origin__ is Union:
            args = get_args(expected_type)
            if len(args) == 2 and type(None) in args:
                non_none_type = args[0] if args[1] is type(None) else args[1]
                return value is None or TypeChecker.check_type(value, non_none_type)
        
        # Handle generic types
        if origin:
            # For List, Dict, etc.
            if origin is list:
                if not isinstance(value, list):
                    return False
                args = get_args(expected_type)
                if args:
                    return all(TypeChecker.check_type(item, args[0]) for item in value)
                return True
            
            if origin is dict:
                if not isinstance(value, dict):
                    return False
                args = get_args(expected_type)
                if args:
                    key_type, value_type = args[0], args[1]
                    return all(
                        TypeChecker.check_type(k, key_type) and TypeChecker.check_type(v, value_type)
                        for k, v in value.items()
                    )
                return True
        
        # Standard type check
        return isinstance(value, expected_type)
    
    @staticmethod
    def validate_function_args(func: callable, *args, **kwargs) -> tuple[bool, Optional[str]]:
        """
        Validate function arguments against type hints.
        
        Args:
            func: Function to validate
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            sig = inspect.signature(func)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            
            # Check each parameter
            for param_name, param_value in bound.arguments.items():
                param = sig.parameters[param_name]
                annotation = param.annotation
                
                if annotation != inspect.Parameter.empty:
                    if not TypeChecker.check_type(param_value, annotation):
                        return False, f"Parameter '{param_name}' type mismatch: expected {annotation}, got {type(param_value)}"
            
            return True, None
        except Exception as e:
            return False, str(e)
    
    @staticmethod
    def get_type_name(type_obj: Type) -> str:
        """
        Get human-readable type name.
        
        Args:
            type_obj: Type object
            
        Returns:
            Type name string
        """
        origin = get_origin(type_obj)
        if origin:
            args = get_args(type_obj)
            args_str = ", ".join(TypeChecker.get_type_name(arg) for arg in args)
            return f"{origin.__name__}[{args_str}]"
        
        if hasattr(type_obj, "__name__"):
            return type_obj.__name__
        
        return str(type_obj)




