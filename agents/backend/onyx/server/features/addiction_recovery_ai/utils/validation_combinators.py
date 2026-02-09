"""
Validation combinators
Combinators for building complex validators
"""

from typing import Callable, List, Any, TypeVar
from utils.predicates import all_true, any_true

T = TypeVar('T')


def combine_validators(*validators: Callable[[T], bool]) -> Callable[[T], bool]:
    """
    Combine multiple validators with AND logic
    
    Args:
        *validators: Validator functions
    
    Returns:
        Combined validator
    """
    def combined(value: T) -> bool:
        return all(validator(value) for validator in validators)
    
    return combined


def combine_validators_or(*validators: Callable[[T], bool]) -> Callable[[T], bool]:
    """
    Combine multiple validators with OR logic
    
    Args:
        *validators: Validator functions
    
    Returns:
        Combined validator
    """
    def combined(value: T) -> bool:
        return any(validator(value) for validator in validators)
    
    return combined


def validate_and_transform(
    validator: Callable[[T], bool],
    transformer: Callable[[T], Any],
    error_message: str = "Validation failed"
) -> Callable[[T], Any]:
    """
    Create validator that transforms value if valid
    
    Args:
        validator: Validation function
        transformer: Transformation function
        error_message: Error message if validation fails
    
    Returns:
        Function that validates and transforms
    """
    def validate_transform(value: T) -> Any:
        if not validator(value):
            raise ValueError(error_message)
        return transformer(value)
    
    return validate_transform


def chain_validators(*validators: Callable[[T], T]) -> Callable[[T], T]:
    """
    Chain validators that transform values
    
    Args:
        *validators: Validator/transformer functions
    
    Returns:
        Chained validator
    """
    def chained(value: T) -> T:
        result = value
        for validator in validators:
            result = validator(result)
        return result
    
    return chained

