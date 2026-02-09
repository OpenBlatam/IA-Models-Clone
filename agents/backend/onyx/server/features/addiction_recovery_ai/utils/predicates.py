"""
Predicate functions
Pure functions that return boolean values
"""

from typing import Any, Callable


def is_none(value: Any) -> bool:
    """Check if value is None"""
    return value is None


def is_not_none(value: Any) -> bool:
    """Check if value is not None"""
    return value is not None


def is_empty(value: Any) -> bool:
    """Check if value is empty"""
    if value is None:
        return True
    
    if isinstance(value, (str, list, dict, tuple)):
        return len(value) == 0
    
    return False


def is_not_empty(value: Any) -> bool:
    """Check if value is not empty"""
    return not is_empty(value)


def is_positive(value: float) -> bool:
    """Check if value is positive"""
    return value > 0


def is_non_negative(value: float) -> bool:
    """Check if value is non-negative"""
    return value >= 0


def is_negative(value: float) -> bool:
    """Check if value is negative"""
    return value < 0


def is_zero(value: float) -> bool:
    """Check if value is zero"""
    return value == 0


def is_in_range(value: float, min_value: float, max_value: float) -> bool:
    """Check if value is in range"""
    return min_value <= value <= max_value


def is_greater_than(value: float, threshold: float) -> bool:
    """Check if value is greater than threshold"""
    return value > threshold


def is_less_than(value: float, threshold: float) -> bool:
    """Check if value is less than threshold"""
    return value < threshold


def is_equal(value: Any, other: Any) -> bool:
    """Check if values are equal"""
    return value == other


def is_not_equal(value: Any, other: Any) -> bool:
    """Check if values are not equal"""
    return value != other


def is_in(value: Any, collection: list) -> bool:
    """Check if value is in collection"""
    return value in collection


def is_not_in(value: Any, collection: list) -> bool:
    """Check if value is not in collection"""
    return value not in collection


def all_true(predicates: list[Callable[[Any], bool]], value: Any) -> bool:
    """Check if all predicates return True"""
    return all(pred(value) for pred in predicates)


def any_true(predicates: list[Callable[[Any], bool]], value: Any) -> bool:
    """Check if any predicate returns True"""
    return any(pred(value) for pred in predicates)

