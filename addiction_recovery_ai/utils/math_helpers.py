"""
Mathematical utility functions
"""

from typing import List


def calculate_percentage(part: float, total: float) -> float:
    """
    Calculate percentage
    
    Args:
        part: Part value
        total: Total value
    
    Returns:
        Percentage (0-100)
    """
    if total == 0:
        return 0.0
    
    return (part / total) * 100.0


def calculate_average(values: List[float]) -> float:
    """
    Calculate average of values
    
    Args:
        values: List of numeric values
    
    Returns:
        Average value
    """
    if not values:
        return 0.0
    
    return sum(values) / len(values)


def calculate_median(values: List[float]) -> float:
    """
    Calculate median of values
    
    Args:
        values: List of numeric values
    
    Returns:
        Median value
    """
    if not values:
        return 0.0
    
    sorted_values = sorted(values)
    length = len(sorted_values)
    
    if length % 2 == 0:
        mid = length // 2
        return (sorted_values[mid - 1] + sorted_values[mid]) / 2.0
    
    return sorted_values[length // 2]


def clamp_value(value: float, min_value: float, max_value: float) -> float:
    """
    Clamp value between min and max
    
    Args:
        value: Value to clamp
        min_value: Minimum allowed value
        max_value: Maximum allowed value
    
    Returns:
        Clamped value
    """
    return max(min_value, min(max_value, value))


def round_to_decimals(value: float, decimals: int = 2) -> float:
    """
    Round value to specified decimal places
    
    Args:
        value: Value to round
        decimals: Number of decimal places
    
    Returns:
        Rounded value
    """
    return round(value, decimals)

