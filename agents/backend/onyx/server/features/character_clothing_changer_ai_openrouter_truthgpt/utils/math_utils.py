"""
Math Utilities
==============

Mathematical utilities.
"""

from typing import List, Optional


def clamp(value: float, min_value: float, max_value: float) -> float:
    """
    Clamp value between min and max.
    
    Args:
        value: Value to clamp
        min_value: Minimum value
        max_value: Maximum value
        
    Returns:
        Clamped value
    """
    return max(min_value, min(value, max_value))


def lerp(start: float, end: float, t: float) -> float:
    """
    Linear interpolation between two values.
    
    Args:
        start: Start value
        end: End value
        t: Interpolation factor (0-1)
        
    Returns:
        Interpolated value
    """
    return start + (end - start) * clamp(t, 0.0, 1.0)


def normalize(value: float, min_value: float, max_value: float) -> float:
    """
    Normalize value to 0-1 range.
    
    Args:
        value: Value to normalize
        min_value: Minimum value
        max_value: Maximum value
        
    Returns:
        Normalized value (0-1)
    """
    if max_value == min_value:
        return 0.0
    
    return (value - min_value) / (max_value - min_value)


def percentage(value: float, total: float) -> float:
    """
    Calculate percentage.
    
    Args:
        value: Value
        total: Total
        
    Returns:
        Percentage (0-100)
    """
    if total == 0:
        return 0.0
    
    return (value / total) * 100.0


def average(values: List[float]) -> float:
    """
    Calculate average of values.
    
    Args:
        values: List of values
        
    Returns:
        Average value
    """
    if not values:
        return 0.0
    
    return sum(values) / len(values)


def median(values: List[float]) -> float:
    """
    Calculate median of values.
    
    Args:
        values: List of values
        
    Returns:
        Median value
    """
    if not values:
        return 0.0
    
    sorted_values = sorted(values)
    n = len(sorted_values)
    
    if n % 2 == 0:
        return (sorted_values[n // 2 - 1] + sorted_values[n // 2]) / 2.0
    else:
        return sorted_values[n // 2]


def round_to(value: float, decimals: int = 2) -> float:
    """
    Round value to specified decimals.
    
    Args:
        value: Value to round
        decimals: Number of decimals
        
    Returns:
        Rounded value
    """
    return round(value, decimals)

