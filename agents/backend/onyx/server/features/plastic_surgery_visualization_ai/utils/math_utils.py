"""Math utilities."""

from typing import List, Optional
import math


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
        t: Interpolation factor (0.0 to 1.0)
        
    Returns:
        Interpolated value
    """
    return start + (end - start) * clamp(t, 0.0, 1.0)


def normalize(value: float, min_val: float, max_val: float) -> float:
    """
    Normalize value to 0-1 range.
    
    Args:
        value: Value to normalize
        min_val: Minimum value of range
        max_val: Maximum value of range
        
    Returns:
        Normalized value (0.0 to 1.0)
    """
    if max_val == min_val:
        return 0.0
    return (value - min_val) / (max_val - min_val)


def denormalize(normalized: float, min_val: float, max_val: float) -> float:
    """
    Denormalize value from 0-1 range.
    
    Args:
        normalized: Normalized value (0.0 to 1.0)
        min_val: Minimum value of range
        max_val: Maximum value of range
        
    Returns:
        Denormalized value
    """
    return min_val + (max_val - min_val) * clamp(normalized, 0.0, 1.0)


def round_to(value: float, decimals: int = 2) -> float:
    """
    Round value to specified decimal places.
    
    Args:
        value: Value to round
        decimals: Number of decimal places
        
    Returns:
        Rounded value
    """
    return round(value, decimals)


def calculate_percentage(part: float, total: float) -> float:
    """
    Calculate percentage.
    
    Args:
        part: Part value
        total: Total value
        
    Returns:
        Percentage (0-100)
    """
    if total == 0:
        return 0.0
    return (part / total) * 100


def calculate_average(values: List[float]) -> float:
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


def calculate_median(values: List[float]) -> float:
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
        return (sorted_values[n // 2 - 1] + sorted_values[n // 2]) / 2
    else:
        return sorted_values[n // 2]


def calculate_standard_deviation(values: List[float]) -> float:
    """
    Calculate standard deviation.
    
    Args:
        values: List of values
        
    Returns:
        Standard deviation
    """
    if not values:
        return 0.0
    
    avg = calculate_average(values)
    variance = sum((x - avg) ** 2 for x in values) / len(values)
    return math.sqrt(variance)


def is_between(value: float, min_val: float, max_val: float, inclusive: bool = True) -> bool:
    """
    Check if value is between min and max.
    
    Args:
        value: Value to check
        min_val: Minimum value
        max_val: Maximum value
        inclusive: Include boundaries
        
    Returns:
        True if value is in range
    """
    if inclusive:
        return min_val <= value <= max_val
    else:
        return min_val < value < max_val

