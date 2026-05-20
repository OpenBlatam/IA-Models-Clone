"""
Math Helper Functions

Helper functions for common mathematical operations and value constraints
that appear throughout the codebase.
"""

from typing import Union


def ensure_non_negative(value: Union[int, float], default: Union[int, float] = 0) -> Union[int, float]:
    """
    Ensure a value is non-negative, returning default if negative.
    
    This helper encapsulates the common pattern of `max(0, value)` that appears
    repeatedly in count calculations and score updates.
    
    Args:
        value: Value to ensure is non-negative
        default: Default value to return if value is negative (default: 0)
        
    Returns:
        Value if >= 0, otherwise default
        
    Example:
        >>> vote_count = ensure_non_negative(chat.vote_count + increment)
        >>> score = ensure_non_negative(calculated_score, default=0.0)
    """
    return max(default, value)


def safe_increment(
    current_value: Union[int, float],
    increment: Union[int, float],
    min_value: Union[int, float] = 0
) -> Union[int, float]:
    """
    Safely increment a value while ensuring it doesn't go below minimum.
    
    This helper encapsulates the common pattern of incrementing counts
    while ensuring they remain non-negative.
    
    Args:
        current_value: Current value to increment
        increment: Amount to increment (can be negative)
        min_value: Minimum allowed value (default: 0)
        
    Returns:
        Incremented value, clamped to minimum
        
    Example:
        >>> new_vote_count = safe_increment(chat.vote_count, vote_increment)
        >>> new_score = safe_increment(chat.score, score_increment, min_value=0.0)
    """
    return ensure_non_negative(current_value + increment, min_value)


def clamp_value(
    value: Union[int, float],
    min_value: Union[int, float],
    max_value: Union[int, float]
) -> Union[int, float]:
    """
    Clamp a value between minimum and maximum bounds.
    
    This helper encapsulates the common pattern of constraining values
    to a specific range.
    
    Args:
        value: Value to clamp
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        
    Returns:
        Clamped value
        
    Raises:
        ValueError: If min_value > max_value
        
    Example:
        >>> page = clamp_value(requested_page, MIN_PAGE, MAX_PAGE)
        >>> score = clamp_value(calculated_score, 0.0, 100.0)
    """
    if min_value > max_value:
        raise ValueError(f"min_value ({min_value}) must be <= max_value ({max_value})")
    
    return max(min_value, min(value, max_value))


def calculate_percentage_change(
    old_value: Union[int, float],
    new_value: Union[int, float]
) -> float:
    """
    Calculate percentage change between two values.
    
    Args:
        old_value: Original value
        new_value: New value
        
    Returns:
        Percentage change as float (can be negative)
        
    Example:
        >>> change = calculate_percentage_change(100, 120)  # Returns 20.0
        >>> change = calculate_percentage_change(100, 80)   # Returns -20.0
    """
    if old_value == 0:
        return 0.0 if new_value == 0 else float('inf') if new_value > 0 else float('-inf')
    
    return ((new_value - old_value) / old_value) * 100.0


def round_to_decimal_places(value: Union[int, float], decimal_places: int = 2) -> float:
    """
    Round a value to specified decimal places.
    
    This helper encapsulates the common pattern of rounding values that appears
    repeatedly in score calculations, metrics, and formatting.
    
    Args:
        value: Value to round
        decimal_places: Number of decimal places (default: 2)
        
    Returns:
        Rounded value as float
        
    Raises:
        ValueError: If decimal_places is negative
        
    Example:
        >>> score = round_to_decimal_places(calculated_score, 2)
        >>> rate = round_to_decimal_places(engagement_rate, 2)
        >>> time = round_to_decimal_places(process_time, 3)
    """
    if decimal_places < 0:
        raise ValueError(f"decimal_places must be >= 0, got {decimal_places}")
    
    return round(float(value), decimal_places)


def round_score(score: Union[int, float]) -> float:
    """
    Round a score to standard decimal places (2).
    
    This helper encapsulates the common pattern of rounding scores that appears
    repeatedly in ranking and engagement calculations.
    
    Args:
        score: Score to round
        
    Returns:
        Rounded score as float
        
    Example:
        >>> final_score = round_score(calculated_score)
    """
    from ..constants import SCORE_DECIMAL_PLACES
    return round_to_decimal_places(score, SCORE_DECIMAL_PLACES)

