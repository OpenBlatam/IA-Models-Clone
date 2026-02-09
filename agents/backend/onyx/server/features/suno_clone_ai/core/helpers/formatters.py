"""
Formatting Helpers

Utilities for formatting numbers, durations, sizes, etc.
"""

import logging
from typing import Union

logger = logging.getLogger(__name__)


def format_number(
    number: float,
    decimals: int = 2,
    use_scientific: bool = False
) -> str:
    """
    Format number for display.
    
    Args:
        number: Number to format
        decimals: Number of decimals
        use_scientific: Use scientific notation
        
    Returns:
        Formatted string
    """
    if use_scientific:
        return f"{number:.{decimals}e}"
    else:
        return f"{number:.{decimals}f}"


def format_duration(
    seconds: float,
    precision: int = 2
) -> str:
    """
    Format duration in seconds to human-readable string.
    
    Args:
        seconds: Duration in seconds
        precision: Decimal precision
        
    Returns:
        Formatted duration string
    """
    if seconds < 60:
        return f"{seconds:.{precision}f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.{precision}f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.{precision}f}h"


def format_size(
    bytes_size: int,
    binary: bool = True
) -> str:
    """
    Format file size to human-readable string.
    
    Args:
        bytes_size: Size in bytes
        binary: Use binary (1024) or decimal (1000) units
        
    Returns:
        Formatted size string
    """
    unit = 1024 if binary else 1000
    units = ['B', 'KB', 'MB', 'GB', 'TB'] if binary else ['B', 'KB', 'MB', 'GB', 'TB']
    
    size = float(bytes_size)
    unit_index = 0
    
    while size >= unit and unit_index < len(units) - 1:
        size /= unit
        unit_index += 1
    
    return f"{size:.2f} {units[unit_index]}"


def format_percentage(
    value: float,
    total: float = 100.0,
    decimals: int = 2
) -> str:
    """
    Format percentage.
    
    Args:
        value: Value
        total: Total value
        decimals: Decimal places
        
    Returns:
        Formatted percentage string
    """
    percentage = (value / total) * 100 if total > 0 else 0
    return f"{percentage:.{decimals}f}%"



