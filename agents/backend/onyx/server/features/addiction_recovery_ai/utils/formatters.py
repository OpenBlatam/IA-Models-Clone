"""
Formatting utilities
Data formatting functions
"""

from typing import Any, Optional
from datetime import datetime
from utils.date_helpers import format_iso_date, get_current_utc


def format_number(
    value: float,
    decimals: int = 2,
    thousands_separator: str = ","
) -> str:
    """
    Format number with decimals and thousands separator
    
    Args:
        value: Number to format
        decimals: Number of decimal places
        thousands_separator: Thousands separator
    
    Returns:
        Formatted number string
    """
    formatted = f"{value:,.{decimals}f}"
    
    if thousands_separator != ",":
        formatted = formatted.replace(",", thousands_separator)
    
    return formatted


def format_percentage(
    value: float,
    decimals: int = 2
) -> str:
    """
    Format value as percentage
    
    Args:
        value: Value to format (0-1 or 0-100)
        decimals: Number of decimal places
    
    Returns:
        Formatted percentage string
    """
    if value > 1:
        value = value / 100
    
    return f"{value * 100:.{decimals}f}%"


def format_currency(
    value: float,
    symbol: str = "$",
    decimals: int = 2
) -> str:
    """
    Format value as currency
    
    Args:
        value: Value to format
        symbol: Currency symbol
        decimals: Number of decimal places
    
    Returns:
        Formatted currency string
    """
    return f"{symbol}{value:,.{decimals}f}"


def format_duration(
    seconds: float,
    format_type: str = "human"
) -> str:
    """
    Format duration in seconds
    
    Args:
        seconds: Duration in seconds
        format_type: Format type (human, iso, compact)
    
    Returns:
        Formatted duration string
    """
    if format_type == "human":
        if seconds < 60:
            return f"{seconds:.1f}s"
        if seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}m"
        if seconds < 86400:
            hours = seconds / 3600
            return f"{hours:.1f}h"
        days = seconds / 86400
        return f"{days:.1f}d"
    
    if format_type == "iso":
        return f"PT{seconds}S"
    
    return str(int(seconds))


def format_bytes(
    bytes_value: int,
    binary: bool = False
) -> str:
    """
    Format bytes to human-readable size
    
    Args:
        bytes_value: Size in bytes
        binary: Use binary (1024) or decimal (1000) units
    
    Returns:
        Formatted size string
    """
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    base = 1024 if binary else 1000
    
    if bytes_value == 0:
        return "0 B"
    
    unit_index = 0
    size = float(bytes_value)
    
    while size >= base and unit_index < len(units) - 1:
        size /= base
        unit_index += 1
    
    return f"{size:.2f} {units[unit_index]}"

