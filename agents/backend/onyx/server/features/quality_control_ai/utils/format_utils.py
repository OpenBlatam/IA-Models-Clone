"""
Format Utilities

Utility functions for formatting data.
"""

from typing import Any, Optional
from datetime import datetime


def format_number(
    number: float,
    decimals: int = 2,
    thousands_sep: str = ","
) -> str:
    """
    Format number with decimals and thousands separator.
    
    Args:
        number: Number to format
        decimals: Number of decimal places
        thousands_sep: Thousands separator
    
    Returns:
        Formatted string
    """
    return f"{number:,.{decimals}f}".replace(",", thousands_sep)


def format_percentage(
    value: float,
    decimals: int = 2
) -> str:
    """
    Format value as percentage.
    
    Args:
        value: Value to format (0.0 to 1.0 or 0 to 100)
        decimals: Number of decimal places
    
    Returns:
        Formatted percentage string
    """
    # If value is between 0 and 1, multiply by 100
    if 0 <= value <= 1:
        value = value * 100
    
    return f"{value:.{decimals}f}%"


def format_currency(
    amount: float,
    currency: str = "USD",
    decimals: int = 2
) -> str:
    """
    Format amount as currency.
    
    Args:
        amount: Amount to format
        currency: Currency code
        decimals: Number of decimal places
    
    Returns:
        Formatted currency string
    """
    symbols = {
        "USD": "$",
        "EUR": "€",
        "GBP": "£",
        "JPY": "¥",
    }
    symbol = symbols.get(currency, currency)
    return f"{symbol}{amount:,.{decimals}f}"


def format_datetime_human(
    dt: datetime,
    format_type: str = "short"
) -> str:
    """
    Format datetime in human-readable format.
    
    Args:
        dt: Datetime to format
        format_type: Format type ('short', 'long', 'relative')
    
    Returns:
        Formatted string
    """
    if format_type == "short":
        return dt.strftime("%Y-%m-%d %H:%M")
    elif format_type == "long":
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    elif format_type == "relative":
        from .date_utils import time_ago
        return time_ago(dt)
    else:
        return dt.isoformat()


def format_list(
    items: list,
    separator: str = ", ",
    max_items: Optional[int] = None
) -> str:
    """
    Format list as string.
    
    Args:
        items: List to format
        separator: Separator string
        max_items: Maximum items to show (None for all)
    
    Returns:
        Formatted string
    """
    if max_items and len(items) > max_items:
        shown = items[:max_items]
        return separator.join(str(item) for item in shown) + f" ... ({len(items)} total)"
    return separator.join(str(item) for item in items)



