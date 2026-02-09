"""Formatting utilities."""

from typing import Any, Optional
from decimal import Decimal


def format_number(
    number: float,
    decimals: int = 2,
    thousands_separator: str = ",",
    decimal_separator: str = "."
) -> str:
    """
    Format number with separators.
    
    Args:
        number: Number to format
        decimals: Decimal places
        thousands_separator: Thousands separator
        decimal_separator: Decimal separator
        
    Returns:
        Formatted number string
    """
    formatted = f"{number:,.{decimals}f}"
    
    if thousands_separator != ",":
        formatted = formatted.replace(",", "TEMP")
        formatted = formatted.replace(".", decimal_separator)
        formatted = formatted.replace("TEMP", thousands_separator)
    elif decimal_separator != ".":
        formatted = formatted.replace(".", decimal_separator)
    
    return formatted


def format_bytes(bytes_value: int, binary: bool = False) -> str:
    """
    Format bytes to human-readable size.
    
    Args:
        bytes_value: Bytes to format
        binary: Use binary (1024) or decimal (1000) units
        
    Returns:
        Formatted string (e.g., "1.5 MB")
    """
    units = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']
    base = 1024 if binary else 1000
    
    if bytes_value == 0:
        return "0 B"
    
    unit_index = 0
    size = float(bytes_value)
    
    while size >= base and unit_index < len(units) - 1:
        size /= base
        unit_index += 1
    
    return f"{size:.2f} {units[unit_index]}"


def format_percentage(value: float, decimals: int = 1) -> str:
    """
    Format value as percentage.
    
    Args:
        value: Value to format (0.0 to 1.0)
        decimals: Decimal places
        
    Returns:
        Formatted percentage string
    """
    return f"{value * 100:.{decimals}f}%"


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
        decimals: Decimal places
        
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
    formatted = format_number(amount, decimals)
    
    return f"{symbol}{formatted}"


def format_phone(phone: str, format: str = "US") -> str:
    """
    Format phone number.
    
    Args:
        phone: Phone number string
        format: Format style (US, international)
        
    Returns:
        Formatted phone number
    """
    import re
    
    # Remove non-digits
    digits = re.sub(r'\D', '', phone)
    
    if format == "US" and len(digits) == 10:
        return f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"
    elif format == "international" and len(digits) >= 10:
        return f"+{digits}"
    
    return phone


def format_credit_card(card_number: str, mask: bool = True) -> str:
    """
    Format credit card number.
    
    Args:
        card_number: Card number string
        mask: Mask middle digits
        
    Returns:
        Formatted card number
    """
    import re
    
    digits = re.sub(r'\D', '', card_number)
    
    if mask and len(digits) >= 4:
        masked = "*" * (len(digits) - 4)
        return f"{masked}{digits[-4:]}"
    
    # Format with spaces
    if len(digits) == 16:
        return f"{digits[:4]} {digits[4:8]} {digits[8:12]} {digits[12:]}"
    
    return card_number


def format_ssn(ssn: str, mask: bool = True) -> str:
    """
    Format Social Security Number.
    
    Args:
        ssn: SSN string
        mask: Mask middle digits
        
    Returns:
        Formatted SSN
    """
    import re
    
    digits = re.sub(r'\D', '', ssn)
    
    if len(digits) == 9:
        if mask:
            return f"***-**-{digits[-4:]}"
        else:
            return f"{digits[:3]}-{digits[3:5]}-{digits[5:]}"
    
    return ssn

