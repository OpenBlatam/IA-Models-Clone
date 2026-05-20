"""
Formatting utilities
"""

from datetime import datetime
from typing import Optional


def format_date(date: datetime, format: str = "%Y-%m-%d") -> str:
    """Formatear fecha"""
    return date.strftime(format)


def format_currency(amount: float, currency: str = "EUR") -> str:
    """Formatear moneda"""
    if currency == "EUR":
        return f"€{amount:,.2f}"
    elif currency == "USD":
        return f"${amount:,.2f}"
    else:
        return f"{amount:,.2f} {currency}"


def format_duration(seconds: int) -> str:
    """Formatear duración en segundos"""
    if seconds < 60:
        return f"{seconds}s"
    elif seconds < 3600:
        minutes = seconds // 60
        return f"{minutes}m {seconds % 60}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{hours}h {minutes}m"




