"""
Validators - Validadores reutilizables
"""

from typing import Any, Optional
import re


class Validators:
    """Clase con validadores reutilizables"""

    @staticmethod
    def is_email(email: str) -> bool:
        """Valida formato de email"""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))

    @staticmethod
    def is_url(url: str) -> bool:
        """Valida formato de URL"""
        pattern = r'^https?://.+'
        return bool(re.match(pattern, url))

    @staticmethod
    def validate_length(value: str, min_length: int = 0, max_length: Optional[int] = None) -> bool:
        """Valida longitud de string"""
        if len(value) < min_length:
            return False
        if max_length and len(value) > max_length:
            return False
        return True
