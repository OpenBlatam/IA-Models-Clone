"""
Validadores Reutilizables

Proporciona validadores comunes para el sistema.
"""

import re
import uuid
from typing import Any, Optional, List
from datetime import datetime


class Validator:
    """Clase base para validadores"""
    
    @staticmethod
    def validate_uuid(value: str) -> bool:
        """Valida un UUID"""
        try:
            uuid.UUID(value)
            return True
        except (ValueError, TypeError):
            return False
    
    @staticmethod
    def validate_email(email: str) -> bool:
        """Valida un email"""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))
    
    @staticmethod
    def validate_url(url: str) -> bool:
        """Valida una URL"""
        pattern = r'^https?://(?:[-\w.])+(?:\:[0-9]+)?(?:/(?:[\w/_.])*)?(?:\?(?:[\w&=%.])*)?(?:#(?:\w)*)?$'
        return bool(re.match(pattern, url))
    
    @staticmethod
    def validate_iso_datetime(value: str) -> bool:
        """Valida un datetime en formato ISO"""
        try:
            datetime.fromisoformat(value.replace('Z', '+00:00'))
            return True
        except (ValueError, TypeError):
            return False
    
    @staticmethod
    def validate_audio_format(filename: str, allowed_formats: Optional[List[str]] = None) -> bool:
        """Valida formato de audio"""
        if allowed_formats is None:
            allowed_formats = ['wav', 'mp3', 'ogg', 'flac', 'm4a']
        
        ext = filename.split('.')[-1].lower()
        return ext in allowed_formats
    
    @staticmethod
    def validate_prompt(prompt: str, min_length: int = 1, max_length: int = 1000) -> bool:
        """Valida un prompt de texto"""
        if not isinstance(prompt, str):
            return False
        length = len(prompt.strip())
        return min_length <= length <= max_length
    
    @staticmethod
    def validate_bpm(bpm: float) -> bool:
        """Valida un valor de BPM"""
        return isinstance(bpm, (int, float)) and 20 <= bpm <= 300
    
    @staticmethod
    def validate_duration(duration: float) -> bool:
        """Valida una duración en segundos"""
        return isinstance(duration, (int, float)) and 0 < duration <= 3600
    
    @staticmethod
    def validate_price(price: float) -> bool:
        """Valida un precio"""
        return isinstance(price, (int, float)) and price >= 0
    
    @staticmethod
    def validate_rating(rating: int) -> bool:
        """Valida un rating (1-5)"""
        return isinstance(rating, int) and 1 <= rating <= 5


class ValidationError(Exception):
    """Excepción de validación"""
    pass


def validate_and_raise(validator_func, value: Any, error_message: str):
    """
    Valida un valor y lanza excepción si falla
    
    Args:
        validator_func: Función validadora
        value: Valor a validar
        error_message: Mensaje de error
    
    Raises:
        ValidationError: Si la validación falla
    """
    if not validator_func(value):
        raise ValidationError(error_message)

