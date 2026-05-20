"""
Input Sanitizers
================
Utilidades para sanitizar y limpiar inputs del usuario.
"""

import re
from typing import Optional


def sanitize_text(text: str, max_length: Optional[int] = None) -> str:
    """
    Sanitizar texto de entrada.
    
    Args:
        text: Texto a sanitizar
        max_length: Longitud máxima opcional
        
    Returns:
        Texto sanitizado
    """
    if not text:
        return ""
    
    # Remover caracteres de control excepto newlines y tabs
    text = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F]', '', text)
    
    # Normalizar espacios en blanco
    text = re.sub(r'\s+', ' ', text)
    
    # Trim
    text = text.strip()
    
    # Limitar longitud si se especifica
    if max_length and len(text) > max_length:
        text = text[:max_length]
    
    return text


def sanitize_dog_breed(breed: Optional[str]) -> Optional[str]:
    """Sanitizar raza de perro."""
    if not breed:
        return None
    
    # Capitalizar correctamente
    breed = breed.strip().title()
    
    # Remover caracteres especiales excepto guiones y espacios
    breed = re.sub(r'[^a-zA-Z\s-]', '', breed)
    
    return breed if breed else None


def sanitize_dog_age(age: Optional[str]) -> Optional[str]:
    """Sanitizar edad del perro."""
    if not age:
        return None
    
    age = age.strip().lower()
    
    # Normalizar formatos comunes
    age = re.sub(r'\s+', ' ', age)
    
    return age if age else None


def sanitize_list(items: Optional[list]) -> list:
    """Sanitizar lista de items."""
    if not items:
        return []
    
    # Filtrar items vacíos y sanitizar
    sanitized = [sanitize_text(str(item)) for item in items if item]
    return [item for item in sanitized if item]

