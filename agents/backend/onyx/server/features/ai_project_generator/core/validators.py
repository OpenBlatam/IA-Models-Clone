"""
Validators - Validadores comunes
================================

Validadores reutilizables para datos y requests.
"""

import re
from typing import Optional, List
from pydantic import BaseModel, validator


def validate_project_name(name: str) -> bool:
    """
    Valida nombre de proyecto.
    
    Args:
        name: Nombre a validar
    
    Returns:
        True si es válido
    """
    if not name or len(name) < 3 or len(name) > 50:
        return False
    return bool(re.match(r'^[a-zA-Z0-9_-]+$', name))


def validate_description(description: str) -> bool:
    """
    Valida descripción.
    
    Args:
        description: Descripción a validar
    
    Returns:
        True si es válida
    """
    if not description or len(description) < 10 or len(description) > 2000:
        return False
    # Verificar que tenga al menos 5 palabras únicas
    words = set(description.split())
    return len(words) >= 5


def validate_email(email: str) -> bool:
    """
    Valida email.
    
    Args:
        email: Email a validar
    
    Returns:
        True si es válido
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))


def validate_url(url: str) -> bool:
    """
    Valida URL.
    
    Args:
        url: URL a validar
    
    Returns:
        True si es válida
    """
    pattern = r'^https?://[^\s/$.?#].[^\s]*$'
    return bool(re.match(pattern, url))


class ProjectNameValidator:
    """Validador de nombres de proyecto"""
    
    @staticmethod
    def validate(name: Optional[str]) -> Optional[str]:
        """
        Valida y normaliza nombre de proyecto.
        
        Args:
            name: Nombre a validar
        
        Returns:
            Nombre normalizado o None
        
        Raises:
            ValueError: Si el nombre no es válido
        """
        if not name:
            return None
        
        name = name.strip()
        
        if not validate_project_name(name):
            raise ValueError(
                "Project name must be 3-50 characters and contain only "
                "letters, numbers, hyphens, and underscores"
            )
        
        return name


class DescriptionValidator:
    """Validador de descripciones"""
    
    @staticmethod
    def validate(description: str) -> str:
        """
        Valida y normaliza descripción.
        
        Args:
            description: Descripción a validar
        
        Returns:
            Descripción normalizada
        
        Raises:
            ValueError: Si la descripción no es válida
        """
        if not description:
            raise ValueError("Description cannot be empty")
        
        description = description.strip()
        
        if not validate_description(description):
            raise ValueError(
                "Description must be 10-2000 characters and contain "
                "at least 5 unique words"
            )
        
        return description















