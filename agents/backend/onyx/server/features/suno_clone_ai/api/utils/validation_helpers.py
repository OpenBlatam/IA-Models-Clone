"""
Helpers para validación optimizada

Incluye funciones de validación reutilizables y optimizadas.
"""

from typing import List, Optional
import uuid
import re
import logging

logger = logging.getLogger(__name__)


def validate_uuid_list(
    id_list: List[str],
    max_items: int = 50,
    field_name: str = "IDs"
) -> List[str]:
    """
    Valida una lista de UUIDs de forma eficiente.
    
    Args:
        id_list: Lista de IDs a validar
        max_items: Número máximo de items permitidos
        field_name: Nombre del campo para mensajes de error
        
    Returns:
        Lista de IDs validados
        
    Raises:
        ValueError: Si algún ID es inválido o se excede el máximo
    """
    # Guard clause: verificar longitud
    if len(id_list) > max_items:
        raise ValueError(f"Maximum {max_items} {field_name} allowed per request")
    
    if not id_list:
        raise ValueError(f"At least one {field_name} is required")
    
    validated_ids = []
    for item_id in id_list:
        item_id = item_id.strip()
        if not item_id:
            continue
        
        try:
            # Validar formato UUID
            uuid.UUID(item_id)
            validated_ids.append(item_id)
        except ValueError:
            raise ValueError(f"Invalid {field_name} format: {item_id}")
    
    return validated_ids


def parse_comma_separated_ids(
    ids_string: str,
    max_items: int = 50,
    field_name: str = "IDs"
) -> List[str]:
    """
    Parsea y valida una cadena de IDs separados por coma.
    
    Args:
        ids_string: Cadena con IDs separados por coma
        max_items: Número máximo de items permitidos
        field_name: Nombre del campo para mensajes de error
        
    Returns:
        Lista de IDs validados
    """
    if not ids_string or not ids_string.strip():
        raise ValueError(f"{field_name} string cannot be empty")
    
    id_list = [tid.strip() for tid in ids_string.split(",") if tid.strip()]
    return validate_uuid_list(id_list, max_items=max_items, field_name=field_name)


def validate_prompt_length(prompt: str, max_length: int = 500) -> str:
    """
    Valida y normaliza la longitud de un prompt.
    
    Args:
        prompt: Prompt a validar
        max_length: Longitud máxima permitida
        
    Returns:
        Prompt validado y normalizado
        
    Raises:
        ValueError: Si el prompt es inválido
    """
    if not prompt or not isinstance(prompt, str):
        raise ValueError("Prompt must be a non-empty string")
    
    prompt = prompt.strip()
    
    if not prompt:
        raise ValueError("Prompt cannot be empty")
    
    if len(prompt) > max_length:
        raise ValueError(f"Prompt exceeds maximum length of {max_length} characters")
    
    return prompt


def sanitize_string(value: str, max_length: Optional[int] = None) -> str:
    """
    Sanitiza una cadena de texto.
    
    Args:
        value: Cadena a sanitizar
        max_length: Longitud máxima opcional
        
    Returns:
        Cadena sanitizada
    """
    if not value:
        return ""
    
    # Normalizar espacios
    value = " ".join(value.split())
    
    # Limitar longitud si se especifica
    if max_length and len(value) > max_length:
        value = value[:max_length]
    
    return value.strip()


def validate_genre(genre: str) -> str:
    """
    Valida y normaliza un género musical.
    
    Args:
        genre: Género a validar
        
    Returns:
        Género validado y normalizado
    """
    if not genre:
        return ""
    
    genre = sanitize_string(genre, max_length=50)
    
    # Géneros comunes (opcional, para validación estricta)
    valid_genres = [
        "rock", "pop", "jazz", "classical", "electronic", "hip-hop",
        "country", "blues", "reggae", "folk", "metal", "punk", "r&b"
    ]
    
    # Si se quiere validación estricta, descomentar:
    # if genre.lower() not in valid_genres:
    #     raise ValueError(f"Invalid genre: {genre}")
    
    return genre.lower()


def validate_duration(duration: int, min_duration: int = 1, max_duration: int = 300) -> int:
    """
    Valida una duración en segundos.
    
    Args:
        duration: Duración a validar
        min_duration: Duración mínima en segundos
        max_duration: Duración máxima en segundos
        
    Returns:
        Duración validada
        
    Raises:
        ValueError: Si la duración es inválida
    """
    if not isinstance(duration, int):
        raise ValueError("Duration must be an integer")
    
    if duration < min_duration:
        raise ValueError(f"Duration must be at least {min_duration} seconds")
    
    if duration > max_duration:
        raise ValueError(f"Duration cannot exceed {max_duration} seconds")
    
    return duration

