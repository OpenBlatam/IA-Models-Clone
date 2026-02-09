"""
Helper functions for platform operations.
Eliminates repetitive platform mapping and validation patterns.
"""

from typing import Dict, Callable, Any, Optional, TypeVar
from enum import Enum

from ..core.models import Platform

T = TypeVar('T')


def get_platform_handler(
    platform: str,
    handlers: Dict[str, Callable[..., T]],
    default: Optional[Callable[..., T]] = None
) -> Optional[Callable[..., T]]:
    """
    Obtiene el handler para una plataforma específica.
    
    Args:
        platform: Nombre de la plataforma
        handlers: Diccionario de handlers por plataforma
        default: Handler por defecto si la plataforma no existe
        
    Returns:
        Handler o None
        
    Usage:
        >>> handler = get_platform_handler(
        ...     "tiktok",
        ...     {
        ...         "tiktok": extractor.extract_tiktok_profile,
        ...         "instagram": extractor.extract_instagram_profile
        ...     }
        ... )
    """
    platform_lower = platform.lower()
    return handlers.get(platform_lower, default)


def execute_for_platform(
    platform: str,
    handlers: Dict[str, Callable[..., T]],
    *args,
    default: Optional[T] = None,
    **kwargs
) -> Optional[T]:
    """
    Ejecuta el handler apropiado para una plataforma.
    
    Args:
        platform: Nombre de la plataforma
        handlers: Diccionario de handlers por plataforma
        *args: Argumentos para el handler
        default: Valor por defecto si la plataforma no existe
        **kwargs: Keyword arguments para el handler
        
    Returns:
        Resultado del handler o default
        
    Usage:
        >>> profile = execute_for_platform(
        ...     "tiktok",
        ...     {
        ...         "tiktok": extractor.extract_tiktok_profile,
        ...         "instagram": extractor.extract_instagram_profile
        ...     },
        ...     username
        ... )
    """
    handler = get_platform_handler(platform, handlers)
    if handler:
        return handler(*args, **kwargs)
    return default


def normalize_platform(platform: str) -> str:
    """
    Normaliza el nombre de la plataforma a minúsculas.
    
    Args:
        platform: Nombre de la plataforma
        
    Returns:
        Plataforma normalizada
        
    Usage:
        >>> normalize_platform("TikTok")
        'tiktok'
    """
    return platform.lower().strip()


def validate_platform_name(platform: str, valid_platforms: Optional[list] = None) -> str:
    """
    Valida y normaliza el nombre de la plataforma.
    
    Args:
        platform: Nombre de la plataforma
        valid_platforms: Lista de plataformas válidas (default: ["tiktok", "instagram", "youtube"])
        
    Returns:
        Plataforma normalizada y validada
        
    Raises:
        ValueError: Si la plataforma no es válida
        
    Usage:
        >>> validate_platform_name("TikTok")
        'tiktok'
        
        >>> validate_platform_name("invalid")
        ValueError: Plataforma debe ser una de: tiktok, instagram, youtube
    """
    if valid_platforms is None:
        valid_platforms = ["tiktok", "instagram", "youtube"]
    
    normalized = normalize_platform(platform)
    
    if normalized not in valid_platforms:
        raise ValueError(
            f"Plataforma debe ser una de: {', '.join(valid_platforms)}"
        )
    
    return normalized


def platform_to_enum(platform: str) -> Platform:
    """
    Convierte string de plataforma a enum Platform.
    
    Args:
        platform: Nombre de la plataforma
        
    Returns:
        Platform enum
        
    Raises:
        ValueError: Si la plataforma no es válida
        
    Usage:
        >>> platform_to_enum("tiktok")
        Platform.TIKTOK
    """
    normalized = validate_platform_name(platform)
    return Platform(normalized.upper())


def get_platform_display_name(platform: str) -> str:
    """
    Obtiene el nombre para mostrar de una plataforma.
    
    Args:
        platform: Nombre de la plataforma
        
    Returns:
        Nombre para mostrar
        
    Usage:
        >>> get_platform_display_name("tiktok")
        'TikTok'
    """
    display_names = {
        "tiktok": "TikTok",
        "instagram": "Instagram",
        "youtube": "YouTube"
    }
    normalized = normalize_platform(platform)
    return display_names.get(normalized, platform.title())








