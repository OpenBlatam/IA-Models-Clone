"""
Platform Factory - Factory de Plataformas
==========================================

Factory para crear handlers de plataformas sociales.
"""

import logging
from typing import Dict, Type, Optional
from .base_platform import SocialPlatform

logger = logging.getLogger(__name__)

# Registry de plataformas
_platform_registry: Dict[str, Type[SocialPlatform]] = {}


def register_platform(name: str, platform_class: Type[SocialPlatform]):
    """
    Registrar una plataforma
    
    Args:
        name: Nombre de la plataforma
        platform_class: Clase handler de la plataforma
    """
    _platform_registry[name.lower()] = platform_class
    logger.info(f"Plataforma registrada: {name}")


def get_platform_handler(platform_name: str) -> SocialPlatform:
    """
    Obtener handler de una plataforma
    
    Args:
        platform_name: Nombre de la plataforma
        
    Returns:
        Instancia del handler de la plataforma
        
    Raises:
        ValueError: Si la plataforma no está registrada
    """
    platform_name = platform_name.lower()
    
    if platform_name not in _platform_registry:
        # Lazy import de plataformas
        _load_platform_handlers()
    
    if platform_name not in _platform_registry:
        raise ValueError(f"Plataforma {platform_name} no está registrada")
    
    platform_class = _platform_registry[platform_name]
    return platform_class(platform_name)


def _load_platform_handlers():
    """Cargar handlers de plataformas"""
    try:
        from . import facebook, instagram, twitter, linkedin, tiktok, youtube
        
        # Registrar todas las plataformas
        if hasattr(facebook, 'FacebookPlatform'):
            register_platform("facebook", facebook.FacebookPlatform)
        
        if hasattr(instagram, 'InstagramPlatform'):
            register_platform("instagram", instagram.InstagramPlatform)
        
        if hasattr(twitter, 'TwitterPlatform'):
            register_platform("twitter", twitter.TwitterPlatform)
        
        if hasattr(linkedin, 'LinkedInPlatform'):
            register_platform("linkedin", linkedin.LinkedInPlatform)
        
        if hasattr(tiktok, 'TikTokPlatform'):
            register_platform("tiktok", tiktok.TikTokPlatform)
        
        if hasattr(youtube, 'YouTubePlatform'):
            register_platform("youtube", youtube.YouTubePlatform)
            
    except ImportError as e:
        logger.warning(f"Error cargando handlers de plataformas: {e}")




