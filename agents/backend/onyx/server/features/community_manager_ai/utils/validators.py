"""
Validators - Validadores
========================

Funciones de validación para el sistema.
"""

import logging
from typing import List, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)

# Límites de caracteres por plataforma
PLATFORM_LIMITS = {
    "twitter": 280,
    "facebook": 5000,
    "instagram": 2200,
    "linkedin": 3000,
    "tiktok": 2200,
    "youtube": 10000,  # Para descripciones
    "x": 280,  # Alias de Twitter
}

# Plataformas soportadas
SUPPORTED_PLATFORMS = [
    "facebook",
    "instagram",
    "twitter",
    "x",
    "linkedin",
    "tiktok",
    "youtube",
]


def validate_platform(platform: str) -> bool:
    """
    Validar que una plataforma sea soportada
    
    Args:
        platform: Nombre de la plataforma
        
    Returns:
        True si es válida
    """
    return platform.lower() in SUPPORTED_PLATFORMS


def validate_platforms(platforms: List[str]) -> Tuple[bool, Optional[str]]:
    """
    Validar lista de plataformas
    
    Args:
        platforms: Lista de plataformas
        
    Returns:
        Tuple (is_valid, error_message)
    """
    if not platforms:
        return False, "Debe especificar al menos una plataforma"
    
    for platform in platforms:
        if not validate_platform(platform):
            return False, f"Plataforma no soportada: {platform}"
    
    return True, None


def validate_content_length(content: str, platform: str) -> Tuple[bool, Optional[str]]:
    """
    Validar longitud de contenido para una plataforma
    
    Args:
        content: Contenido a validar
        platform: Plataforma objetivo
        
    Returns:
        Tuple (is_valid, error_message)
    """
    if not content or not isinstance(content, str):
        return False, "El contenido debe ser una cadena de texto no vacía"
    
    platform = platform.lower()
    
    if platform not in PLATFORM_LIMITS:
        return True, None
    
    limit = PLATFORM_LIMITS[platform]
    length = len(content)
    
    if length == 0:
        return False, "El contenido no puede estar vacío"
    
    if length > limit:
        return False, f"Contenido excede el límite de {limit} caracteres para {platform} (actual: {length})"
    
    min_length = 10
    if length < min_length:
        return False, f"El contenido debe tener al menos {min_length} caracteres (actual: {length})"
    
    return True, None


def validate_scheduled_time(scheduled_time: datetime) -> Tuple[bool, Optional[str]]:
    """
    Validar que la fecha programada sea en el futuro
    
    Args:
        scheduled_time: Fecha programada
        
    Returns:
        Tuple (is_valid, error_message)
    """
    from datetime import timedelta
    
    now = datetime.now()
    
    if scheduled_time < now:
        return False, "La fecha programada debe ser en el futuro"
    
    max_future_days = 365
    max_future = now + timedelta(days=max_future_days)
    
    if scheduled_time > max_future:
        return False, f"La fecha programada no puede ser más de {max_future_days} días en el futuro"
    
    min_future_minutes = 1
    min_future = now + timedelta(minutes=min_future_minutes)
    
    if scheduled_time < min_future:
        return False, f"La fecha programada debe ser al menos {min_future_minutes} minuto(s) en el futuro"
    
    return True, None


def validate_media_paths(media_paths: List[str], platform: str) -> Tuple[bool, Optional[str]]:
    """
    Validar rutas de medios
    
    Args:
        media_paths: Lista de rutas
        platform: Plataforma objetivo
        
    Returns:
        Tuple (is_valid, error_message)
    """
    import os
    from pathlib import Path
    
    if not media_paths:
        return True, None
    
    platform_lower = platform.lower()
    
    if platform_lower in ["instagram", "tiktok"]:
        if not media_paths:
            return False, f"{platform} requiere al menos un archivo multimedia"
    
    max_files = {
        "instagram": 10,
        "facebook": 10,
        "twitter": 4,
        "x": 4,
        "linkedin": 9,
        "tiktok": 1,
        "youtube": 1
    }
    
    max_count = max_files.get(platform_lower, 10)
    if len(media_paths) > max_count:
        return False, f"{platform} permite máximo {max_count} archivos multimedia"
    
    for path in media_paths:
        if not path or not isinstance(path, str):
            return False, "Ruta de archivo inválida"
        
        path_obj = Path(path)
        
        if not path_obj.exists():
            return False, f"Archivo no encontrado: {path}"
        
        if not path_obj.is_file():
            return False, f"La ruta no es un archivo: {path}"
        
        file_size = path_obj.stat().st_size
        max_size_mb = 100
        if file_size > max_size_mb * 1024 * 1024:
            return False, f"Archivo demasiado grande: {path} ({file_size / (1024*1024):.2f}MB > {max_size_mb}MB)"
        
        ext = path_obj.suffix.lower()
        
        image_extensions = [".jpg", ".jpeg", ".png", ".gif", ".webp"]
        video_extensions = [".mp4", ".mov", ".avi", ".mkv", ".webm"]
        valid_extensions = image_extensions + video_extensions
        
        if ext not in valid_extensions:
            return False, f"Formato de archivo no soportado: {ext}. Formatos válidos: {', '.join(valid_extensions)}"
        
        if platform_lower == "tiktok" and ext not in video_extensions:
            return False, f"TikTok requiere archivos de video, no imágenes"
        
        if platform_lower == "youtube" and ext not in video_extensions:
            return False, f"YouTube requiere archivos de video"
    
    return True, None



