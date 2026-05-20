"""
Validadores adicionales para sanitización y validación de datos.
"""

import re
from typing import Optional, List, Dict, Any
from urllib.parse import urlparse
from config.logging_config import get_logger

logger = get_logger(__name__)


def sanitize_string(value: str, max_length: Optional[int] = None, allow_empty: bool = False) -> str:
    """
    Sanitizar string eliminando caracteres peligrosos.
    
    Args:
        value: String a sanitizar
        max_length: Longitud máxima permitida
        allow_empty: Permitir strings vacíos
        
    Returns:
        String sanitizado
        
    Raises:
        ValueError: Si el string es inválido
    """
    if not isinstance(value, str):
        raise ValueError("Value must be a string")
    
    # Eliminar caracteres de control excepto \n, \r, \t
    sanitized = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]', '', value)
    
    # Normalizar espacios en blanco
    sanitized = ' '.join(sanitized.split())
    
    if not allow_empty and not sanitized.strip():
        raise ValueError("String cannot be empty")
    
    if max_length and len(sanitized) > max_length:
        sanitized = sanitized[:max_length]
        logger.warning(f"String truncated to {max_length} characters")
    
    return sanitized


def validate_github_repository(owner: str, repo: str) -> tuple[str, str]:
    """
    Validar y sanitizar nombres de repositorio de GitHub.
    
    Args:
        owner: Propietario del repositorio
        repo: Nombre del repositorio
        
    Returns:
        Tupla (owner, repo) validados
        
    Raises:
        ValueError: Si los valores son inválidos
    """
    # GitHub repository names: alphanumeric, hyphens, underscores, dots
    github_pattern = re.compile(r'^[a-zA-Z0-9._-]+$')
    
    owner = sanitize_string(owner, max_length=100)
    repo = sanitize_string(repo, max_length=100)
    
    if not github_pattern.match(owner):
        raise ValueError(f"Invalid GitHub owner: {owner}")
    
    if not github_pattern.match(repo):
        raise ValueError(f"Invalid GitHub repository: {repo}")
    
    return owner, repo


def validate_url(url: str, allowed_schemes: Optional[List[str]] = None) -> str:
    """
    Validar URL.
    
    Args:
        url: URL a validar
        allowed_schemes: Esquemas permitidos (default: http, https)
        
    Returns:
        URL validada
        
    Raises:
        ValueError: Si la URL es inválida
    """
    if allowed_schemes is None:
        allowed_schemes = ['http', 'https']
    
    try:
        parsed = urlparse(url)
        
        if not parsed.scheme:
            raise ValueError("URL must have a scheme")
        
        if parsed.scheme not in allowed_schemes:
            raise ValueError(f"Scheme {parsed.scheme} not allowed. Allowed: {allowed_schemes}")
        
        if not parsed.netloc:
            raise ValueError("URL must have a netloc (domain)")
        
        return url
        
    except Exception as e:
        raise ValueError(f"Invalid URL: {e}")


def validate_email(email: str) -> str:
    """
    Validar email.
    
    Args:
        email: Email a validar
        
    Returns:
        Email validado
        
    Raises:
        ValueError: Si el email es inválido
    """
    email_pattern = re.compile(
        r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    )
    
    email = sanitize_string(email.strip().lower(), max_length=255)
    
    if not email_pattern.match(email):
        raise ValueError(f"Invalid email format: {email}")
    
    return email


def validate_task_id(task_id: str) -> str:
    """
    Validar ID de tarea (UUID).
    
    Args:
        task_id: ID a validar
        
    Returns:
        ID validado
        
    Raises:
        ValueError: Si el ID es inválido
    """
    uuid_pattern = re.compile(
        r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
        re.IGNORECASE
    )
    
    task_id = sanitize_string(task_id.strip(), max_length=36)
    
    if not uuid_pattern.match(task_id):
        raise ValueError(f"Invalid task ID format: {task_id}")
    
    return task_id


def sanitize_metadata(metadata: Dict[str, Any], max_depth: int = 5) -> Dict[str, Any]:
    """
    Sanitizar metadata eliminando valores peligrosos.
    
    Args:
        metadata: Diccionario de metadata
        max_depth: Profundidad máxima de recursión
        
    Returns:
        Metadata sanitizada
    """
    if max_depth <= 0:
        return {}
    
    sanitized = {}
    
    for key, value in metadata.items():
        # Sanitizar key
        try:
            safe_key = sanitize_string(str(key), max_length=100)
        except ValueError:
            logger.warning(f"Skipping invalid metadata key: {key}")
            continue
        
        # Sanitizar value
        if isinstance(value, str):
            try:
                sanitized[safe_key] = sanitize_string(value, max_length=10000)
            except ValueError:
                logger.warning(f"Skipping invalid metadata value for key: {key}")
        
        elif isinstance(value, (int, float, bool)):
            sanitized[safe_key] = value
        
        elif isinstance(value, dict):
            sanitized[safe_key] = sanitize_metadata(value, max_depth - 1)
        
        elif isinstance(value, list):
            sanitized[safe_key] = [
                sanitize_string(str(item), max_length=1000) if isinstance(item, str) else item
                for item in value[:100]  # Limitar a 100 items
            ]
        
        else:
            # Convertir otros tipos a string
            try:
                sanitized[safe_key] = sanitize_string(str(value), max_length=1000)
            except ValueError:
                pass
    
    return sanitized


def validate_pagination_params(page: Optional[int] = None, limit: Optional[int] = None) -> tuple[int, int]:
    """
    Validar parámetros de paginación.
    
    Args:
        page: Número de página (1-indexed)
        limit: Límite de items por página
        
    Returns:
        Tupla (page, limit) validados
    """
    if page is None or page < 1:
        page = 1
    
    if limit is None:
        limit = 100
    elif limit < 1:
        limit = 1
    elif limit > 1000:
        limit = 1000
    
    return page, limit
