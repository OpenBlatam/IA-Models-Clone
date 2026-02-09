"""
Common router helper functions
"""

from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


def validate_track_ids_count(
    track_ids: List[str],
    min_count: int,
    max_count: int,
    entity_name: str = "canciones"
) -> None:
    """
    Validate track IDs count
    
    Args:
        track_ids: List of track IDs
        min_count: Minimum required count
        max_count: Maximum allowed count
        entity_name: Name of entity for error messages
    
    Raises:
        ValueError: If count is invalid
    """
    if len(track_ids) < min_count:
        raise ValueError(
            f"Se necesitan al menos {min_count} {entity_name} para esta operación"
        )
    
    if len(track_ids) > max_count:
        raise ValueError(
            f"No se pueden procesar más de {max_count} {entity_name} a la vez"
        )


def extract_track_ids_from_request(
    request: Any,
    field_name: str = "track_ids"
) -> List[str]:
    """
    Extract track IDs from request object
    
    Args:
        request: Request object
        field_name: Name of field containing track IDs
    
    Returns:
        List of track IDs
    """
    if hasattr(request, field_name):
        track_ids = getattr(request, field_name)
        if isinstance(track_ids, str):
            return [track_ids]
        elif isinstance(track_ids, list):
            return track_ids
    return []


def build_pagination_params(
    page: Optional[int] = None,
    page_size: Optional[int] = None,
    default_page: int = 1,
    default_page_size: int = 20,
    max_page_size: int = 100
) -> Dict[str, int]:
    """
    Build pagination parameters with validation
    
    Args:
        page: Page number
        page_size: Items per page
        default_page: Default page number
        default_page_size: Default page size
        max_page_size: Maximum page size
    
    Returns:
        Dictionary with page and page_size
    """
    page = page if page is not None and page > 0 else default_page
    page_size = page_size if page_size is not None and page_size > 0 else default_page_size
    
    if page_size > max_page_size:
        page_size = max_page_size
        logger.warning(f"Page size limited to {max_page_size}")
    
    return {
        "page": page,
        "page_size": page_size,
        "offset": (page - 1) * page_size
    }


def format_error_message(
    message: str,
    field: Optional[str] = None,
    value: Optional[Any] = None
) -> str:
    """
    Format error message consistently
    
    Args:
        message: Base error message
        field: Field name (optional)
        value: Field value (optional)
    
    Returns:
        Formatted error message
    """
    if field and value is not None:
        return f"{message} (campo: {field}, valor: {value})"
    elif field:
        return f"{message} (campo: {field})"
    return message

