"""
Utilidades generales para la comunidad Lovable (backward compatibility)

Este archivo mantiene compatibilidad hacia atrás importando desde los módulos modulares.
Las funciones están ahora organizadas en:
- helpers/: Funciones de ayuda generales (text, tags, pagination, search, common)
- utils/: Utilidades avanzadas (performance, retry, validation, batch, security)

Para nuevas importaciones, use:
    from .helpers import sanitize_text, format_datetime, etc.
    from .utils import timer, retry, validate_length, etc.
"""

# Import from helpers/ for backward compatibility
from .helpers.text import (
    sanitize_text,
    truncate_text,
    format_datetime,
    slugify,
    calculate_time_ago,
)
from .helpers.validation import validate_uuid_format
from .helpers.tags import normalize_tags
from .helpers.pagination import (
    calculate_pagination_metadata,
    validate_page_params,
)
from .helpers.search import (
    extract_search_terms,
    build_search_filter,
)
from .helpers.common import (
    parse_date_range,
    remove_duplicates,
    safe_int,
    safe_float,
    format_bytes,
    get_percentage,
)

# Import from utils/ for backward compatibility
from .utils.batch import chunk_list
from .utils.security import (
    generate_hash,
    mask_email,
    validate_email,
    validate_url,
)

# Create aliases for backward compatibility
sanitize_string = sanitize_text
is_valid_email = validate_email

# Re-export calculate_pagination_info as alias for calculate_pagination_metadata
def calculate_pagination_info(page: int, page_size: int, total: int):
    """
    Calculates pagination information (backward compatibility alias).
    
    Args:
        page: Current page
        page_size: Page size
        total: Total items
        
    Returns:
        Dictionary with pagination information
    """
    result = calculate_pagination_metadata(total, page, page_size)
    return {
        "page": result["page"],
        "page_size": result["page_size"],
        "total": result["total"],
        "total_pages": result["total_pages"],
        "has_next": result.get("has_more", False),
        "has_prev": result.get("has_previous", False),
        "has_more": result.get("has_more", False)
    }

__all__ = [
    # Text utilities (from helpers.text)
    "sanitize_string",
    "sanitize_text",
    "truncate_text",
    "format_datetime",
    "slugify",
    "calculate_time_ago",
    # Validation (from helpers.validation)
    "validate_uuid_format",
    # Tags (from helpers.tags)
    "normalize_tags",
    # Pagination (from helpers.pagination)
    "calculate_pagination_info",
    "calculate_pagination_metadata",
    "validate_page_params",
    # Search (from helpers.search)
    "extract_search_terms",
    "build_search_filter",
    # Common utilities (from helpers.common)
    "parse_date_range",
    "remove_duplicates",
    "safe_int",
    "safe_float",
    "format_bytes",
    "get_percentage",
    # Batch utilities (from utils.batch)
    "chunk_list",
    # Security utilities (from utils.security)
    "generate_hash",
    "mask_email",
    "is_valid_email",
    "validate_email",
    "validate_url",
]
