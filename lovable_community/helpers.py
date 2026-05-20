"""
Helpers para la comunidad Lovable (backward compatibility)

Este archivo mantiene compatibilidad hacia atrás importando desde el módulo helpers/.
Las funciones están ahora organizadas en módulos específicos para mejor modularidad.

Para nuevas importaciones, use:
    from .helpers import chat_to_response, etc.
"""

# Import all helpers from the modular structure for backward compatibility
from .helpers.converters import (
    chat_to_response,
    chats_to_responses,
    remix_to_response,
    remixes_to_responses,
    vote_to_response,
)
from .helpers.tags import (
    parse_tags_string,
    format_tags_list,
    extract_tags_from_text,
)
from .helpers.text import (
    sanitize_text,
    truncate_text,
    format_datetime,
    get_chat_summary,
)
from .helpers.pagination import (
    calculate_pagination_metadata,
)
from .helpers.search import (
    normalize_search_query,
)
from .helpers.engagement import (
    calculate_engagement_rate,
    calculate_trending_score,
)
from .helpers.filters import (
    group_chats_by_user,
    sort_chats_by_score,
    filter_public_chats,
    filter_featured_chats,
)
from .helpers.validation import (
    validate_uuid_format,
)

# Re-export all for backward compatibility
__all__ = [
    "chat_to_response",
    "chats_to_responses",
    "remix_to_response",
    "remixes_to_responses",
    "vote_to_response",
    "parse_tags_string",
    "format_tags_list",
    "extract_tags_from_text",
    "sanitize_text",
    "truncate_text",
    "format_datetime",
    "get_chat_summary",
    "calculate_pagination_metadata",
    "normalize_search_query",
    "calculate_engagement_rate",
    "calculate_trending_score",
    "group_chats_by_user",
    "sort_chats_by_score",
    "filter_public_chats",
    "filter_featured_chats",
    "validate_uuid_format",
]

