"""
Validadores reutilizables para la comunidad Lovable (backward compatibility)

Este archivo mantiene compatibilidad hacia atrás importando desde el módulo validators/.
Las funciones están ahora organizadas en módulos específicos para mejor modularidad.

Para nuevas importaciones, use:
    from .validators import validate_chat_id, etc.
"""

# Import all validators from the modular structure for backward compatibility
from .validators.ids import (
    validate_chat_id,
    validate_user_id,
)
from .validators.content import (
    validate_title,
    validate_description,
    validate_chat_content,
)
from .validators.tags_validators import (
    validate_tags,
)
from .validators.pagination_validators import (
    validate_page,
    validate_page_size,
    validate_limit,
)
from .validators.search_validators import (
    validate_search_query,
)
from .validators.votes import (
    validate_vote_type,
)
from .validators.sorting import (
    validate_sort_by,
    validate_order,
    validate_period,
)

# Re-export all for backward compatibility
__all__ = [
    "validate_chat_id",
    "validate_user_id",
    "validate_title",
    "validate_description",
    "validate_chat_content",
    "validate_tags",
    "validate_page",
    "validate_page_size",
    "validate_limit",
    "validate_search_query",
    "validate_vote_type",
    "validate_sort_by",
    "validate_order",
    "validate_period",
]
