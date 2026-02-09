"""
Utils Module - Utilidades Generales
Utilidades generales, helpers, y funciones comunes.

Rol en el Ecosistema IA:
- Funciones helper reutilizables, sin dependencias
- Procesamiento de datos, validaciones, transformaciones
- Utilidades compartidas entre módulos

Reglas de Importación:
- Este módulo NO debe importar otros módulos del proyecto
- Puede importar solo librerías externas estándar
"""

from .base import BaseUtil
from .service import UtilService
from .helpers import Helpers
from .validators import Validators
from .main import (
    get_util_service,
    safe_get,
    chunk_list,
    flatten_dict,
    is_email,
    is_url,
    validate_length,
)

# New unified utilities
from .serialization_utils import (
    SerializationUtils,
    to_dict,
    to_dict_list,
)
from .cache_key_utils import (
    CacheKeyGenerator,
    generate_cache_key,
    generate_cache_key_from_dict,
    generate_simple_cache_key,
)
# Environment utilities
from .env_utils import (
    EnvUtils,
    get_env,
    get_env_bool,
    get_env_int,
    get_env_float,
    get_env_list,
    get_env_path,
    load_env_file,
)
# Date/time utilities
from .datetime_utils import (
    DateTimeUtils,
    now,
    now_iso,
    parse_date,
    format_date,
    format_iso,
    add_days,
    time_ago,
    is_past,
    is_future,
)
# String utilities
from .string_utils import (
    StringUtils,
    slugify,
    sanitize_filename,
    truncate,
    normalize_whitespace,
    camel_to_snake,
    snake_to_camel,
    mask_sensitive,
)

__all__ = [
    # Clases principales
    "BaseUtil",
    "UtilService",
    "Helpers",
    "Validators",
    # Funciones de acceso rápido
    "get_util_service",
    "safe_get",
    "chunk_list",
    "flatten_dict",
    "is_email",
    "is_url",
    "validate_length",
    # Serialization utilities
    "SerializationUtils",
    "to_dict",
    "to_dict_list",
    # Cache key utilities
    "CacheKeyGenerator",
    "generate_cache_key",
    "generate_cache_key_from_dict",
    "generate_simple_cache_key",
    # Environment utilities
    "EnvUtils",
    "get_env",
    "get_env_bool",
    "get_env_int",
    "get_env_float",
    "get_env_list",
    "get_env_path",
    "load_env_file",
    # Date/time utilities
    "DateTimeUtils",
    "now",
    "now_iso",
    "parse_date",
    "format_date",
    "format_iso",
    "add_days",
    "time_ago",
    "is_past",
    "is_future",
    # String utilities
    "StringUtils",
    "slugify",
    "sanitize_filename",
    "truncate",
    "normalize_whitespace",
    "camel_to_snake",
    "snake_to_camel",
    "mask_sensitive",
]
