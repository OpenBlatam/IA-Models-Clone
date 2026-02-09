"""
Public API - Funciones públicas del módulo MCP Server
=====================================================

Funciones de utilidad públicas que pueden ser usadas por los usuarios
del módulo para obtener información sobre el estado y configuración.
"""

import logging
from typing import TYPE_CHECKING, Dict, Any, List, Tuple

logger = logging.getLogger(__name__)

__version__ = "2.2.0"
__author__ = "Blatam Academy"
__license__ = "Proprietary"


def get_version() -> str:
    """
    Retorna la versión del módulo MCP Server.
    
    Returns:
        Versión del módulo (ej: "2.0.0")
    """
    return __version__


def check_imports() -> Dict[str, bool]:
    """
    Verifica qué componentes están disponibles.
    
    Útil para debugging y verificar que todas las dependencias
    estén correctamente instaladas.
    
    Returns:
        Diccionario con estado de cada componente (True = disponible)
    """
    if TYPE_CHECKING:
        return {}
    
    from .utils.module_info import get_cached_imports_status
    return get_cached_imports_status()


def get_missing_imports() -> List[str]:
    """
    Retorna lista de componentes que no están disponibles.
    
    Returns:
        Lista de nombres de componentes faltantes
    """
    if TYPE_CHECKING:
        return []
    
    from .utils.module_info import get_cached_missing_imports
    return get_cached_missing_imports()


def get_available_features() -> Dict[str, bool]:
    """
    Retorna diccionario con características disponibles del módulo.
    
    Returns:
        Diccionario indicando qué características están disponibles
    """
    if TYPE_CHECKING:
        return {}
    
    from .utils.module_info import get_cached_available_features
    return get_cached_available_features()


def get_module_info() -> Dict[str, Any]:
    """
    Retorna información completa del módulo.
    
    Returns:
        Diccionario con información del módulo incluyendo versión,
        autor, características disponibles, y componentes faltantes
    """
    if TYPE_CHECKING:
        return {
            "version": __version__,
            "author": __author__,
            "license": __license__,
        }
    
    from .utils.module_info import (
        get_cached_available_features,
        get_cached_missing_imports,
        get_cached_imports_status,
        get_cached_module_statistics,
    )
    
    return {
        "version": __version__,
        "author": __author__,
        "license": __license__,
        "available_features": get_cached_available_features(),
        "missing_imports": get_cached_missing_imports(),
        "import_status": get_cached_imports_status(),
        "statistics": get_cached_module_statistics(),
    }


def get_diagnostics() -> Dict[str, Any]:
    """
    Obtener diagnóstico completo del módulo.
    
    Returns:
        Diccionario con información de diagnóstico completa.
    """
    if TYPE_CHECKING:
        return {}
    
    try:
        from .utils.diagnostics import get_module_diagnostics
        return get_module_diagnostics()
    except ImportError:
        logger.debug("Diagnostics utilities not available")
        return {
            "version": __version__,
            "error": "Diagnostics utilities not available",
        }


def check_health() -> Dict[str, Any]:
    """
    Verificar salud del módulo.
    
    Returns:
        Diccionario con estado de salud del módulo.
    """
    if TYPE_CHECKING:
        return {"status": "unknown"}
    
    try:
        from .utils.diagnostics import check_module_health
        return check_module_health()
    except ImportError:
        logger.debug("Health check utilities not available")
        return {
            "status": "unknown",
            "error": "Health check utilities not available",
        }


def validate_setup() -> Tuple[bool, List[str]]:
    """
    Validar configuración del módulo.
    
    Returns:
        Tupla (is_valid, errors) donde is_valid indica si la configuración es válida
        y errors es una lista de errores encontrados.
    """
    if TYPE_CHECKING:
        return True, []
    
    try:
        from .utils.diagnostics import validate_module_setup
        return validate_module_setup()
    except ImportError:
        logger.debug("Validation utilities not available")
        return False, ["Validation utilities not available"]

