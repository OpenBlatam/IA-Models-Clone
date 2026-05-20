"""
Module Info - Cached module information utilities
=================================================

Utilidades para obtener información del módulo de forma eficiente,
usando caché para evitar recrear ImportManager múltiples veces.
"""

import logging
from typing import Dict, Any, Optional, List
from functools import lru_cache

logger = logging.getLogger(__name__)

# Caché global para ImportManager
_module_info_cache: Optional[Dict[str, Any]] = None


def _get_cached_import_manager():
    """
    Obtener ImportManager con caché para evitar recrearlo.
    
    Returns:
        Tupla (ImportManager, namespace)
        
    Raises:
        RuntimeError: Si hay error al crear el ImportManager
    """
    global _module_info_cache
    
    if _module_info_cache is None:
        try:
            from .._import_manager import ImportManager
            
            _temp_namespace: Dict[str, Any] = {}
            _import_manager = ImportManager(_temp_namespace)
            _import_manager.import_all()
            
            _module_info_cache = {
                "manager": _import_manager,
                "namespace": _temp_namespace,
            }
            logger.debug("ImportManager cached successfully")
        except ImportError as e:
            logger.error(f"Failed to import ImportManager: {e}", exc_info=True)
            raise RuntimeError(f"Failed to import ImportManager: {e}") from e
        except Exception as e:
            logger.error(f"Error creating ImportManager: {e}", exc_info=True)
            raise RuntimeError(f"Failed to create ImportManager: {e}") from e
    
    return _module_info_cache["manager"], _module_info_cache["namespace"]


def clear_module_info_cache() -> None:
    """
    Limpiar caché de información del módulo.
    
    Útil para forzar recarga de imports en desarrollo.
    """
    global _module_info_cache
    _module_info_cache = None


@lru_cache(maxsize=1)
def get_cached_imports_status() -> Dict[str, bool]:
    """
    Obtener estado de imports con caché.
    
    Returns:
        Diccionario con estado de cada componente (True = disponible)
        
    Raises:
        RuntimeError: Si hay error al obtener el estado de imports
    """
    try:
        manager, _ = _get_cached_import_manager()
        if not hasattr(manager, 'check_imports'):
            raise RuntimeError("ImportManager does not have check_imports method")
        return manager.check_imports()
    except Exception as e:
        logger.error(f"Error getting imports status: {e}", exc_info=True)
        raise RuntimeError(f"Failed to get imports status: {e}") from e


@lru_cache(maxsize=1)
def get_cached_available_features() -> Dict[str, bool]:
    """
    Obtener características disponibles con caché.
    
    Returns:
        Diccionario con características disponibles (True = disponible)
        
    Raises:
        RuntimeError: Si hay error al obtener las características
    """
    try:
        manager, _ = _get_cached_import_manager()
        if not hasattr(manager, 'get_available_features'):
            raise RuntimeError("ImportManager does not have get_available_features method")
        return manager.get_available_features()
    except Exception as e:
        logger.error(f"Error getting available features: {e}", exc_info=True)
        raise RuntimeError(f"Failed to get available features: {e}") from e


def get_cached_missing_imports() -> List[str]:
    """
    Obtener imports faltantes con caché.
    
    Returns:
        Lista de componentes faltantes (ordenada alfabéticamente)
        
    Raises:
        RuntimeError: Si hay error al obtener los imports faltantes
    """
    try:
        imports_status = get_cached_imports_status()
        if not isinstance(imports_status, dict):
            raise RuntimeError(f"Expected dict from get_cached_imports_status, got {type(imports_status)}")
        missing = [name for name, available in imports_status.items() if not available]
        return sorted(missing)  # Ordenar para consistencia
    except Exception as e:
        logger.error(f"Error getting missing imports: {e}", exc_info=True)
        raise RuntimeError(f"Failed to get missing imports: {e}") from e


def get_cached_module_statistics() -> Dict[str, Any]:
    """
    Obtener estadísticas del módulo con caché.
    
    Returns:
        Diccionario con estadísticas del módulo, incluyendo:
        - total_components: Total de componentes
        - available_components: Componentes disponibles
        - missing_components: Componentes faltantes
        - cache_hit: Si se usó caché
        
    Raises:
        RuntimeError: Si hay error al obtener las estadísticas
    """
    try:
        manager, _ = _get_cached_import_manager()
        if not hasattr(manager, 'get_import_statistics'):
            # Si no tiene el método, calcular estadísticas manualmente
            imports_status = get_cached_imports_status()
            total = len(imports_status)
            available = sum(1 for v in imports_status.values() if v)
            missing = total - available
            
            return {
                "total_components": total,
                "available_components": available,
                "missing_components": missing,
                "cache_hit": True,
            }
        
        stats = manager.get_import_statistics()
        if not isinstance(stats, dict):
            raise RuntimeError(f"Expected dict from get_import_statistics, got {type(stats)}")
        
        stats["cache_hit"] = True
        return stats
    except Exception as e:
        logger.error(f"Error getting module statistics: {e}", exc_info=True)
        raise RuntimeError(f"Failed to get module statistics: {e}") from e

