"""
Status Utilities - Utilidades para obtener estado de componentes
================================================================

Utilidades para obtener y formatear el estado de componentes de forma consistente.
"""

import logging
from typing import Dict, Any, Optional, Callable, List
from functools import wraps

logger = logging.getLogger(__name__)


def safe_get_status(
    component_name: str,
    component: Optional[Any],
    status_func: Callable[[Any], Dict[str, Any]],
    default_status: Optional[Dict[str, Any]] = None,
    logger_instance: Optional[Any] = None
) -> Optional[Dict[str, Any]]:
    """
    Obtener estado de un componente de forma segura.
    
    Args:
        component_name: Nombre del componente (para logging).
        component: Instancia del componente (puede ser None).
        status_func: Función que obtiene el estado del componente.
        default_status: Estado por defecto si el componente es None o falla.
        logger_instance: Logger a usar (opcional).
    
    Returns:
        Diccionario con el estado o None si falla.
    """
    log = logger_instance or logger
    
    if component is None:
        return default_status
    
    try:
        return status_func(component)
    except Exception as e:
        log.error(f"Error getting {component_name} status: {e}", exc_info=True)
        return default_status


def get_component_status_dict(
    component_name: str,
    component: Optional[Any],
    status_getters: Dict[str, Callable[[Any], Any]],
    logger_instance: Optional[Any] = None
) -> Optional[Dict[str, Any]]:
    """
    Obtener estado de un componente usando múltiples getters.
    
    Args:
        component_name: Nombre del componente.
        component: Instancia del componente.
        status_getters: Diccionario de nombre -> función getter.
        logger_instance: Logger a usar (opcional).
    
    Returns:
        Diccionario con el estado del componente o None si falla.
    
    Example:
        status = get_component_status_dict(
            "test_runner",
            self.test_runner,
            {
                "test_runs": lambda c: len(c.test_results),
                "lint_runs": lambda c: len(c.lint_results)
            }
        )
    """
    log = logger_instance or logger
    
    if component is None:
        return None
    
    try:
        status = {}
        for key, getter in status_getters.items():
            try:
                status[key] = getter(component)
            except Exception as e:
                log.warning(f"Error getting {component_name}.{key}: {e}")
                status[key] = None
        return status
    except Exception as e:
        log.error(f"Error getting {component_name} status: {e}", exc_info=True)
        return None


def aggregate_status(
    status_dict: Dict[str, Any],
    component_name: str,
    status: Optional[Dict[str, Any]],
    logger_instance: Optional[Any] = None
) -> None:
    """
    Agregar estado de un componente al diccionario de estado general.
    
    Args:
        status_dict: Diccionario de estado general a actualizar.
        component_name: Nombre del componente.
        status: Estado del componente (puede ser None).
        logger_instance: Logger a usar (opcional).
    """
    if status is not None:
        status_dict[component_name] = status


def get_simple_count_status(
    component: Optional[Any],
    count_getter: Callable[[Any], int],
    status_key: str = "total"
) -> Optional[Dict[str, Any]]:
    """
    Obtener estado simple con un contador.
    
    Args:
        component: Instancia del componente.
        count_getter: Función que retorna el conteo.
        status_key: Clave para el contador en el diccionario.
    
    Returns:
        Diccionario con el conteo o None si el componente es None.
    
    Example:
        status = get_simple_count_status(
            self.completion_verifier,
            lambda c: len(c.get_all_tasks()),
            "total_tasks"
        )
    """
    if component is None:
        return None
    
    try:
        return {status_key: count_getter(component)}
    except Exception as e:
        logger.warning(f"Error getting count status: {e}")
        return None


def get_list_length_status(
    component: Optional[Any],
    list_getter: Callable[[Any], List[Any]],
    status_key: str = "total"
) -> Optional[Dict[str, Any]]:
    """
    Obtener estado basado en la longitud de una lista.
    
    Args:
        component: Instancia del componente.
        list_getter: Función que retorna una lista.
        status_key: Clave para el conteo en el diccionario.
    
    Returns:
        Diccionario con la longitud o None si el componente es None.
    """
    if component is None:
        return None
    
    try:
        items = list_getter(component)
        return {status_key: len(items) if items else 0}
    except Exception as e:
        logger.warning(f"Error getting list length status: {e}")
        return None




