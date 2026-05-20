"""
Component Lifecycle Utilities - Utilidades para ciclo de vida de componentes
===========================================================================

Utilidades para iniciar y detener componentes de forma consistente y segura.
"""

import asyncio
import logging
from typing import Optional, Any, List, Callable
from .error_handling import safe_async_call

logger = logging.getLogger(__name__)


async def safe_stop_component(
    component_name: str,
    component: Optional[Any],
    stop_func: Optional[Callable[[Any], Any]] = None,
    logger_instance: Optional[Any] = None
) -> None:
    """
    Detener un componente de forma segura.
    
    Args:
        component_name: Nombre del componente (para logging).
        component: Instancia del componente (puede ser None).
        stop_func: Función async para detener el componente. Si es None, usa stop().
        logger_instance: Logger a usar (opcional).
    """
    log = logger_instance or logger
    
    if component is None:
        return
    
    if stop_func:
        async def call_stop_func():
            if asyncio.iscoroutinefunction(stop_func):
                return await stop_func(component)
            else:
                return stop_func(component)
        
        await safe_async_call(
            call_stop_func,
            operation=f"stopping {component_name}",
            logger_instance=log,
            reraise=False
        )
    elif hasattr(component, 'stop'):
        await safe_async_call(
            component.stop,
            operation=f"stopping {component_name}",
            logger_instance=log,
            reraise=False
        )


async def safe_stop_components(
    components: List[tuple[str, Optional[Any], Optional[Callable]]],
    logger_instance: Optional[Any] = None
) -> None:
    """
    Detener múltiples componentes de forma segura.
    
    Args:
        components: Lista de tuplas (nombre, componente, stop_func opcional).
        logger_instance: Logger a usar (opcional).
    """
    for component_name, component, stop_func in components:
        await safe_stop_component(
            component_name,
            component,
            stop_func,
            logger_instance
        )


async def safe_start_component(
    component_name: str,
    component: Optional[Any],
    start_func: Optional[Callable[[Any], Any]] = None,
    logger_instance: Optional[Any] = None
) -> None:
    """
    Iniciar un componente de forma segura.
    
    Args:
        component_name: Nombre del componente (para logging).
        component: Instancia del componente (puede ser None).
        start_func: Función async para iniciar el componente. Si es None, usa start().
        logger_instance: Logger a usar (opcional).
    """
    log = logger_instance or logger
    
    if component is None:
        return
    
    if start_func:
        async def call_start_func():
            if asyncio.iscoroutinefunction(start_func):
                return await start_func(component)
            else:
                return start_func(component)
        
        await safe_async_call(
            call_start_func,
            operation=f"starting {component_name}",
            logger_instance=log,
            reraise=False
        )
    elif hasattr(component, 'start'):
        await safe_async_call(
            component.start,
            operation=f"starting {component_name}",
            logger_instance=log,
            reraise=False
        )
