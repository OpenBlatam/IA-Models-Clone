"""
Helpers para trabajar con servicios async/sync

Proporciona funciones utilitarias para manejar servicios de forma transparente.
"""

from typing import Optional, Any, Callable, Awaitable, Union
import logging

logger = logging.getLogger(__name__)


async def get_song_async_or_sync(
    song_service: Any,
    operation: str,
    *args: Any,
    **kwargs: Any
) -> Any:
    """
    Ejecuta una operación en el servicio async si está disponible, 
    sino usa el servicio síncrono.
    
    Args:
        song_service: Instancia del servicio síncrono
        operation: Nombre de la operación a ejecutar (ej: 'get_song', 'list_songs')
        *args: Argumentos posicionales para la operación
        **kwargs: Argumentos nombrados para la operación
        
    Returns:
        Resultado de la operación
        
    Example:
        song = await get_song_async_or_sync(song_service, 'get_song', song_id)
        songs = await get_song_async_or_sync(song_service, 'list_songs', limit=50)
    """
    # Intentar usar servicio async
    try:
        from ..dependencies import get_song_service_async
        async_service = await get_song_service_async()
        
        # Verificar que el servicio async tiene el método
        if hasattr(async_service, operation):
            method = getattr(async_service, operation)
            # Si es async, await; si no, llamar directamente
            if callable(method):
                result = method(*args, **kwargs)
                if hasattr(result, '__await__'):
                    return await result
                return result
    except (ImportError, AttributeError) as e:
        logger.debug(f"Async service not available, using sync: {e}")
    
    # Fallback a servicio síncrono
    if not hasattr(song_service, operation):
        raise AttributeError(f"Service does not have operation: {operation}")
    
    method = getattr(song_service, operation)
    return method(*args, **kwargs)


async def execute_async_operation(
    async_func: Optional[Callable[..., Awaitable[Any]]],
    sync_func: Callable[..., Any],
    *args: Any,
    **kwargs: Any
) -> Any:
    """
    Ejecuta una operación async si está disponible, sino usa la síncrona.
    
    Args:
        async_func: Función async opcional
        sync_func: Función síncrona de fallback
        *args: Argumentos posicionales
        **kwargs: Argumentos nombrados
        
    Returns:
        Resultado de la operación
    """
    if async_func and callable(async_func):
        try:
            result = async_func(*args, **kwargs)
            if hasattr(result, '__await__'):
                return await result
            return result
        except Exception as e:
            logger.debug(f"Async operation failed, using sync: {e}")
    
    return sync_func(*args, **kwargs)

