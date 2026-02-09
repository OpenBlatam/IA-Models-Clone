"""
Utilidades para procesamiento en batch

Incluye funciones para procesar múltiples items de forma eficiente.
"""

from typing import List, Callable, Any, Optional, TypeVar, Awaitable
import asyncio
import logging

logger = logging.getLogger(__name__)

T = TypeVar('T')
R = TypeVar('R')


async def process_batch_async(
    items: List[T],
    processor: Callable[[T], Awaitable[R]],
    max_concurrent: int = 10,
    error_handler: Optional[Callable[[T, Exception], R]] = None
) -> List[R]:
    """
    Procesa una lista de items de forma asíncrona con límite de concurrencia.
    
    Args:
        items: Lista de items a procesar
        processor: Función async que procesa cada item
        max_concurrent: Número máximo de operaciones concurrentes
        error_handler: Función opcional para manejar errores
        
    Returns:
        Lista de resultados (puede incluir None para items con error)
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    results: List[Optional[R]] = []
    
    async def process_item(item: T) -> Optional[R]:
        async with semaphore:
            try:
                return await processor(item)
            except Exception as e:
                logger.warning(f"Error processing item: {e}")
                if error_handler:
                    return error_handler(item, e)
                return None
    
    tasks = [process_item(item) for item in items]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Filtrar excepciones y convertir a lista de resultados
    processed_results: List[R] = []
    for result in results:
        if isinstance(result, Exception):
            logger.error(f"Exception in batch processing: {result}")
            continue
        if result is not None:
            processed_results.append(result)
    
    return processed_results


def process_batch_sync(
    items: List[T],
    processor: Callable[[T], R],
    batch_size: int = 10,
    error_handler: Optional[Callable[[T, Exception], R]] = None
) -> List[R]:
    """
    Procesa una lista de items de forma síncrona en batches.
    
    Args:
        items: Lista de items a procesar
        processor: Función que procesa cada item
        batch_size: Tamaño del batch
        error_handler: Función opcional para manejar errores
        
    Returns:
        Lista de resultados
    """
    results: List[R] = []
    
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        for item in batch:
            try:
                result = processor(item)
                if result is not None:
                    results.append(result)
            except Exception as e:
                logger.warning(f"Error processing item: {e}")
                if error_handler:
                    result = error_handler(item, e)
                    if result is not None:
                        results.append(result)
    
    return results


async def batch_get_songs(
    song_service: Any,
    song_ids: List[str],
    max_concurrent: int = 10
) -> List[Dict[str, Any]]:
    """
    Obtiene múltiples canciones de forma eficiente usando batch processing.
    
    Args:
        song_service: Servicio de canciones
        song_ids: Lista de IDs de canciones
        max_concurrent: Número máximo de requests concurrentes
        
    Returns:
        Lista de canciones encontradas
    """
    from ..helpers.service_helpers import get_song_async_or_sync
    
    async def get_song(song_id: str) -> Optional[Dict[str, Any]]:
        try:
            return await get_song_async_or_sync(song_service, 'get_song', song_id)
        except Exception as e:
            logger.warning(f"Error getting song {song_id}: {e}")
            return None
    
    results = await process_batch_async(
        song_ids,
        get_song,
        max_concurrent=max_concurrent
    )
    
    # Filtrar None values
    return [song for song in results if song is not None]

