"""
Utilidades para retry con backoff exponencial
"""

import asyncio
import logging
from typing import Callable, TypeVar, Optional
from functools import wraps

logger = logging.getLogger(__name__)

T = TypeVar('T')


async def retry_async(
    func: Callable[..., T],
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    exceptions: tuple = (Exception,)
) -> T:
    """
    Retry async function con backoff exponencial
    
    Args:
        func: Función async a ejecutar
        max_attempts: Número máximo de intentos
        initial_delay: Delay inicial en segundos
        max_delay: Delay máximo en segundos
        exponential_base: Base para cálculo exponencial
        exceptions: Tupla de excepciones a capturar
        
    Returns:
        Resultado de la función
    """
    delay = initial_delay
    
    for attempt in range(max_attempts):
        try:
            return await func()
        except exceptions as e:
            if attempt == max_attempts - 1:
                logger.error(f"Falló después de {max_attempts} intentos: {e}")
                raise
            
            logger.warning(f"Intento {attempt + 1}/{max_attempts} falló: {e}. Reintentando en {delay:.1f}s...")
            await asyncio.sleep(delay)
            delay = min(delay * exponential_base, max_delay)
    
    raise Exception("No se pudo completar después de todos los intentos")








