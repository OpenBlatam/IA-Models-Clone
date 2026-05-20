"""
Error Handler
=============

Manejo robusto de errores con recuperación automática.
"""

import logging
import traceback
from typing import Optional, Callable, Any, Dict
from functools import wraps
import asyncio

logger = logging.getLogger(__name__)


class ErrorHandler:
    """Manejador de errores robusto."""
    
    @staticmethod
    def retry_on_error(
        max_retries: int = 3,
        delay: float = 1.0,
        backoff: float = 2.0,
        exceptions: tuple = (Exception,)
    ):
        """
        Decorador para reintentar en caso de error.
        
        Args:
            max_retries: Número máximo de reintentos
            delay: Delay inicial
            backoff: Factor de backoff
            exceptions: Excepciones a capturar
        """
        def decorator(func: Callable):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                last_exception = None
                current_delay = delay
                
                for attempt in range(max_retries):
                    try:
                        return await func(*args, **kwargs)
                    except exceptions as e:
                        last_exception = e
                        if attempt < max_retries - 1:
                            logger.warning(
                                f"Intento {attempt + 1}/{max_retries} falló: {str(e)}. "
                                f"Reintentando en {current_delay}s..."
                            )
                            await asyncio.sleep(current_delay)
                            current_delay *= backoff
                        else:
                            logger.error(f"Todos los intentos fallaron: {str(e)}")
                
                raise last_exception
            
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                last_exception = None
                current_delay = delay
                
                for attempt in range(max_retries):
                    try:
                        return func(*args, **kwargs)
                    except exceptions as e:
                        last_exception = e
                        if attempt < max_retries - 1:
                            logger.warning(
                                f"Intento {attempt + 1}/{max_retries} falló: {str(e)}. "
                                f"Reintentando en {current_delay}s..."
                            )
                            import time
                            time.sleep(current_delay)
                            current_delay *= backoff
                        else:
                            logger.error(f"Todos los intentos fallaron: {str(e)}")
                
                raise last_exception
            
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            return sync_wrapper
        
        return decorator
    
    @staticmethod
    def safe_execute(
        func: Callable,
        default_return: Any = None,
        log_error: bool = True
    ) -> Any:
        """
        Ejecutar función de forma segura.
        
        Args:
            func: Función a ejecutar
            default_return: Valor por defecto en caso de error
            log_error: Registrar error
        
        Returns:
            Resultado o valor por defecto
        """
        try:
            return func()
        except Exception as e:
            if log_error:
                logger.error(f"Error en ejecución segura: {str(e)}\n{traceback.format_exc()}")
            return default_return
    
    @staticmethod
    async def safe_execute_async(
        func: Callable,
        default_return: Any = None,
        log_error: bool = True
    ) -> Any:
        """Ejecutar función async de forma segura."""
        try:
            return await func()
        except Exception as e:
            if log_error:
                logger.error(f"Error en ejecución async segura: {str(e)}\n{traceback.format_exc()}")
            return default_return




