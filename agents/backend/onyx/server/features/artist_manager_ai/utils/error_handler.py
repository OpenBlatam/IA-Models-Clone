"""
Error Handler
=============

Manejo avanzado de errores.
"""

import logging
import traceback
import asyncio
from typing import Dict, Any, Optional, Callable, Type
from functools import wraps
from datetime import datetime

logger = logging.getLogger(__name__)


class ErrorHandler:
    """Manejador de errores."""
    
    @staticmethod
    def handle_error(
        error: Exception,
        context: Optional[Dict[str, Any]] = None,
        raise_again: bool = False
    ) -> Dict[str, Any]:
        """
        Manejar error de forma estructurada.
        
        Args:
            error: Excepción
            context: Contexto adicional
            raise_again: Si debe relanzar la excepción
        
        Returns:
            Información del error
        """
        error_info = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": traceback.format_exc(),
            "timestamp": datetime.now().isoformat(),
            "context": context or {}
        }
        
        logger.error(
            f"Error: {error_info['error_type']} - {error_info['error_message']}",
            extra=error_info,
            exc_info=True
        )
        
        if raise_again:
            raise
        
        return error_info
    
    @staticmethod
    def error_handler(
        error_types: Optional[tuple] = None,
        default_return: Any = None,
        log_error: bool = True
    ):
        """
        Decorador para manejo de errores.
        
        Args:
            error_types: Tipos de error a capturar
            default_return: Valor por defecto en caso de error
            log_error: Si debe loguear el error
        
        Returns:
            Decorador
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    if error_types and not isinstance(e, error_types):
                        raise
                    
                    if log_error:
                        ErrorHandler.handle_error(e, {
                            "function": func.__name__,
                            "args": str(args),
                            "kwargs": str(kwargs)
                        })
                    
                    return default_return
            
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if error_types and not isinstance(e, error_types):
                        raise
                    
                    if log_error:
                        ErrorHandler.handle_error(e, {
                            "function": func.__name__,
                            "args": str(args),
                            "kwargs": str(kwargs)
                        })
                    
                    return default_return
            
            if hasattr(func, '__code__') and func.__code__.co_flags & 0x80:
                return async_wrapper
            return sync_wrapper
        
        return decorator
    
    @staticmethod
    def safe_execute(
        func: Callable,
        *args,
        default_return: Any = None,
        **kwargs
    ) -> Any:
        """
        Ejecutar función de forma segura.
        
        Args:
            func: Función a ejecutar
            *args: Argumentos
            default_return: Valor por defecto
            **kwargs: Argumentos nombrados
        
        Returns:
            Resultado o valor por defecto
        """
        try:
            return func(*args, **kwargs)
        except Exception as e:
            ErrorHandler.handle_error(e)
            return default_return
    
    @staticmethod
    async def safe_execute_async(
        func: Callable,
        *args,
        default_return: Any = None,
        **kwargs
    ) -> Any:
        """
        Ejecutar función async de forma segura.
        
        Args:
            func: Función a ejecutar
            *args: Argumentos
            default_return: Valor por defecto
            **kwargs: Argumentos nombrados
        
        Returns:
            Resultado o valor por defecto
        """
        try:
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            else:
                return func(*args, **kwargs)
        except Exception as e:
            ErrorHandler.handle_error(e)
            return default_return

