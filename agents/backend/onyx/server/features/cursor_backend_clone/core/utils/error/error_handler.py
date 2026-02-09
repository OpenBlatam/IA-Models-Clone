"""
Error Handler - Sistema de Manejo de Errores Avanzado
======================================================

Sistema centralizado para manejo, formateo y recuperación de errores.
"""

import logging
import traceback
import asyncio
from typing import Any, Optional, Dict, Callable, List, Type
from functools import wraps
from datetime import datetime
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ErrorInfo:
    """Información detallada de error"""
    error_type: str
    message: str
    traceback: str
    timestamp: datetime = field(default_factory=datetime.now)
    context: Dict[str, Any] = field(default_factory=dict)
    error_code: Optional[str] = None
    recoverable: bool = False


class ErrorHandler:
    """
    Manejador centralizado de errores.
    
    Proporciona formateo, logging y recuperación de errores.
    """
    
    def __init__(self):
        self.error_history: List[ErrorInfo] = []
        self.max_history: int = 1000
        self.recovery_handlers: Dict[Type[Exception], Callable] = {}
        self.error_formatters: Dict[Type[Exception], Callable] = {}
    
    def handle(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None,
        log: bool = True
    ) -> ErrorInfo:
        """
        Manejar error y generar ErrorInfo.
        
        Args:
            error: Excepción a manejar
            context: Contexto adicional
            log: Si registrar en log
            
        Returns:
            ErrorInfo con información del error
        """
        error_info = ErrorInfo(
            error_type=type(error).__name__,
            message=str(error),
            traceback=traceback.format_exc(),
            context=context or {},
            error_code=getattr(error, 'error_code', None),
            recoverable=self._is_recoverable(error)
        )
        
        # Agregar a historial
        self.error_history.append(error_info)
        if len(self.error_history) > self.max_history:
            self.error_history.pop(0)
        
        # Logging
        if log:
            self._log_error(error_info)
        
        return error_info
    
    def format_error(self, error: Exception) -> Dict[str, Any]:
        """
        Formatear error para respuesta.
        
        Args:
            error: Excepción a formatear
            
        Returns:
            Diccionario formateado
        """
        # Verificar si hay formatter personalizado
        error_type = type(error)
        if error_type in self.error_formatters:
            return self.error_formatters[error_type](error)
        
        # Formato por defecto
        result = {
            "error": str(error),
            "error_type": error_type.__name__,
            "error_code": getattr(error, 'error_code', None)
        }
        
        # Agregar detalles si existen
        if hasattr(error, 'details'):
            result["details"] = error.details
        
        # Agregar traceback en modo debug
        if hasattr(error, '__traceback__'):
            result["traceback"] = traceback.format_tb(error.__traceback__)
        
        return result
    
    def register_recovery_handler(
        self,
        error_type: Type[Exception],
        handler: Callable[[Exception], Any]
    ) -> None:
        """
        Registrar handler de recuperación.
        
        Args:
            error_type: Tipo de excepción
            handler: Función de recuperación
        """
        self.recovery_handlers[error_type] = handler
        logger.info(f"✅ Recovery handler registered for {error_type.__name__}")
    
    def register_formatter(
        self,
        error_type: Type[Exception],
        formatter: Callable[[Exception], Dict[str, Any]]
    ) -> None:
        """
        Registrar formatter personalizado.
        
        Args:
            error_type: Tipo de excepción
            formatter: Función de formateo
        """
        self.error_formatters[error_type] = formatter
        logger.info(f"✅ Error formatter registered for {error_type.__name__}")
    
    def try_recover(self, error: Exception) -> Optional[Any]:
        """
        Intentar recuperar de error.
        
        Args:
            error: Excepción a recuperar
            
        Returns:
            Resultado de recuperación o None
        """
        error_type = type(error)
        
        # Buscar handler exacto
        if error_type in self.recovery_handlers:
            try:
                return self.recovery_handlers[error_type](error)
            except Exception as e:
                logger.error(f"Recovery handler failed: {e}")
                return None
        
        # Buscar handler por herencia
        for handler_type, handler in self.recovery_handlers.items():
            if issubclass(error_type, handler_type):
                try:
                    return handler(error)
                except Exception as e:
                    logger.error(f"Recovery handler failed: {e}")
                    return None
        
        return None
    
    def get_recent_errors(self, limit: int = 10) -> List[ErrorInfo]:
        """
        Obtener errores recientes.
        
        Args:
            limit: Número máximo de errores
            
        Returns:
            Lista de errores recientes
        """
        return self.error_history[-limit:]
    
    def get_errors_by_type(self, error_type: str) -> List[ErrorInfo]:
        """
        Obtener errores por tipo.
        
        Args:
            error_type: Nombre del tipo de error
            
        Returns:
            Lista de errores del tipo
        """
        return [e for e in self.error_history if e.error_type == error_type]
    
    def clear_history(self) -> int:
        """
        Limpiar historial de errores.
        
        Returns:
            Número de errores eliminados
        """
        count = len(self.error_history)
        self.error_history.clear()
        logger.info(f"🧹 Error history cleared ({count} errors)")
        return count
    
    def _is_recoverable(self, error: Exception) -> bool:
        """Verificar si error es recuperable"""
        error_type = type(error)
        return error_type in self.recovery_handlers or any(
            issubclass(error_type, handler_type)
            for handler_type in self.recovery_handlers.keys()
        )
    
    def _log_error(self, error_info: ErrorInfo) -> None:
        """Registrar error en log"""
        log_message = f"❌ {error_info.error_type}: {error_info.message}"
        
        if error_info.context:
            log_message += f" | Context: {error_info.context}"
        
        if error_info.recoverable:
            log_message += " | Recoverable"
        
        logger.error(log_message)
        
        # Log traceback en modo debug
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Traceback:\n{error_info.traceback}")


def handle_errors(
    handler: Optional[ErrorHandler] = None,
    default_return: Any = None,
    log: bool = True
):
    """
    Decorador para manejo automático de errores.
    
    Args:
        handler: ErrorHandler a usar (usa global si None)
        default_return: Valor a retornar en caso de error
        log: Si registrar errores
        
    Returns:
        Decorador
    """
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                error_handler = handler or _global_error_handler
                error_handler.handle(e, context={"function": func.__name__}, log=log)
                
                # Intentar recuperar
                recovery_result = error_handler.try_recover(e)
                if recovery_result is not None:
                    return recovery_result
                
                if default_return is not None:
                    return default_return
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_handler = handler or _global_error_handler
                error_handler.handle(e, context={"function": func.__name__}, log=log)
                
                # Intentar recuperar
                recovery_result = error_handler.try_recover(e)
                if recovery_result is not None:
                    return recovery_result
                
                if default_return is not None:
                    return default_return
                raise
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


# Handler global
_global_error_handler = ErrorHandler()

