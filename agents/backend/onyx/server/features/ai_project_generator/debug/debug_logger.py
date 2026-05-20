"""
Debug Logger - Logger avanzado para debugging
============================================

Logger con niveles de debug, contexto, y formateo avanzado.
"""

import logging
import sys
import traceback
from typing import Optional, Dict, Any
from datetime import datetime
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class DebugLogger:
    """
    Logger avanzado para debugging.
    
    Características:
    - Múltiples niveles de debug
    - Contexto de ejecución
    - Stack traces detallados
    - Formato estructurado
    - Exportación a archivos
    """
    
    def __init__(
        self,
        name: str = "debug",
        level: int = logging.DEBUG,
        enable_file_logging: bool = True,
        log_dir: str = "logs"
    ):
        """
        Args:
            name: Nombre del logger
            level: Nivel de logging
            enable_file_logging: Habilitar logging a archivo
            log_dir: Directorio para logs
        """
        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # Formato estructurado
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        
        # Handler para consola
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # Handler para archivo
        if enable_file_logging:
            log_path = Path(log_dir)
            log_path.mkdir(exist_ok=True)
            
            file_handler = logging.FileHandler(
                log_path / f"{name}_{datetime.now().strftime('%Y%m%d')}.log"
            )
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        
        self.context: Dict[str, Any] = {}
    
    def set_context(self, **kwargs):
        """Establece contexto para logs"""
        self.context.update(kwargs)
    
    def clear_context(self):
        """Limpia contexto"""
        self.context.clear()
    
    def debug(self, message: str, **kwargs):
        """Log de debug"""
        self._log(logging.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log de info"""
        self._log(logging.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log de warning"""
        self._log(logging.WARNING, message, **kwargs)
    
    def error(self, message: str, exc_info: bool = True, **kwargs):
        """Log de error con stack trace"""
        self._log(logging.ERROR, message, exc_info=exc_info, **kwargs)
    
    def critical(self, message: str, exc_info: bool = True, **kwargs):
        """Log crítico"""
        self._log(logging.CRITICAL, message, exc_info=exc_info, **kwargs)
    
    def _log(self, level: int, message: str, exc_info: bool = False, **kwargs):
        """Log interno con contexto"""
        if self.context:
            message = f"{message} | Context: {json.dumps(self.context)}"
        
        if kwargs:
            message = f"{message} | Data: {json.dumps(kwargs)}"
        
        self.logger.log(level, message, exc_info=exc_info)
    
    def log_request(self, method: str, path: str, **kwargs):
        """Log de request HTTP"""
        self.debug(
            f"Request: {method} {path}",
            method=method,
            path=path,
            **kwargs
        )
    
    def log_response(self, status_code: int, duration: float, **kwargs):
        """Log de response HTTP"""
        self.debug(
            f"Response: {status_code} ({duration:.4f}s)",
            status_code=status_code,
            duration=duration,
            **kwargs
        )
    
    def log_exception(self, exception: Exception, context: Optional[Dict] = None):
        """Log de excepción con contexto"""
        self.error(
            f"Exception: {type(exception).__name__}: {str(exception)}",
            exc_info=True,
            exception_type=type(exception).__name__,
            exception_message=str(exception),
            context=context or {}
        )
    
    def log_service_call(
        self,
        service_name: str,
        method_name: str,
        duration: float,
        success: bool,
        **kwargs
    ):
        """Log de llamada a servicio"""
        level = logging.INFO if success else logging.ERROR
        self.logger.log(
            level,
            f"Service call: {service_name}.{method_name} ({duration:.4f}s) - {'SUCCESS' if success else 'FAILED'}",
            service=service_name,
            method=method_name,
            duration=duration,
            success=success,
            **kwargs
        )
    
    def log_sql_query(self, query: str, duration: float, **kwargs):
        """Log de query SQL"""
        self.debug(
            f"SQL Query: {query[:100]}... ({duration:.4f}s)",
            query=query,
            duration=duration,
            **kwargs
        )
    
    def log_cache_operation(self, operation: str, key: str, hit: bool, **kwargs):
        """Log de operación de cache"""
        self.debug(
            f"Cache {operation}: {key} - {'HIT' if hit else 'MISS'}",
            operation=operation,
            key=key,
            hit=hit,
            **kwargs
        )


# Instancia global
_debug_logger: Optional[DebugLogger] = None


def get_debug_logger(
    name: str = "debug",
    level: int = logging.DEBUG,
    enable_file_logging: bool = True
) -> DebugLogger:
    """Obtiene instancia de debug logger"""
    global _debug_logger
    if _debug_logger is None:
        _debug_logger = DebugLogger(
            name=name,
            level=level,
            enable_file_logging=enable_file_logging
        )
    return _debug_logger















