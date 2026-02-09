"""
Middleware System
================

Sistema de middleware para procesamiento de requests.
"""

from typing import Callable, Any, Optional, Dict
from functools import wraps
import time
import logging

logger = logging.getLogger(__name__)


class Middleware:
    """
    Clase base para middleware.
    
    Los middleware procesan requests/responses en una cadena.
    """
    
    def __init__(self, name: str):
        """
        Inicializar middleware.
        
        Args:
            name: Nombre del middleware
        """
        self.name = name
        self.enabled = True
    
    def process_request(self, request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Procesar request.
        
        Args:
            request: Datos del request
            
        Returns:
            Request modificado o None para detener procesamiento
        """
        return request
    
    def process_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Procesar response.
        
        Args:
            response: Datos del response
            
        Returns:
            Response modificado
        """
        return response
    
    def process_error(self, error: Exception) -> Optional[Dict[str, Any]]:
        """
        Procesar error.
        
        Args:
            error: Excepción
            
        Returns:
            Response de error o None
        """
        return None


class MiddlewareChain:
    """
    Cadena de middleware.
    
    Ejecuta middleware en orden.
    """
    
    def __init__(self):
        """Inicializar cadena de middleware."""
        self.middlewares: List[Middleware] = []
    
    def add(self, middleware: Middleware) -> None:
        """Agregar middleware a la cadena."""
        self.middlewares.append(middleware)
        logger.debug(f"Added middleware: {middleware.name}")
    
    def process(
        self,
        request: Dict[str, Any],
        handler: Callable[[Dict[str, Any]], Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Procesar request a través de la cadena.
        
        Args:
            request: Datos del request
            handler: Función handler final
            
        Returns:
            Response
        """
        # Procesar requests
        for middleware in self.middlewares:
            if not middleware.enabled:
                continue
            
            request = middleware.process_request(request)
            if request is None:
                return {"error": "Request rejected by middleware"}
        
        # Ejecutar handler
        try:
            response = handler(request)
        except Exception as e:
            # Procesar error
            for middleware in reversed(self.middlewares):
                if not middleware.enabled:
                    continue
                
                error_response = middleware.process_error(e)
                if error_response:
                    return error_response
            
            raise
        
        # Procesar responses
        for middleware in reversed(self.middlewares):
            if not middleware.enabled:
                continue
            
            response = middleware.process_response(response)
        
        return response


class LoggingMiddleware(Middleware):
    """Middleware para logging de requests."""
    
    def __init__(self):
        """Inicializar middleware de logging."""
        super().__init__("logging")
    
    def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Loggear request."""
        request["_start_time"] = time.time()
        logger.info(f"Request: {request.get('action', 'unknown')}")
        return request
    
    def process_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Loggear response."""
        duration = time.time() - response.get("_start_time", time.time())
        logger.info(f"Response: {response.get('status', 'unknown')} ({duration:.3f}s)")
        return response


class MetricsMiddleware(Middleware):
    """Middleware para métricas."""
    
    def __init__(self):
        """Inicializar middleware de métricas."""
        super().__init__("metrics")
    
    def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Registrar métrica de request."""
        from .metrics import increment_counter
        increment_counter("requests.total")
        action = request.get("action", "unknown")
        increment_counter(f"requests.{action}")
        return request
    
    def process_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Registrar métrica de response."""
        from .metrics import record_timing, increment_counter
        
        duration = time.time() - response.get("_start_time", time.time())
        record_timing("request.duration", duration)
        
        status = response.get("status", "unknown")
        increment_counter(f"responses.{status}")
        
        return response


class ValidationMiddleware(Middleware):
    """Middleware para validación."""
    
    def __init__(self, validator: Callable[[Dict[str, Any]], bool]):
        """
        Inicializar middleware de validación.
        
        Args:
            validator: Función de validación
        """
        super().__init__("validation")
        self.validator = validator
    
    def process_request(self, request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Validar request."""
        if not self.validator(request):
            logger.warning("Request validation failed")
            return None
        return request


def middleware_decorator(middleware: Middleware):
    """
    Decorador para aplicar middleware a función.
    
    Args:
        middleware: Middleware a aplicar
        
    Usage:
        @middleware_decorator(LoggingMiddleware())
        def my_handler(request):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(request: Dict[str, Any]) -> Dict[str, Any]:
            # Procesar request
            processed_request = middleware.process_request(request)
            if processed_request is None:
                return {"error": "Request rejected"}
            
            # Ejecutar función
            try:
                response = func(processed_request)
            except Exception as e:
                error_response = middleware.process_error(e)
                if error_response:
                    return error_response
                raise
            
            # Procesar response
            return middleware.process_response(response)
        
        return wrapper
    return decorator






