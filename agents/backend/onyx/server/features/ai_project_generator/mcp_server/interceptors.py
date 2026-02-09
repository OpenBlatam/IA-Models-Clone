"""
MCP Interceptors - Interceptores de request/response
=====================================================
"""

import logging
from typing import Callable, Any, Dict, Optional, List
from functools import wraps

logger = logging.getLogger(__name__)


class RequestInterceptor:
    """
    Interceptor de requests
    
    Permite interceptar y modificar requests antes de procesarlos.
    """
    
    def __init__(self):
        self._interceptors: List[Callable] = []
    
    def register(self, interceptor: Callable[[Dict[str, Any]], Dict[str, Any]]):
        """
        Registra un interceptor
        
        Args:
            interceptor: Función que intercepta el request
        """
        self._interceptors.append(interceptor)
        logger.info(f"Registered request interceptor: {interceptor.__name__}")
    
    def intercept(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Aplica todos los interceptores
        
        Args:
            request: Request original
            
        Returns:
            Request interceptado
        """
        result = request
        for interceptor in self._interceptors:
            try:
                result = interceptor(result)
            except Exception as e:
                logger.error(f"Error in request interceptor: {e}")
        return result


class ResponseInterceptor:
    """
    Interceptor de responses
    
    Permite interceptar y modificar responses después de procesarlos.
    """
    
    def __init__(self):
        self._interceptors: List[Callable] = []
    
    def register(self, interceptor: Callable[[Dict[str, Any]], Dict[str, Any]]):
        """
        Registra un interceptor
        
        Args:
            interceptor: Función que intercepta el response
        """
        self._interceptors.append(interceptor)
        logger.info(f"Registered response interceptor: {interceptor.__name__}")
    
    def intercept(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Aplica todos los interceptores
        
        Args:
            response: Response original
            
        Returns:
            Response interceptado
        """
        result = response
        for interceptor in self._interceptors:
            try:
                result = interceptor(result)
            except Exception as e:
                logger.error(f"Error in response interceptor: {e}")
        return result


# Interceptores comunes

def logging_interceptor(request: Dict[str, Any]) -> Dict[str, Any]:
    """Interceptor que agrega logging"""
    logger.info(f"Intercepted request: {request.get('resource_id')}")
    return request


def timing_interceptor(response: Dict[str, Any]) -> Dict[str, Any]:
    """
    Interceptor que agrega timing timezone-aware.
    
    Args:
        response: Diccionario de respuesta
        
    Returns:
        Diccionario con timestamp agregado
    """
    from datetime import datetime, timezone
    response["intercepted_at"] = datetime.now(timezone.utc).isoformat()
    return response


def validation_interceptor(request: Dict[str, Any]) -> Dict[str, Any]:
    """Interceptor que valida request"""
    if not request.get("resource_id"):
        raise ValueError("resource_id is required")
    return request

