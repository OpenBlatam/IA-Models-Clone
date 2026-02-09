"""
Helpers para procesamiento de requests

Incluye funciones para procesar y optimizar requests HTTP.
"""

from typing import Optional, Dict, Any
from fastapi import Request, Response
import logging

logger = logging.getLogger(__name__)


def get_client_ip(request: Request) -> str:
    """
    Obtiene la IP del cliente desde el request.
    
    Args:
        request: Request object de FastAPI
        
    Returns:
        IP del cliente
    """
    # Intentar obtener IP real si hay proxy
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip
    
    # Fallback a client host
    if request.client:
        return request.client.host
    
    return "unknown"


def get_user_agent(request: Request) -> str:
    """
    Obtiene el User-Agent del request.
    
    Args:
        request: Request object de FastAPI
        
    Returns:
        User-Agent string
    """
    return request.headers.get("User-Agent", "unknown")


def get_accept_encoding(request: Request) -> Optional[str]:
    """
    Obtiene el header Accept-Encoding del request.
    
    Args:
        request: Request object de FastAPI
        
    Returns:
        Accept-Encoding header o None
    """
    return request.headers.get("Accept-Encoding")


def add_cache_headers(
    response: Response,
    max_age: int = 60,
    public: bool = True,
    must_revalidate: bool = False
) -> None:
    """
    Agrega headers de cache a una respuesta.
    
    Args:
        response: Response object
        max_age: Tiempo máximo de cache en segundos
        public: Si el cache es público
        must_revalidate: Si se debe revalidar
    """
    cache_control = []
    
    if public:
        cache_control.append("public")
    else:
        cache_control.append("private")
    
    cache_control.append(f"max-age={max_age}")
    
    if must_revalidate:
        cache_control.append("must-revalidate")
    
    response.headers["Cache-Control"] = ", ".join(cache_control)


def add_cors_headers(
    response: Response,
    origin: Optional[str] = None,
    allow_methods: Optional[str] = None,
    allow_headers: Optional[str] = None
) -> None:
    """
    Agrega headers CORS a una respuesta.
    
    Args:
        response: Response object
        origin: Origin permitido
        allow_methods: Métodos permitidos
        allow_headers: Headers permitidos
    """
    if origin:
        response.headers["Access-Control-Allow-Origin"] = origin
    
    if allow_methods:
        response.headers["Access-Control-Allow-Methods"] = allow_methods
    
    if allow_headers:
        response.headers["Access-Control-Allow-Headers"] = allow_headers
    
    response.headers["Access-Control-Allow-Credentials"] = "true"


def get_request_metadata(request: Request) -> Dict[str, Any]:
    """
    Obtiene metadatos del request para logging y análisis.
    
    Args:
        request: Request object de FastAPI
        
    Returns:
        Diccionario con metadatos del request
    """
    return {
        "method": request.method,
        "url": str(request.url),
        "path": request.url.path,
        "query_params": dict(request.query_params),
        "client_ip": get_client_ip(request),
        "user_agent": get_user_agent(request),
        "accept_encoding": get_accept_encoding(request),
        "content_type": request.headers.get("Content-Type"),
        "content_length": request.headers.get("Content-Length")
    }

