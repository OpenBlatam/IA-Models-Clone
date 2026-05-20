"""
Helpers para respuestas HTTP (optimizado)

Incluye funciones para construir respuestas consistentes y agregar headers útiles.
"""

from typing import Optional, Dict, Any
from fastapi import Response
from datetime import datetime


def add_cache_headers(
    response: Response,
    max_age: int = 60,
    public: bool = True,
    must_revalidate: bool = False
) -> Response:
    """
    Agrega headers de cache a una respuesta (optimizado).
    
    Args:
        response: Response object
        max_age: Tiempo máximo de cache en segundos
        public: Si el cache es público
        private: Si el cache es privado
        must_revalidate: Si debe revalidar antes de usar cache
        
    Returns:
        Response con headers de cache
    """
    cache_type = "public" if public else "private"
    cache_control = f"{cache_type}, max-age={max_age}"
    
    if must_revalidate:
        cache_control += ", must-revalidate"
    
    response.headers["Cache-Control"] = cache_control
    response.headers["X-Cache-TTL"] = str(max_age)
    
    return response


def add_cors_headers(
    response: Response,
    origin: Optional[str] = None,
    methods: Optional[list] = None,
    headers: Optional[list] = None
) -> Response:
    """
    Agrega headers CORS a una respuesta (optimizado).
    
    Args:
        response: Response object
        origin: Origen permitido
        methods: Métodos permitidos
        headers: Headers permitidos
        
    Returns:
        Response con headers CORS
    """
    if origin:
        response.headers["Access-Control-Allow-Origin"] = origin
    
    if methods:
        response.headers["Access-Control-Allow-Methods"] = ", ".join(methods)
    
    if headers:
        response.headers["Access-Control-Allow-Headers"] = ", ".join(headers)
    
    return response


def add_security_headers(response: Response) -> Response:
    """
    Agrega headers de seguridad a una respuesta (optimizado).
    
    Args:
        response: Response object
        
    Returns:
        Response con headers de seguridad
    """
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    
    return response


def create_error_response(
    message: str,
    status_code: int = 400,
    details: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Crea una respuesta de error consistente (optimizado).
    
    Args:
        message: Mensaje de error
        status_code: Código de estado HTTP
        details: Detalles adicionales (opcional)
        
    Returns:
        Diccionario con respuesta de error
    """
    response = {
        "error": True,
        "message": message,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    if details:
        response["details"] = details
    
    return response


def create_success_response(
    data: Any,
    message: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Crea una respuesta de éxito consistente (optimizado).
    
    Args:
        data: Datos de la respuesta
        message: Mensaje opcional
        metadata: Metadatos adicionales (opcional)
        
    Returns:
        Diccionario con respuesta de éxito
    """
    response = {
        "success": True,
        "data": data,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    if message:
        response["message"] = message
    
    if metadata:
        response["metadata"] = metadata
    
    return response


def add_pagination_headers(
    response: Response,
    page: int,
    page_size: int,
    total: int
) -> Response:
    """
    Agrega headers de paginación a una respuesta (optimizado).
    
    Args:
        response: Response object
        page: Página actual
        page_size: Tamaño de página
        total: Total de items
        
    Returns:
        Response con headers de paginación
    """
    total_pages = (total + page_size - 1) // page_size
    
    response.headers["X-Page"] = str(page)
    response.headers["X-Page-Size"] = str(page_size)
    response.headers["X-Total"] = str(total)
    response.headers["X-Total-Pages"] = str(total_pages)
    response.headers["X-Has-Next"] = str(page < total_pages)
    response.headers["X-Has-Prev"] = str(page > 1)
    
    return response

