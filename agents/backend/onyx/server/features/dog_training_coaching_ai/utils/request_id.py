"""
Request ID Utilities
====================
Utilidades para generar y rastrear IDs de requests.
"""

import uuid
from typing import Optional
from fastapi import Request


def generate_request_id() -> str:
    """Generar un ID único para el request."""
    return str(uuid.uuid4())


def get_request_id(request: Request) -> str:
    """
    Obtener o generar un request ID.
    
    Args:
        request: Request de FastAPI
        
    Returns:
        Request ID
    """
    # Intentar obtener del header
    request_id = request.headers.get("X-Request-ID")
    
    if not request_id:
        # Generar uno nuevo
        request_id = generate_request_id()
        # Agregar al state para uso posterior
        request.state.request_id = request_id
    
    return request_id

