"""
Format Utils

Utilities for format utils.
"""

from typing import Any, Dict, Optional
from datetime import datetime

def format_error_response(
    error_code: str,
    message: str,
    details: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Formatear respuesta de error"""
    response = {
        "error": {
            "code": error_code,
            "message": message,
            "timestamp": datetime.utcnow().isoformat()
        }
    }
    if details:
        response["error"]["details"] = details
    return response

def format_success_response(
    data: Any,
    message: Optional[str] = None
) -> Dict[str, Any]:
    """Formatear respuesta exitosa"""
    response = {
        "success": True,
        "data": data,
        "timestamp": datetime.utcnow().isoformat()
    }
    if message:
        response["message"] = message
    return response

def format_bytes(bytes_count: int) -> str:
    """Formatear bytes en formato legible"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_count < 1024.0:
            return f"{bytes_count:.2f} {unit}"
        bytes_count /= 1024.0
    return f"{bytes_count:.2f} PB"

