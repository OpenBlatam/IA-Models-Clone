"""
Helpers para rate limiting optimizado

Incluye funciones para manejar rate limiting de forma eficiente.
"""

import time
from typing import Dict, Optional
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

# Cache simple para rate limiting (en producción usar Redis)
_rate_limit_cache: Dict[str, Dict[str, float]] = defaultdict(dict)


def check_rate_limit(
    identifier: str,
    max_requests: int,
    window_seconds: int
) -> tuple[bool, Optional[int]]:
    """
    Verifica si un identificador ha excedido el rate limit.
    
    Args:
        identifier: Identificador único (user_id, IP, etc.)
        max_requests: Número máximo de requests permitidos
        window_seconds: Ventana de tiempo en segundos
        
    Returns:
        Tupla (is_allowed, retry_after_seconds)
        - is_allowed: True si está permitido, False si excedió el límite
        - retry_after_seconds: Segundos hasta que se puede hacer otro request (None si está permitido)
    """
    current_time = time.time()
    cache_key = f"{identifier}:{window_seconds}"
    
    # Obtener historial de requests
    if cache_key not in _rate_limit_cache:
        _rate_limit_cache[cache_key] = {}
    
    request_times = _rate_limit_cache[cache_key]
    
    # Limpiar requests antiguos fuera de la ventana
    cutoff_time = current_time - window_seconds
    request_times = {
        req_time: count
        for req_time, count in request_times.items()
        if req_time > cutoff_time
    }
    _rate_limit_cache[cache_key] = request_times
    
    # Contar requests en la ventana
    total_requests = sum(request_times.values())
    
    if total_requests >= max_requests:
        # Calcular tiempo hasta el request más antiguo expire
        oldest_request = min(request_times.keys()) if request_times else current_time
        retry_after = int(window_seconds - (current_time - oldest_request)) + 1
        return False, retry_after
    
    # Registrar nuevo request
    request_times[current_time] = request_times.get(current_time, 0) + 1
    _rate_limit_cache[cache_key] = request_times
    
    return True, None


def get_rate_limit_info(
    identifier: str,
    max_requests: int,
    window_seconds: int
) -> Dict[str, int]:
    """
    Obtiene información sobre el rate limit de un identificador.
    
    Args:
        identifier: Identificador único
        max_requests: Número máximo de requests permitidos
        window_seconds: Ventana de tiempo en segundos
        
    Returns:
        Diccionario con información del rate limit
    """
    current_time = time.time()
    cache_key = f"{identifier}:{window_seconds}"
    
    if cache_key not in _rate_limit_cache:
        return {
            "remaining": max_requests,
            "limit": max_requests,
            "reset_in": window_seconds
        }
    
    request_times = _rate_limit_cache[cache_key]
    cutoff_time = current_time - window_seconds
    request_times = {
        req_time: count
        for req_time, count in request_times.items()
        if req_time > cutoff_time
    }
    
    total_requests = sum(request_times.values())
    remaining = max(0, max_requests - total_requests)
    
    oldest_request = min(request_times.keys()) if request_times else current_time
    reset_in = int(window_seconds - (current_time - oldest_request)) + 1
    
    return {
        "remaining": remaining,
        "limit": max_requests,
        "reset_in": max(0, reset_in)
    }


def clear_rate_limit(identifier: Optional[str] = None) -> None:
    """
    Limpia el rate limit cache.
    
    Args:
        identifier: Identificador específico (opcional, limpia todo si es None)
    """
    if identifier:
        keys_to_remove = [key for key in _rate_limit_cache.keys() if key.startswith(identifier)]
        for key in keys_to_remove:
            _rate_limit_cache.pop(key, None)
    else:
        _rate_limit_cache.clear()

